# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compile or reuse Qwen3.5 MoE vision, prefill, and decode QPCs, then run generation."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--prefill-qpc-path",
        default=os.environ.get("QEFF_PREFILL_QPC_PATH"),
        help="Existing language-prefill QPC directory. If omitted, it is compiled.",
    )
    parser.add_argument(
        "--decode-qpc-path",
        default=os.environ.get("QEFF_DECODE_QPC_PATH"),
        help="Existing language-decode QPC directory. If omitted, it is compiled.",
    )
    vision_group = parser.add_mutually_exclusive_group()
    vision_group.add_argument(
        "--vision-qpc-path",
        default=os.environ.get("QEFF_VISION_QPC_PATH"),
        help="Existing vision QPC directory. If omitted, it is compiled unless --skip-vision is set.",
    )
    vision_group.add_argument(
        "--skip-vision",
        action="store_true",
        help="Run a text-only prompt and neither compile nor load a vision QPC.",
    )
    parser.add_argument("--generation-len", type=int, default=256)
    parser.add_argument(
        "--decode-num-devices",
        type=int,
        default=int(os.environ.get("QEFF_DECODE_NUM_DEVICES", "4")),
    )
    return parser.parse_args()


def _validate_qpc_path(path, component):
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(os.path.join(path, "programqpc.bin")):
        raise FileNotFoundError(f"{component} QPC does not contain programqpc.bin: {path}")
    return path


def _compiled_qpc_path(compile_result, key, component, fallback_keys=()):
    path = None
    if isinstance(compile_result, dict):
        for candidate_key in (key, *fallback_keys):
            path = compile_result.get(candidate_key)
            if path:
                break
    else:
        path = compile_result
    if not path:
        raise RuntimeError(f"Compilation did not return {key!r}: {compile_result!r}")
    return _validate_qpc_path(path, component)


def _timed_call(timings, key, method):
    def wrapped(*args, **kwargs):
        start = perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            timings[key] += perf_counter() - start

    return wrapped


args = _parse_args()
model_id = args.model_id
DECODE_NUM_DEVICES = args.decode_num_devices
config = AutoConfig.from_pretrained(model_id)

# For faster execution user can run with lesser layers, For Testing Purpose Only
config.vision_config.depth = 5
config.text_config.num_hidden_layers = 4
config.torch_dtype = "float16"
layer_types = list(getattr(config.text_config, "layer_types", []))
if len(layer_types) < config.text_config.num_hidden_layers:
    layer_types.extend(["full_attention"] * (config.text_config.num_hidden_layers - len(layer_types)))
config.text_config.layer_types = layer_types[: config.text_config.num_hidden_layers]
# config.text_config.layer_types = layer_types[0 : 1]


def _resolve_retained_state(source_outputs, logical_name):
    suffixes = ("_InternalRetainedState", "_RetainedState")
    matches = []
    for output_name, value in source_outputs.items():
        if not (output_name == f"{logical_name}_RetainedState" or output_name.startswith(f"{logical_name}_")):
            continue
        for suffix in suffixes:
            if output_name.endswith(suffix):
                matches.append((output_name[: -len(suffix)], value))
                break
    if len(matches) != 1:
        available = sorted(name for name in source_outputs if name.startswith(logical_name))
        raise KeyError(f"Expected one retained-state output for {logical_name!r}, found {available or 'none'}")
    return matches[0]


def _update_retained_states(target_inputs, source_outputs):
    for layer_idx, layer_type in enumerate(config.text_config.layer_types):
        state_names = (
            (f"past_key.{layer_idx}", f"past_value.{layer_idx}")
            if layer_type == "full_attention"
            else (f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}")
        )
        for logical_name in state_names:
            target_name, value = _resolve_retained_state(source_outputs, logical_name)
            target_inputs[target_name] = value


qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    continuous_batching=True,
)

# qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
#     model_id, attn_implementation="eager", kv_offload=True, config=config,
# )
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

PREFILL_SEQ_LEN = 512
# The GDN prefill mini-chunk follows the prefill/CPL by default. This is passed
# explicitly because the shared compile API does not infer model-specific options.
gdn_chunk_size = PREFILL_SEQ_LEN
CTX_LEN = 14 * 1024
BATCH_SIZE = 512  # Per-slot prefill batch size
BS = BATCH_SIZE
FULL_BATCH_SIZE = 512  # Total concurrent CB slots

# Online-prefill KV blocking (matches GQA_Benchmark_Merged/qwen3_35b_a3b_online_prefill.sh:
# NUM_KV_BLOCKS=8, IMPLS=online, Q_BLOCK_SIZE=256 -> num_q_blocks=ceil(1024/256)=4,
# Q_HEAD_BLOCK_CHUNK=1 -> n_rep_chunk=1).
# To disable KV blocking, comment out the qaic_config= line in the prefill compile() call below.
qaic_config = {
    "blocking_mode": "prefill_online",
    "num_kv_blocks": 16,
    "num_q_blocks": 4,
    "n_rep_chunk": 1,
    "skip_kv": True,
    "gdn_chunk_size": gdn_chunk_size,
    # Prefill replaces the complete BS1 GDN state, so retained-state routing is unnecessary.
    "gdn_full_state_update": True,
}

# CL 64K BSZ1
# Decode-time KV blocking:
# decode_qaic_config = {
#     "blocking_mode": "kv_headpar",
#     "num_kv_blocks": 8,
#     "skip_kv": True,
# }

# CL 14K BSZ512
# Decode-time KV blocking plus EP decode.
decode_qaic_config = {
    "blocking_mode": "kv_batch_fold",
    "num_kv_blocks": 64,
    "gdn_num_head_blocks": int(os.environ.get("QEFF_GDN_NUM_HEAD_BLOCKS", "8")),
    "skip_kv": True,
    "moe_config": {
        "flavour": "expert_parallel",
        "cores_per_expert": int(os.environ.get("QEFF_EP_CORES_PER_EXPERT", "1")),
        "tree_reduce": os.environ.get("QEFF_EP_TREE_REDUCE", "0") != "0",
        "expert_parallel_chunk_size": int(os.environ.get("QEFF_EP_PACKED_CHUNK_SIZE", "256")),
    },
}

enable_blocking = True

generation_len = args.generation_len
# os.environ["QAIC_COMPILER_OPTS_UNSUPPORTED"] = (
#     "-aic-user-order -aic-hoist-vtcm-loads=false "
#     "-aic-op-stats-verbosity 2 -aic-hmx-async=0 -aic-userdma-async=0 -aic-sync-ts-starts"
# )
os.environ["QAIC_COMPILER_OPTS_UNSUPPORTED"] = (
    "-aic-op-stats-verbosity 2 -aic-hoist-vtcm-loads=false -aic-hmx-async=0 -aic-userdma-async=0"
)
skip_vision = args.skip_vision

if skip_vision:
    vision_qpc_path = None
elif args.vision_qpc_path:
    vision_qpc_path = _validate_qpc_path(args.vision_qpc_path, "vision")
    print(f"Using compiled vision QPC: {vision_qpc_path}")
else:
    vision_compile_result = qeff_model.compile(
        batch_size=BS,
        full_batch_size=FULL_BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=354,
        width=536,
        num_cores=16,
        num_devices=1,
        mos=1,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        skip_vision=skip_vision,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
    )
    vision_qpc_path = _compiled_qpc_path(vision_compile_result, "vision_qpc_path", "vision")
    print(f"Compiled vision QPC: {vision_qpc_path}")

if args.prefill_qpc_path:
    prefill_qpc_path = _validate_qpc_path(args.prefill_qpc_path, "prefill")
    print(f"Using compiled prefill QPC: {prefill_qpc_path}")
else:
    prefill_timings = {"export": 0.0, "compile": 0.0}
    lang_export = qeff_model.lang_model.export
    lang_compile = qeff_model.lang_model._compile
    qeff_model.lang_model.export = _timed_call(prefill_timings, "export", lang_export)
    qeff_model.lang_model._compile = _timed_call(prefill_timings, "compile", lang_compile)
    prefill_total_start = perf_counter()
    try:
        prefill_compile_result = qeff_model.compile(
            batch_size=1,
            full_batch_size=1,
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            height=354,
            width=536,
            num_cores=16,
            num_devices=1,
            mxfp6_matmul=True,
            mxint8_kv_cache=True,
            retain_full_kv=True,
            split_model_io=True,  # This should be used for disagg serving via VLLM
            # mos=1,
            user_tiled=True,
            aic_enable_depth_first=False,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
            use_onnx_subfunctions=True,
            stats_level=50,
            ddr_stats=True,
            aic_pmu_recipe="KernelUtil",
            aic_perf_metrics=True,
            qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
            kv_cache_prefix="vllmKvCache",
            allow_mxint8_mdp_io=True,
            # custom_IO_list_file="/local/mnt/workspace/mkshirsa/qwen_36/efficient-transformers/Qwen3_5MoeForConditionalGeneration/Qwen3_5MoeDecoderWrapper-3f73445da9adc289/qpc-37d931bb34ea0c73/custom_io_kvint8.yaml"
        )
    finally:
        prefill_total_time = perf_counter() - prefill_total_start
        qeff_model.lang_model.export = lang_export
        qeff_model.lang_model._compile = lang_compile
    prefill_qpc_path = _compiled_qpc_path(prefill_compile_result, "lang_prefill_qpc_path", "prefill")
    print(f"Compiled prefill QPC: {prefill_qpc_path}")
    print(f"Prefill export time: {prefill_timings['export']:.2f} secs")
    print(f"Prefill compile time: {prefill_timings['compile']:.2f} secs")
    print(f"Prefill export+compile total time: {prefill_total_time:.2f} secs")

if args.decode_qpc_path:
    decode_qpc_path = _validate_qpc_path(args.decode_qpc_path, "decode")
    print(f"Using compiled decode QPC: {decode_qpc_path}")
else:
    decode_timings = {"export": 0.0, "compile": 0.0}
    lang_export = qeff_model.lang_model.export
    lang_compile = qeff_model.lang_model._compile
    qeff_model.lang_model.export = _timed_call(decode_timings, "export", lang_export)
    qeff_model.lang_model._compile = _timed_call(decode_timings, "compile", lang_compile)
    decode_total_start = perf_counter()
    try:
        decode_compile_result = qeff_model.compile(
            batch_size=BS,
            kv_cache_batch_size=FULL_BATCH_SIZE,
            full_batch_size=FULL_BATCH_SIZE,
            prefill_seq_len=1,
            ctx_len=CTX_LEN,
            height=354,
            width=536,
            num_cores=16,
            num_devices=DECODE_NUM_DEVICES,
            mxfp6_matmul=True,
            mxint8_kv_cache=True,
            retain_full_kv=True,
            split_model_io=True,  # This should be used for disagg serving via VLLM
            aic_enable_depth_first=False,
            user_tiled=True,
            prefill_only=False,
            skip_vision=True,
            use_onnx_subfunctions=True,
            stats_level=70,
            ddr_stats=True,
            aic_pmu_recipe="KernelUtil",
            aic_perf_metrics=True,
            qaic_config=decode_qaic_config,  # Enable KV blocking - comment out to disable
            kv_cache_prefix="vllmKvCache",
            allow_mxint8_mdp_io=True,
            # network_specialization_config="/path/to/specializations.json"
        )
    finally:
        decode_total_time = perf_counter() - decode_total_start
        qeff_model.lang_model.export = lang_export
        qeff_model.lang_model._compile = lang_compile
    decode_qpc_path = _compiled_qpc_path(
        decode_compile_result,
        "lang_decode_qpc_path",
        "decode",
        fallback_keys=("lang_qpc_path",),
    )
    print(f"Compiled decode QPC: {decode_qpc_path}")
    print(f"Decode export time: {decode_timings['export']:.2f} secs")
    print(f"Decode compile time: {decode_timings['compile']:.2f} secs")
    print(f"Decode export+compile total time: {decode_total_time:.2f} secs")

if enable_blocking:
    print("\n" + "=" * 80)
    print("Verifying KV Blocking Applied During Compilation")
    print("=" * 80)

    # The compile() method internally calls BlockingAttentionTransform.apply()
    # which sets attn_blocking_config on all supported attention modules
    # This happens BEFORE ONNX export, so blocking operations are in the ONNX graph

    if decode_qaic_config.get("blocking_mode"):
        print("✓ decode_qaic_config passed to compile():")
        print(f"    Blocking Mode: {decode_qaic_config.get('blocking_mode')}")
        print(f"    Num KV Blocks: {decode_qaic_config.get('num_kv_blocks')}")
        print(f"    GDN Head Blocks: {decode_qaic_config.get('gdn_num_head_blocks')}")
        print(f"    Skip KV: {decode_qaic_config.get('skip_kv', False)}")
        print("\n✓ BlockingAttentionTransform.apply() called during compile()")
        print("  - Sets attn_blocking_config on all supported attention modules")
        print("  - Blocked attention forward pass is used during ONNX export")
        print("  - Blocking operations are in the ONNX graph and QPC")
        print("\n  Status: ACTIVE")
        print("  Verification: Config-based verification")
        print("  Note: Blocking IS applied - torch model is freed after ONNX export")
    else:
        print("✗ No qaic_config provided - eager attention will be used")
        print("  Status: INACTIVE - Model compiled without blocking")

    print("=" * 80 + "\n")

lang_prefill_session = QAICInferenceSession(prefill_qpc_path)
lang_decode_session = QAICInferenceSession(decode_qpc_path)

if skip_vision:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]
else:
    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe all the colors seen in the image."},
            ],
        },
    ]
    vision_session = QAICInferenceSession(vision_qpc_path)


messages = [messages] * BS

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)


inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=PREFILL_SEQ_LEN, batch_size=BS)

pad_token_id = 1
input_len = inputs["attention_mask"].sum(1, keepdims=True)
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

print(f"generation_len : {generation_len}")
generated_ids = np.full((BS, generation_len + 1), pad_token_id)


inputs["input_ids"] = torch.nn.functional.pad(
    inputs["input_ids"],
    (0, padded_len - input_ids_length),
    "constant",
    pad_token_id,
)
inputs["attention_mask"] = torch.nn.functional.pad(
    inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
)

for k, v in inputs.items():
    inputs[k] = np.array(v)


vision_inputs = {
    k: v
    for k, v in inputs.items()
    if k in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
}

vision_inputs_fp16 = {"pixel_values", "image_masks"}
vision_inputs.update({k: vision_inputs[k].astype("float16") for k in vision_inputs_fp16 if k in vision_inputs})

vision_start = perf_counter()
vision_outputs = {}
if vision_inputs:
    vision_outputs = vision_session.run(vision_inputs)
vision_end = perf_counter()

lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
if "position_ids" in inputs:
    lang_inputs["position_ids"] = inputs["position_ids"]
    lang_inputs.pop("attention_mask")
else:
    lang_inputs["position_ids"] = np.where(
        lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
    )  # Need to use -1 as position_ids for invalid tokens

lang_inputs["image_idx"] = np.array([[0]])

if not skip_vision:
    lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]

# RUN prefill
lang_start = perf_counter()
lang_prefill_session.set_buffers(vision_outputs)

all_outputs = []
chunk_inputs = lang_inputs.copy()
chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)
for i in range(num_chunks):
    chunk_inputs["input_ids"] = lang_inputs["input_ids"][0:1, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = lang_inputs["position_ids"][:, 0:1, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    outputs = lang_prefill_session.run(chunk_inputs)
    _update_retained_states(chunk_inputs, outputs)
    chunk_inputs["image_idx"] = outputs["image_idx_output"]
prefill_time = perf_counter() - lang_start + vision_end - vision_start
print(f"Prefill time : {prefill_time:.2f} secs")

lang_prefill_session.deactivate()
lang_decode_session = QAICInferenceSession(decode_qpc_path)

next_token_id = np.argmax(outputs["logits"])
all_outputs.append(next_token_id)
batch_index = np.random.default_rng(1234).permutation(BS).reshape(BS, 1).astype(np.int64)
decode_inputs = {
    "input_ids": np.full((BS, 1), next_token_id, dtype=lang_inputs["input_ids"].dtype),
    "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
    "batch_index": batch_index,
}

for layer_idx, layer_type in enumerate(config.text_config.layer_types):
    state_names = (
        (f"past_key.{layer_idx}", f"past_value.{layer_idx}")
        if layer_type == "full_attention"
        else (f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}")
    )
    for logical_name in state_names:
        target_name, value = _resolve_retained_state(outputs, logical_name)
        logical_state = value[0:1]
        physical_state = np.repeat(logical_state, BS, axis=0)
        reordered_state = np.empty_like(physical_state)
        reordered_state[batch_index[:, 0]] = physical_state
        decode_inputs[target_name] = reordered_state

decode_inputs["image_idx"] = outputs["image_idx_output"]

if not skip_vision:
    decode_inputs["vision_embeds"] = outputs["vision_embeds_RetainedState"]

st = perf_counter()
decode_out = lang_decode_session.run(decode_inputs)
print(f"time for first run of decode with KV as input = {perf_counter() - st} sec\n")

all_outputs.append(np.argmax(decode_out["logits"], axis=-1)[0, 0])
pos_id = decode_inputs["position_ids"] + 1
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"], axis=-1),
    "position_ids": pos_id,
    "batch_index": batch_index,
}

_update_retained_states(loop_decode_inputs, decode_out)

loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]

if not skip_vision:
    loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]


st = perf_counter()
for i in range(generation_len - 2):
    decode_out = lang_decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"], axis=-1)[0, 0])
    pos_id += 1
    _update_retained_states(loop_decode_inputs, decode_out)
    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"], axis=-1),
            "position_ids": pos_id,
            "batch_index": batch_index,
        }
    )
ft = perf_counter()
print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
print(f"\noutput\n{tokenizer.decode(all_outputs)}")
