# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""DiffusionGemma unified, split, or staged-prefill execution on Cloud AI 100."""

import argparse

from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.transformers.models.diffusion_gemma_single_qpc_example_utils import (
    build_step_callback,
    clean_diffusion_text,
    compile_split_qpcs,
    compile_unified_qpc,
    diffusion_gemma_prefill_split_qpc_chunked,
    prepare_prompt_inputs,
)

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
CTX_LEN = 1024
CANVAS_LENGTH = 256
DIFFUSION_STEPS = 48
NUM_CORES = 16
NUM_DEVICES = 4
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
IMAGE_PROMPT = "Describe this image in detail in not less than 512 tokens"
# TEXT_PROMPT = "What is the capital city of Zimbabwe? Answer in one sentence."
# TEXT_PROMPT = "What are the seven continents? Answer in one sentence."
# TEXT_PROMPT = "Can you summarize Romeo and Juliet story by William Shakespere? Explain in more detail with answer no less than 1280 tokens"
TEXT_PROMPT = "What is diffusion based generative learning?"
# TEXT_PROMPT = "How to make pizza?"
# TEXT_PROMPT = "What is diffusion based generative learning? Answer in one sentence."


def _apply_reduced_layer_config(config, num_lang_layers: int):
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        config.text_config.num_hidden_layers = num_lang_layers
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_lang_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]
    return config


def load_model_and_processor(model_id: str, canvas_length: int, num_lang_layers: int = None):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if num_lang_layers is not None:
        config = _apply_reduced_layer_config(config, num_lang_layers=num_lang_layers)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=False,
        ignore_mismatched_sizes=num_lang_layers is not None,
    )
    qeff_model.model.config.canvas_length = canvas_length
    return processor, qeff_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run DiffusionGemma unified, split, or staged-prefill inference.")
    parser.add_argument("--runtime-mode", choices=("unified", "split", "prefill"), default="unified")
    parser.add_argument("--text-only", action="store_true", help="Run without image tokens.")
    parser.add_argument("--prompt", help="Override the default image or text prompt.")
    parser.add_argument("--seed", type=int, default=1234, help="Use -1 for an unseeded sampler.")
    parser.add_argument("--ctx-len", type=int, default=CTX_LEN, help="Compiled retained-KV context length.")
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH, help="Tokens per denoising canvas.")
    parser.add_argument(
        "--prefill-stages",
        type=int,
        default=1,
        help="Encoder prefill pipeline partitions for split and prefill modes.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Total generated tokens.")
    parser.add_argument("--diffusion-steps", type=int, default=DIFFUSION_STEPS, help="Steps per canvas.")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Use a reduced number of language layers; defaults to the full model.",
    )
    parser.add_argument(
        "--sampler",
        choices=("local", "hf"),
        default="local",
        help="Cumulative local freezing or Hugging Face per-step re-noising.",
    )
    parser.add_argument("--no-stop-on-eos", action="store_true", help="Do not stop at the first EOS token.")
    parser.add_argument("--truncate-first-sentence", action="store_true", help="Return the first sentence only.")
    parser.add_argument("--verbose-steps", action="store_true", help="Decode a preview after each step.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.canvas_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Canvas length and max new tokens must be positive.")
    if args.prefill_stages <= 0:
        raise ValueError("Prefill stages must be positive.")
    if args.num_layers is not None and args.num_layers <= 0:
        raise ValueError("Number of layers must be positive.")
    if args.runtime_mode == "prefill" and not args.text_only:
        raise ValueError("Prefill-only mode requires --text-only.")

    device_ids = None  # [int(device_id) for device_id in os.environ.get("DG", "4,5,6,7").split(",")]
    processor, qeff_model = load_model_and_processor(
        MODEL_ID,
        args.canvas_length,
        num_lang_layers=args.num_layers,
    )
    print(f"Compiling a {qeff_model.model.config.text_config.num_hidden_layers}-layer DiffusionGemma model.")
    if args.runtime_mode == "unified":
        qpc_path = compile_unified_qpc(
            qeff_model.model,
            prefill_seq_len=args.canvas_length,
            ctx_len=args.ctx_len,
            canvas_length=args.canvas_length,
            num_devices=NUM_DEVICES,
            num_cores=NUM_CORES,
        )
        qpc_kwargs = {"qpc_path": qpc_path}
    else:
        encoder_qpc_path, decoder_qpc_path = compile_split_qpcs(
            qeff_model.model,
            prefill_seq_len=args.canvas_length,
            ctx_len=args.ctx_len,
            canvas_length=args.canvas_length,
            num_devices=NUM_DEVICES,
            num_cores=NUM_CORES,
            prefill_stages=args.prefill_stages,
        )
        qpc_kwargs = {
            "encoder_qpc_path": encoder_qpc_path,
            "decoder_qpc_path": decoder_qpc_path,
        }

    prompt = args.prompt or (TEXT_PROMPT if args.text_only else IMAGE_PROMPT)
    inputs = prepare_prompt_inputs(
        processor=processor,
        qeff_model=qeff_model.model,
        model_id=MODEL_ID,
        prompt=prompt,
        text_only=args.text_only,
        image_url=IMAGE_URL,
    )
    print(f"Canvas length is {CANVAS_LENGTH} and input ids is of size {inputs['input_ids'].shape[1]}")
    if args.runtime_mode == "prefill":
        result = diffusion_gemma_prefill_split_qpc_chunked(
            qeff_model=qeff_model,
            inputs=inputs,
            encoder_qpc_path=encoder_qpc_path,
            decoder_qpc_path=decoder_qpc_path,
            device_ids=device_ids,
            seed=args.seed,
            prefill_stages=args.prefill_stages,
        )
        print(f"Prompt length: {result.prompt_length}")
        print(f"Next position: {result.next_position}")
        print(f"Prefill time: {result.total_time:.3f}s")
        print(f"Encoder outputs: {list(result.encoder_outputs)}")
        print(f"Retained KV buffers: {len(result.kv_caches)}")
        for index, cache in enumerate(result.kv_caches):
            print(f"  [{index}] shape={cache.shape} dtype={cache.dtype}")
        print(f"ENCODER_QPC={encoder_qpc_path}")
        print(f"DECODER_QPC={decoder_qpc_path}")
        return

    result = qeff_model.cloud_ai_100_diffusion_generate(
        inputs=inputs,
        generation_len=args.max_new_tokens,
        runtime_mode=args.runtime_mode,
        device_ids=device_ids,
        ctx_len=args.ctx_len,
        max_denoising_steps=args.diffusion_steps,
        sampler=args.sampler,
        seed=args.seed,
        stop_on_eos=not args.no_stop_on_eos,
        step_callback=build_step_callback(processor.tokenizer, args.verbose_steps),
        **({"prefill_stages": args.prefill_stages} if args.runtime_mode == "split" else {}),
        **qpc_kwargs,
    )

    raw_output = processor.tokenizer.decode(result.generated_ids[0].tolist(), skip_special_tokens=True)
    output_text = clean_diffusion_text(
        raw_output,
        truncate_first_sentence=(result.generated_ids.shape[1] <= args.canvas_length or args.truncate_first_sentence),
    )
    clean_token_count = len(processor.tokenizer(output_text, add_special_tokens=False).input_ids)
    print(f"\nOutput:\n{output_text}")
    print(f"\nTTFT: {result.ttft:.2f}s ({result.retained_kv_buffers} KV buffers retained)")
    print(
        f"Average number of steps: {result.total_steps / result.executed_blocks}\n "
        f"Tokens per second: {clean_token_count / result.total_time:.1f}\n"
        f"Tokens per second(TTFT excluded): {clean_token_count / (result.total_time - result.ttft):.1f}\n"
    )
    for name, path in qpc_kwargs.items():
        print(f"\n{name.upper()}={path}")


if __name__ == "__main__":
    main()
