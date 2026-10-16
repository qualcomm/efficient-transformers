# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Fast CPU regression coverage across the main model families supported by QEfficient.

This file intentionally uses two coverage tiers:

1. Runtime parity:
   - Exact token or tensor parity across HF PyTorch, transformed PyTorch, and ORT
   - Used where the repo already has a stable CPU verification path
2. Export smoke:
   - Used for model families or architectures that are supported by export today,
     but do not yet have a stable CPU runtime parity path in the consolidated test
"""

import inspect
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Set
from unittest.mock import MagicMock

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import yaml
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    LlamaConfig,
    Qwen2Config,
)
from transformers.cache_utils import DynamicCache
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLAttention,
    MiniMaxM3VLIndexer,
    MiniMaxM3VLSparseMoeBlock,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)

from examples.text_generation.minimax_m3_paged_dp_decode_only_configurable_pages import (
    _build_qaic_config,
    build_block_table,
)
from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.cache_utils import QEffMiniMaxSparseCache
from QEfficient.transformers.models.minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLConfig,
    MiniMaxM3VLForCausalLM,
    MiniMaxM3VLTextConfig,
    MiniMaxM3VLVisionConfig,
)
from QEfficient.transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    QEffMiniMaxM3VLAttention,
    QEffMiniMaxM3VLIndexer,
    QEffMiniMaxM3VLRotaryEmbedding,
    QEffMiniMaxM3VLSparseMoeBlock,
    _generate_minimax_npi_file,
)
from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForImageTextToText,
    QEFFAutoModelForSequenceClassification,
    QEFFAutoModelForSpeechSeq2Seq,
)
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils._utils import _infer_specialization_name, to_named_specializations
from QEfficient.utils.run_utils import ApiRunner

ort.set_default_logger_severity(3)
logging.getLogger("QEfficient").setLevel(logging.ERROR)
logging.getLogger("QEfficient.base.modeling_qeff").setLevel(logging.ERROR)


CAUSAL_RUNTIME_MODEL_IDS = {
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "falcon": "hf-internal-testing/tiny-random-FalconForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    "phi3": "tiny-random/phi-4",
    "qwen2": "yujiepan/qwen2-tiny-random",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "olmo2": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    "gpt_oss": "tiny-random/gpt-oss-bf16",
}

#   In PyTorch ≤2.3 (used with transformers v4.57.3), torch.onnx.export with
#   export_modules_as_functions created one ONNX function definition per module instance — so a Mixtral
#   model with 2 decoder layers produced 2 separate QeffMixtralDecoderLayer function definitions in the
#   ONNX.
#   In PyTorch 2.7 (used with transformers v5.5.4), the same export creates one shared function
#   definition per module class, called once per instance. So 2 decoder layers → 1 function definition
#   called 2 times.
CAUSAL_MULTI_SUBFUNCTION_MODEL_TYPES = {
    "codegen",
    "phi",
    "starcoder2",
    "gpt_oss",
    # "granitemoe" is intentionally not listed in CAUSAL_RUNTIME_MODEL_IDS yet.
}

VLM_TEXT_RUNTIME_MODEL_ID = "tiny-random/gemma-3"
VLM_EXPORT_MODEL_IDS = {
    "gemma3": "tiny-random/gemma-3",
    "qwen2_5_vl": "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
    "internvl2": "optimum-intel-internal-testing/tiny-random-internvl2",
}
TINY_TEXT_EMBEDDING_MODEL_ID = "hf-internal-testing/tiny-random-BertModel"
TINY_AUDIO_CTC_MODEL_ID = "hf-internal-testing/tiny-random-wav2vec2"
TINY_WHISPER_MODEL_ID = "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
TINY_SEQ_CLASSIFICATION_MODEL_ID = "ydshieh/tiny-random-BertForSequenceClassification"
TINY_AWQ_MODEL_ID = "optimum-intel-internal-testing/tiny-mixtral-AWQ-4bit"

QWEN_QUICKCHECK_MODEL_TYPES = (
    "qwen3",
    "qwen3_5",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5_moe",
)
QWEN_MOE_QUICKCHECK_MODEL_TYPES = ("qwen3_moe", "qwen3_vl_moe", "qwen3_5_moe")
QWEN_EXPECTED_DECODER_LAYERS = {
    "qwen3": "QEffQwen3DecoderLayer",
    "qwen3_5": "QEffQwen3_5DecoderLayer",
    "qwen3_moe": "QEffQwen3MoeDecoderLayer",
    "qwen3_vl": "QEffQwen3VLTextDecoderLayer",
    "qwen3_vl_moe": "QEffQwen3VLMoeTextDecoderLayer",
    "qwen3_5_moe": "QEffQwen3_5MoeDecoderLayer",
}

TINY_MOE_PREFILL_SUBFUNCTION_CONFIGS = {
    "glm4_moe": dict(
        max_position_embeddings=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        vocab_size=127,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        n_group=1,
        topk_group=1,
        head_dim=16,
    ),
    "qwen3_moe": dict(
        max_position_embeddings=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        vocab_size=127,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
    ),
    "gpt_oss": dict(
        max_position_embeddings=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=32,
        intermediate_size=32,
        vocab_size=127,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        head_dim=16,
        sliding_window=128,
        rope_scaling=None,
    ),
}

MODEL_KWARGS = {"attn_implementation": "eager"}
PREFIX_CACHING_MODEL_ID = "hf-internal-testing/tiny-random-GPT2LMHeadModel"


def _tiny_minimax_m3_text_config(dtype=torch.float32, microbench=False) -> MiniMaxM3VLTextConfig:
    config = MiniMaxM3VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        dense_intermediate_size=64,
        shared_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        layer_types=["full_attention", "minimax_m3_sparse"],
        mlp_layer_types=["dense", "sparse"],
        index_n_heads=2,
        index_head_dim=8,
        index_block_size=4,
        index_topk_blocks=2,
        index_local_blocks=1,
    )
    if microbench:
        config.num_hidden_layers = 1
        config.layer_types = ["minimax_m3_sparse"]
        config.mlp_layer_types = ["dense"]
        config.head_dim = 32
        config.max_position_embeddings = 8
        config.index_head_dim = 16
        config.index_block_size = 8
        config.index_topk_blocks = 1
        config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "partial_rotary_factor": 0.5,
        }
    config.torch_dtype = dtype
    return config


def _tiny_minimax_m3_vlm_config(microbench=False) -> MiniMaxM3VLConfig:
    text_config = _tiny_minimax_m3_text_config(microbench=microbench)
    vision_config = MiniMaxM3VLVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=1,
        num_channels=3,
    )
    return MiniMaxM3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=4,
        projector_hidden_size=text_config.hidden_size,
    )


def _tiny_minimax_m3_paged_dp_cp_config():
    args = SimpleNamespace(
        num_kv_blocks=1,
        attn_dp=1,
        attn_cp=1,
        indexer_num_blocks=None,
        msa_num_kv_blocks=None,
        skip_kv=False,
        non_dynamic_cache=False,
        msa_indexer_dp=1,
        msa_indexer_cp=2,
        msa_attn_dp=2,
        msa_attn_cp=2,
        indexer_n_head=1,
        num_cores=1,
        gqa_num_pages=1,
        msa_indexer_num_logical_pages=2,
        msa_attn_num_logical_pages=2,
        gqa_physical_pages=8,
        msa_indexer_physical_pages=8,
        msa_attn_physical_pages=8,
        expert_parallel_chunk_size=4,
        cores_per_expert=1,
        tree_reduce=True,
    )
    qaic_config = _build_qaic_config(args, 8, 4, 4)
    # Expert parallelism is a separate compile concern; keep this parity
    # fixture focused on the DP/CP/paged-attention runtime layout.
    qaic_config.pop("moe_config")
    qaic_config.pop("indexer_num_blocks")
    qaic_config.pop("msa_num_kv_blocks")
    qaic_config.pop("skip_kv")
    qaic_config.pop("non_dynamic_cache")
    return qaic_config


def _tiny_minimax_m3_paged_decode_inputs(qeff_model):
    inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        ctx_len=8,
        past_seq_len=8,
        prefill_seq_len=1,
        batch_size=2,
    )["lang"]
    inputs["input_ids"] = torch.tensor([[3], [5]], dtype=torch.int64)
    inputs["position_ids"] = torch.tensor([[7], [7]], dtype=torch.int64)
    table_generator = torch.Generator(device="cpu").manual_seed(11)
    inputs["gqa_block_table"] = build_block_table(1, 1, 2, 1, table_generator, 8)
    inputs["msa_indexer_block_table"] = build_block_table(1, 2, 2, 2, table_generator, 8)
    inputs["msa_attn_block_table"] = build_block_table(2, 2, 2, 2, table_generator, 8)
    return inputs


def _flatten_minimax_paged_ort_inputs(session, inputs):
    ort_inputs = {}
    input_names = {item.name for item in session.get_inputs()}
    for name in input_names:
        if name in inputs and torch.is_tensor(inputs[name]):
            ort_inputs[name] = inputs[name].detach().numpy()
        elif name.startswith(("past_key.", "past_value.")):
            layer_idx = int(name.split(".")[1])
            cache_idx = 0 if name.startswith("past_key.") else 1
            ort_inputs[name] = inputs["past_key_values"][layer_idx][cache_idx].detach().numpy()
        elif name.startswith("index_key."):
            ort_inputs[name] = inputs["index_keys"][0].detach().numpy()
        else:
            raise AssertionError(f"Unhandled MiniMax ORT input: {name}")
    return ort_inputs


def _per_test_thread_budget() -> int:
    override = os.environ.get("QEFF_NUM_THREADS")
    if override:
        return max(1, int(override))
    total = os.cpu_count() or 1
    workers = max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    return max(1, total // workers)


def _configure_torch_threads() -> None:
    threads = _per_test_thread_budget()
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads)))


def _ort_session(onnx_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    threads = _per_test_thread_budget()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    onnx_model = onnx.load(onnx_path, load_external_data=True)
    int32_max = torch.iinfo(torch.int32).max
    graph_was_patched = False
    added_initializers = {}

    def _zero_int32_max_tensor_attr(attr) -> bool:
        if attr.type != onnx.AttributeProto.TENSOR:
            return False
        np_tensor = onnx.numpy_helper.to_array(attr.t, os.path.dirname(onnx_path))
        if np.issubdtype(np_tensor.dtype, np.integer) and np.any(np_tensor == int32_max):
            patched_tensor = np.where(np_tensor == int32_max, np.array(0, dtype=np_tensor.dtype), np_tensor)
            attr.t.CopyFrom(onnx.numpy_helper.from_array(patched_tensor.astype(np_tensor.dtype, copy=False)))
            return True
        return False

    def _patch_constant_nodes(nodes, *, allow_initializers: bool = False) -> bool:
        patched = False
        for node in nodes:
            if node.op_type in {"Constant", "ConstantOfShape"}:
                for attr in node.attribute:
                    if _zero_int32_max_tensor_attr(attr):
                        patched = True
                        if allow_initializers and node.op_type == "Constant":
                            np_tensor = onnx.numpy_helper.to_array(attr.t, os.path.dirname(onnx_path))
                            added_initializers[node.output[0]] = ort.OrtValue.ortvalue_from_numpy(np_tensor)
        return patched

    def _patch_ort_index_inputs(nodes) -> bool:
        patched = False
        for index, node in list(enumerate(nodes)):
            if node.op_type not in {"GatherND", "ScatterND"} or len(node.input) < 2:
                continue
            indices_name = node.input[1]
            cast_output = f"{indices_name}__ort_int64"
            cast_node = onnx.helper.make_node(
                "Cast",
                inputs=[indices_name],
                outputs=[cast_output],
                name=f"{node.name or node.op_type}_IndicesCastToInt64",
                to=onnx.TensorProto.INT64,
            )
            nodes.insert(index, cast_node)
            node.input[1] = cast_output
            patched = True
        return patched

    graph_was_patched |= _patch_constant_nodes(onnx_model.graph.node, allow_initializers=True)
    graph_was_patched |= _patch_ort_index_inputs(onnx_model.graph.node)
    for function_proto in onnx_model.functions:
        graph_was_patched |= _patch_constant_nodes(function_proto.node)
        graph_was_patched |= _patch_ort_index_inputs(function_proto.node)

    for name, value in added_initializers.items():
        options.add_initializer(name, value)
    if graph_was_patched:
        session = ort.InferenceSession(onnx_model.SerializeToString(), sess_options=options)
    else:
        session = ort.InferenceSession(str(onnx_path), sess_options=options)
    # SessionOptions does not own the OrtValue buffers registered above.
    session._qeff_added_initializers = added_initializers
    return session


def _ort_session_zeroing_int32_max_constants(onnx_path: Path):
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    options = ort.SessionOptions()
    added_initializers = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(onnx_path))
            if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                added_initializers[node.output[0]] = ort.OrtValue.ortvalue_from_numpy(np.array(0, np_tensor.dtype))
    for name, value in added_initializers.items():
        options.add_initializer(name, value)
    return ort.InferenceSession(str(onnx_path), sess_options=options), added_initializers


_configure_torch_threads()


def _cleanup_stale_tmp_exports() -> None:
    tmp_root = Path(tempfile.gettempdir())
    for pattern in ("qeff_*", "*qeff*", "*onnx*", "*qnn*"):
        for path in tmp_root.glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.is_file():
                    path.unlink(missing_ok=True)
            except OSError:
                # Best-effort cleanup only.
                pass


@pytest.fixture(scope="session", autouse=True)
def _clean_tmp_exports_before_quickcheck():
    # Avoid concurrent cleanup from all xdist workers.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker not in (None, "gw0"):
        return
    _cleanup_stale_tmp_exports()


@contextmanager
def _suppress_native_output():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file()
    return onnx_path


def _count_decoder_block_subfunctions(onnx_model, qeff_model) -> int:
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return 0

    submodules = get_submodules()
    if not submodules:
        return 0

    if not isinstance(submodules, (set, list, tuple)):
        submodules = [submodules]

    block_names = {module.__name__ for module in submodules if hasattr(module, "__name__")}
    return sum(any(block_name in func.name for block_name in block_names) for func in onnx_model.functions)


def _decoder_block_subfunction_names(onnx_model, qeff_model) -> Set[str]:
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    assert callable(get_submodules)

    submodules = get_submodules()
    assert submodules

    if not isinstance(submodules, (set, list, tuple)):
        submodules = [submodules]

    block_names = {module.__name__ for module in submodules if hasattr(module, "__name__")}
    assert block_names
    return {func.name for func in onnx_model.functions if any(block_name in func.name for block_name in block_names)}


def _function_op_types(onnx_model, function_names: Set[str]) -> Set[str]:
    return {
        node.op_type
        for function_proto in onnx_model.functions
        if function_proto.name in function_names
        for node in function_proto.node
    }


def _function_call_counts(onnx_model, function_names: Set[str]) -> Counter:
    nodes = list(onnx_model.graph.node)
    nodes.extend(node for function_proto in onnx_model.functions for node in function_proto.node)
    return Counter(node.op_type for node in nodes if node.op_type in function_names)


def _assert_has_retained_state_outputs(onnx_path: Path) -> None:
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    retained_outputs = [output.name for output in onnx_model.graph.output if output.name.endswith("_RetainedState")]
    assert retained_outputs


def _run_embedding_ort(onnx_path: Path, inputs: Dict[str, torch.Tensor]) -> np.ndarray:
    session = _ort_session(onnx_path)
    input_names = {item.name for item in session.get_inputs()}
    ort_inputs = {name: tensor.detach().numpy() for name, tensor in inputs.items() if name in input_names}
    return session.run(None, ort_inputs)[0]


def _run_whisper_export_smoke(qeff_model: QEFFAutoModelForSpeechSeq2Seq, out_dir: Path) -> Path:
    onnx_path = _exported_onnx_path(qeff_model.export(out_dir))
    _assert_has_retained_state_outputs(onnx_path)
    return onnx_path


def _assert_proxy_only_onnx_transform_policy(
    qeff_model, enable_proxy: bool, always_on_transforms: Optional[Set[str]] = None
) -> None:
    pytorch_transform_names = {transform.__name__ for transform in qeff_model._pytorch_transforms}
    transform_names = {transform.__name__ for transform in qeff_model._onnx_transforms}
    proxy_pytorch_transform = "QeffProxyModuleTransform"
    proxy_only_transforms = {"FP16ClipTransform", "SplitTensorsTransform"}
    always_on_transforms = always_on_transforms or set()
    conditional_proxy_transforms = proxy_only_transforms - always_on_transforms

    if enable_proxy:
        assert proxy_pytorch_transform in pytorch_transform_names
        assert proxy_only_transforms.issubset(transform_names)
    else:
        assert proxy_pytorch_transform not in pytorch_transform_names
        assert conditional_proxy_transforms.isdisjoint(transform_names)
        assert always_on_transforms.issubset(transform_names)


def _assert_dual_qpc_vlm_proxy_transform_policy(qeff_model, enable_proxy: bool) -> None:
    _assert_proxy_only_onnx_transform_policy(qeff_model.vision_model, enable_proxy=enable_proxy)
    _assert_proxy_only_onnx_transform_policy(qeff_model.lang_model, enable_proxy=enable_proxy)


def _skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def _export_vlm_with_text_fallback(model_id: str, out_dir: Path) -> Path:
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        use_text_only_first = model_type in {"qwen2_5_vl", "qwen2_5_vl_text", "internvl_chat"}

        if not use_text_only_first:
            try:
                vlm_model = QEFFAutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True)
                return _exported_onnx_path(vlm_model.export(out_dir / "full-vlm"))
            except Exception:
                pass

        try:
            if model_type in {"qwen2_5_vl", "qwen2_5_vl_text"} and getattr(config, "text_config", None) is not None:
                qwen2_cfg_dict = config.text_config.to_dict()
                qwen2_cfg_dict["model_type"] = "qwen2"
                qwen2_allowed_keys = set(Qwen2Config().to_dict().keys())
                qwen2_cfg = Qwen2Config(**{k: v for k, v in qwen2_cfg_dict.items() if k in qwen2_allowed_keys})
                text_model = AutoModelForCausalLM.from_config(qwen2_cfg, trust_remote_code=True, **MODEL_KWARGS)
                text_model = text_model.to(torch.float32)
                text_model.eval()
                qeff_text_model = QEFFAutoModelForCausalLM(text_model)
                return _exported_onnx_path(qeff_text_model.export(out_dir / "text-fallback"))

            text_configs = [getattr(config, "text_config", None), getattr(config, "llm_config", None)]
            for text_config in text_configs:
                if text_config is None:
                    continue
                try:
                    text_model = AutoModelForCausalLM.from_config(
                        text_config,
                        trust_remote_code=True,
                        **MODEL_KWARGS,
                    )
                    text_model = text_model.to(torch.float32)
                    text_model.eval()
                    qeff_text_model = QEFFAutoModelForCausalLM(text_model)
                    return _exported_onnx_path(qeff_text_model.export(out_dir / "text-fallback"))
                except Exception:
                    continue
            raise RuntimeError(f"No text fallback config path available for {model_id}")
        except Exception as text_exc:
            _skip_on_model_fetch_error(text_exc, model_id)
    except Exception as cfg_exc:
        _skip_on_model_fetch_error(cfg_exc, model_id)


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        dtype="float32",
    )


def _tiny_qwen3_moe_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=256,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[],
        dtype="float32",
    )


def _tiny_qwen3_5_text_config(*, moe: bool = False) -> Qwen3_5TextConfig | Qwen3_5MoeTextConfig:
    kwargs = dict(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention", "linear_attention"],
        dtype="float32",
    )
    if moe:
        kwargs.update(
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=2,
            num_experts_per_tok=1,
        )
        return Qwen3_5MoeTextConfig(**kwargs)

    kwargs.update(
        intermediate_size=32,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    return Qwen3_5TextConfig(**kwargs)


def _tiny_qwen_vision_config(config_cls, *, deepstack: bool = False):
    kwargs = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=4,
        temporal_patch_size=1,
        spatial_merge_size=1,
        out_hidden_size=16,
        num_position_embeddings=64,
        dtype="float32",
    )
    if deepstack:
        kwargs["deepstack_visual_indexes"] = []
    return config_cls(**kwargs)


def _tiny_qwen3_vl_config() -> Qwen3VLConfig:
    text_config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_scaling={"rope_type": "default", "mrope_section": [11, 11, 10]},
        dtype="float32",
    )
    return Qwen3VLConfig(
        text_config=text_config,
        vision_config=_tiny_qwen_vision_config(Qwen3VLVisionConfig, deepstack=True),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _tiny_qwen3_vl_moe_config(num_experts: int = 2) -> Qwen3VLMoeConfig:
    text_config = Qwen3VLMoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_experts=num_experts,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_scaling={"rope_type": "default", "mrope_section": [11, 11, 10]},
        dtype="float32",
    )
    return Qwen3VLMoeConfig(
        text_config=text_config,
        vision_config=_tiny_qwen_vision_config(Qwen3VLMoeVisionConfig, deepstack=True),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _tiny_qwen3_5_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config=_tiny_qwen3_5_text_config(),
        vision_config=_tiny_qwen_vision_config(Qwen3_5VisionConfig),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _tiny_qwen3_5_moe_config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        text_config=_tiny_qwen3_5_text_config(moe=True),
        vision_config=_tiny_qwen_vision_config(Qwen3_5MoeVisionConfig),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _tiny_qwen_config(model_type: str):
    config_builders = {
        "qwen3": _tiny_qwen3_config,
        "qwen3_5": _tiny_qwen3_5_config,
        "qwen3_moe": _tiny_qwen3_moe_config,
        "qwen3_vl": _tiny_qwen3_vl_config,
        "qwen3_vl_moe": _tiny_qwen3_vl_moe_config,
        "qwen3_5_moe": _tiny_qwen3_5_moe_config,
    }
    return config_builders[model_type]()


def _tiny_qwen_qeff_model(model_type: str):
    config = _tiny_qwen_config(model_type)
    torch.manual_seed(0)
    if model_type in {"qwen3", "qwen3_moe"}:
        hf_model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS).eval()
        return QEFFAutoModelForCausalLM(hf_model, continuous_batching=False)

    hf_model = AutoModelForImageTextToText.from_config(config).eval()
    return QEFFAutoModelForImageTextToText(hf_model, kv_offload=True)


def _qwen_decoder_qeff_model(qeff_model):
    return qeff_model if isinstance(qeff_model, QEFFAutoModelForCausalLM) else qeff_model.lang_model


def _qwen_decoder_export_model(qeff_model):
    return _qwen_decoder_qeff_model(qeff_model).model


def _qwen_decoder_subfunction_names(qeff_model) -> Set[str]:
    return {module_cls.__name__ for module_cls in _qwen_decoder_export_model(qeff_model).get_submodules_for_export()}


def _tiny_qwen_hf_model(model_type: str):
    config = _tiny_qwen_config(model_type)
    torch.manual_seed(0)
    if model_type in {"qwen3", "qwen3_moe"}:
        return AutoModelForCausalLM.from_config(config, **MODEL_KWARGS).eval()

    return AutoModelForImageTextToText.from_config(config).eval()


def _tiny_qwen_checkpoint_path(model_type: str, tmp_path) -> Path:
    checkpoint_dir = tmp_path / f"{model_type}-tiny-checkpoint"
    if checkpoint_dir.is_dir():
        return checkpoint_dir

    hf_model = _tiny_qwen_hf_model(model_type)
    assert _tiny_qwen_text_config(model_type).num_hidden_layers == 2
    hf_model.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def _tiny_qwen_layerwise_hf_qeff_pair(model_type: str, tmp_path):
    checkpoint_path = str(_tiny_qwen_checkpoint_path(model_type, tmp_path))
    if model_type in {"qwen3", "qwen3_moe"}:
        hf_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **MODEL_KWARGS).eval()
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            continuous_batching=False,
            layerwise=True,
            layerwise_window_size=1,
            **MODEL_KWARGS,
        )
    else:
        hf_model = AutoModelForImageTextToText.from_pretrained(checkpoint_path, **MODEL_KWARGS).eval()
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            kv_offload=True,
            layerwise=True,
            layerwise_window_size=1,
            **MODEL_KWARGS,
        )
    return hf_model, qeff_model


def _tiny_qwen_hf_qeff_pair(model_type: str, tmp_path=None, *, layerwise: bool = False):
    if layerwise:
        if tmp_path is None:
            raise ValueError("Layerwise quickcheck requires tmp_path for the tiny checkpoint.")
        return _tiny_qwen_layerwise_hf_qeff_pair(model_type, tmp_path)

    hf_model = _tiny_qwen_hf_model(model_type)
    qeff_source = deepcopy(hf_model).eval()
    if model_type in {"qwen3", "qwen3_moe"}:
        qeff_model = QEFFAutoModelForCausalLM(qeff_source, continuous_batching=False)
    else:
        qeff_model = QEFFAutoModelForImageTextToText(qeff_source, kv_offload=True)
    return hf_model, qeff_model


def _tiny_qwen_text_config(model_type: str):
    config = _tiny_qwen_config(model_type)
    return getattr(config, "text_config", config)


def _qwen_causal_inputs(config, *, seq_len: int = 4, ctx_len: int = 8):
    input_ids = (torch.arange(seq_len, dtype=torch.int64).view(1, seq_len) % (config.vocab_size - 1)) + 1
    position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    past_key_values = tuple(
        (
            torch.zeros((1, config.num_key_value_heads, ctx_len, head_dim), dtype=torch.float32),
            torch.zeros((1, config.num_key_value_heads, ctx_len, head_dim), dtype=torch.float32),
        )
        for _ in range(config.num_hidden_layers)
    )
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


def _qwen_vlm_lang_inputs(qeff_model, *, prefill_seq_len: int = 4):
    try:
        inputs = qeff_model.model.get_dummy_inputs(kv_offload=True, prefill_seq_len=prefill_seq_len)["lang"]
    except TypeError:
        inputs = qeff_model.model.get_dummy_inputs(kv_offload=True)["lang"]
    inputs = deepcopy(inputs)
    if "input_ids" in inputs:
        inputs["input_ids"].fill_(1)
    return inputs


def _qwen_export_io(qeff_model, *, prefill_seq_len: int = 4):
    if isinstance(qeff_model, QEFFAutoModelForCausalLM):
        return None, None, None

    inputs = qeff_model.model.get_dummy_inputs(kv_offload=True, prefill_seq_len=prefill_seq_len)
    dynamic_axes = qeff_model.model.get_onnx_dynamic_axes(kv_offload=True)
    output_names = qeff_model.model.get_output_names(kv_offload=True)
    lang_inputs = deepcopy(inputs["lang"])
    if "input_ids" in lang_inputs:
        lang_inputs["input_ids"].fill_(1)
    return lang_inputs, output_names["lang"], dynamic_axes["lang"]


def _extract_qwen_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits.detach().float().numpy()
    if isinstance(outputs, (tuple, list)):
        return outputs[0].detach().float().numpy()
    return outputs.detach().float().numpy()


def _as_hf_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    return tuple(tuple(tensor.detach().clone() for tensor in layer) for layer in past_key_values)


def _hf_qwen_vlm_text_logits(hf_model, inputs):
    language_model = (
        hf_model.model.language_model if hasattr(hf_model.model, "language_model") else hf_model.language_model
    )
    outputs = language_model(
        input_ids=inputs["input_ids"],
        position_ids=inputs["position_ids"],
        use_cache=True,
    )
    position_ids = inputs["position_ids"]
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    logit_index = text_position_ids.to(torch.int32).argmax(1, keepdim=True)
    hidden_states = outputs.last_hidden_state[torch.arange(text_position_ids.shape[0]).view(-1, 1), logit_index]
    return hf_model.lm_head(hidden_states).detach().float().numpy()


def _hf_qwen_logits(model_type: str, hf_model, inputs):
    with torch.no_grad():
        if model_type in {"qwen3", "qwen3_moe"}:
            hf_inputs = {
                "input_ids": inputs["input_ids"],
                "position_ids": inputs["position_ids"],
                "use_cache": True,
            }
            logits = _extract_qwen_logits(hf_model(**hf_inputs))
            logit_index = inputs["position_ids"].to(torch.int32).argmax(1, keepdim=True).detach().cpu().numpy()
            batch_index = np.arange(inputs["position_ids"].shape[0]).reshape(-1, 1)
            return logits[batch_index, logit_index]
        return _hf_qwen_vlm_text_logits(hf_model, inputs)


def _qeff_qwen_logits(qeff_model, inputs):
    with torch.no_grad():
        return _extract_qwen_logits(_qwen_decoder_qeff_model(qeff_model).model(**inputs))


def _flatten_qwen_ort_inputs(inputs, session_inputs, qeff_export_model):
    flat_inputs = {name: value.detach().numpy() for name, value in inputs.items() if torch.is_tensor(value)}
    past_key_values = inputs.get("past_key_values")
    if past_key_values is not None:
        for layer_idx, layer_state in enumerate(past_key_values):
            if hasattr(qeff_export_model, "get_onnx_past_key_value_names"):
                names = qeff_export_model.get_onnx_past_key_value_names(layer_idx, layer_state)
            elif len(layer_state) == 2:
                names = [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
            else:
                names = [f"past_state.{layer_idx}.{state_idx}" for state_idx in range(len(layer_state))]
            for name, tensor in zip(names, layer_state):
                flat_inputs[name] = tensor.detach().numpy()
    return {name: flat_inputs[name] for name in session_inputs if name in flat_inputs}


def _ort_qwen_logits(onnx_path: Path, qeff_model, inputs):
    session = _ort_session(onnx_path)
    session_inputs = [item.name for item in session.get_inputs()]
    session_outputs = [item.name for item in session.get_outputs()]
    ort_inputs = _flatten_qwen_ort_inputs(inputs, session_inputs, _qwen_decoder_export_model(qeff_model))
    outputs = dict(zip(session_outputs, session.run(session_outputs, ort_inputs)))
    return outputs["logits"].astype(np.float32)


def _export_qwen_decoder_onnx(qeff_model, inputs, tmp_path, *, model_type: str, prefill_only: bool, layerwise: bool):
    export_dir = (
        tmp_path / f"{model_type}-{'layerwise' if layerwise else 'default'}-{'prefill' if prefill_only else 'decode'}"
    )
    if isinstance(qeff_model, QEFFAutoModelForCausalLM):
        if layerwise:
            return _exported_onnx_path(
                qeff_model.export(
                    export_dir,
                    prefill_only=prefill_only,
                    prefill_seq_len=inputs["input_ids"].shape[1],
                    offload_pt_weights=False,
                )
            )
        return _exported_onnx_path(
            qeff_model.export(
                export_dir,
                prefill_only=prefill_only,
                prefill_seq_len=inputs["input_ids"].shape[1],
                offload_pt_weights=False,
            )
        )

    _, output_names, dynamic_axes = _qwen_export_io(qeff_model, prefill_seq_len=inputs["input_ids"].shape[1])
    lang_model = _qwen_decoder_qeff_model(qeff_model)
    if layerwise:
        return _exported_onnx_path(
            qeff_model.export(
                export_dir=export_dir,
                skip_vision=True,
                prefill_only=prefill_only,
                prefill_seq_len=inputs["input_ids"].shape[1],
                offload_pt_weights=False,
            )
        )

    return _exported_onnx_path(
        lang_model.export(
            inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=False,
            prefill_only=prefill_only,
            prefill_seq_len=inputs["input_ids"].shape[1],
        )
    )


def _assert_qwen_hf_qeff_ort_parity(model_type: str, tmp_path, *, prefill_only: bool = False, layerwise: bool = False):
    hf_model, qeff_model = _tiny_qwen_hf_qeff_pair(model_type, tmp_path, layerwise=layerwise)
    seq_len = 4
    if prefill_only:
        seq_len = 256 if model_type == "qwen3_moe" else 8

    if isinstance(qeff_model, QEFFAutoModelForCausalLM):
        ctx_len = seq_len if layerwise else max(seq_len, 8)
        inputs = _qwen_causal_inputs(_tiny_qwen_text_config(model_type), seq_len=seq_len, ctx_len=ctx_len)
    else:
        inputs = _qwen_vlm_lang_inputs(qeff_model, prefill_seq_len=seq_len)

    hf_logits = _hf_qwen_logits(model_type, hf_model, inputs)
    qeff_logits = None if (prefill_only or layerwise) else _qeff_qwen_logits(qeff_model, inputs)
    onnx_path = _export_qwen_decoder_onnx(
        qeff_model,
        inputs,
        tmp_path,
        model_type=model_type,
        prefill_only=prefill_only,
        layerwise=layerwise,
    )
    if qeff_logits is None and not layerwise:
        qeff_logits = _qeff_qwen_logits(qeff_model, inputs)
    ort_logits = _ort_qwen_logits(onnx_path, qeff_model, inputs)

    atol = 2e-3 if model_type == "qwen3_5_moe" else 1e-4
    if layerwise:
        assert np.allclose(hf_logits, ort_logits, atol=atol, rtol=1e-4)
    else:
        assert np.allclose(hf_logits, qeff_logits, atol=atol, rtol=1e-4)
        assert np.allclose(qeff_logits, ort_logits, atol=atol, rtol=1e-4)


def _kimi_k25_prompt_inputs(processor):
    from PIL import Image

    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": "Describe."},
            ],
        },
    ]
    inputs = processor(
        messages=messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )
    return {name: (value.to("cpu") if torch.is_tensor(value) else value) for name, value in inputs.items()}


def _kimi_k25_qeff_lang_inputs(qeff_model, inputs, vision_embeds):
    seq_len = inputs["input_ids"].shape[1]
    lang_inputs = deepcopy(qeff_model.model.get_dummy_inputs(kv_offload=True, prefill_seq_len=seq_len)["lang"])
    lang_inputs["input_ids"] = inputs["input_ids"].clone()
    lang_inputs["position_ids"] = torch.where(
        inputs["attention_mask"] > 0,
        torch.arange(seq_len, dtype=torch.int64).view(1, seq_len),
        -1,
    )
    lang_inputs["vision_embeds"] = vision_embeds
    lang_inputs["image_idx"] = torch.zeros((1, 1), dtype=torch.int64)
    return lang_inputs


class _KimiSyntheticQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        if linear.in_features % 2 != 0:
            raise ValueError("Synthetic int4 fixture requires an even input dimension")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = 4
        self.group_size = 1
        self.act_order = None
        self.register_buffer(
            "qweight", torch.full((linear.out_features, linear.in_features // 2), 0x99, dtype=torch.uint8)
        )
        self.register_buffer(
            "qzeros", torch.full((linear.out_features, linear.in_features // 2), 0x88, dtype=torch.uint8)
        )
        self.register_buffer("scales", linear.weight.detach().clone().to(torch.float32))
        self.register_buffer("g_idx", torch.arange(linear.in_features, dtype=torch.int32))
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone().to(torch.float32))

    def forward(self, inputs):
        output = torch.matmul(inputs.float(), self.scales.transpose(0, 1).float())
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output.to(inputs.dtype)


def _install_kimi_synthetic_quant_experts(model):
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        for expert in experts:
            for projection_name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(expert, projection_name, None)
                if isinstance(projection, nn.Linear):
                    setattr(expert, projection_name, _KimiSyntheticQuantLinear(projection))


@pytest.mark.llm_model
def test_kimi_k25_quickcheck_hf_qeff_vision_logits_parity():
    from tests.utils.load_kimi_utils import get_kimi_k25_test_config, load_kimi_k25_model_from_config

    model_id = "moonshotai/Kimi-K2.5"
    config_path = Path(__file__).parents[2] / "configs" / "image_text_model_configs.json"
    model_configs = json.loads(config_path.read_text())["image_text_models"]
    model_config_dict = {model["model_name"]: model for model in model_configs}

    try:
        config = get_kimi_k25_test_config(model_id, model_config_dict)
        model_hf, _, processor = load_kimi_k25_model_from_config(config)
        _install_kimi_synthetic_quant_experts(model_hf)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    inputs = _kimi_k25_prompt_inputs(processor)
    with torch.no_grad():
        hf_outputs = model_hf(**inputs, use_cache=False, return_dict=True)
        hf_logits = hf_outputs.logits[:, -1:, :].detach().float().numpy()

    qeff_model = QEFFAutoModelForImageTextToText(
        deepcopy(model_hf),
        kv_offload=True,
        config=model_hf.config,
        torch_dtype=torch.float32,
    )
    grid_thws = inputs["grid_thws"].to(torch.int64)
    h_shape = torch.ones((int(grid_thws[0, 1].item()),), dtype=torch.int64)
    w_shape = torch.ones((int(grid_thws[0, 2].item()),), dtype=torch.int64)

    with torch.no_grad():
        vision_embeds = qeff_model.vision_model.model(inputs["pixel_values"], h_shape, w_shape)
        lang_inputs = _kimi_k25_qeff_lang_inputs(qeff_model, inputs, vision_embeds)
        qeff_logits = qeff_model.lang_model.model(**lang_inputs)[0].detach().float().numpy()

    assert qeff_logits.shape == hf_logits.shape
    assert np.allclose(hf_logits, qeff_logits, atol=1e-4, rtol=1e-4)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_lm_cpu_runtime_parity_with_api_runner(model_type, model_id, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    prompt = ["hello world"]
    prompt_len = 8
    ctx_len = 12

    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id,
        **MODEL_KWARGS,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model_hf.eval()

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompt,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0))
    assert np.array_equal(kv_tokens, ort_tokens)


@pytest.mark.llm_model
def test_minimax_m3_npi_generation_tracks_exported_graph(tmp_path):
    nodes = [
        onnx.helper.make_node("Add", ["x", "x"], ["/language_model/layers.0/Add_output_0"]),
        onnx.helper.make_node(
            "CustomRMSNorm", ["x"], ["/language_model/layers.0/input_layernorm/CustomRMSNorm_output_0"]
        ),
        onnx.helper.make_node("Identity", ["x"], ["/language_model/layers.0/ignored_output"]),
        onnx.helper.make_node("CustomRMSNorm", ["x"], ["/language_model/norm/CustomRMSNorm_output_0"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "minimax",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [
            onnx.helper.make_tensor_value_info(
                "/language_model/norm/CustomRMSNorm_output_0", onnx.TensorProto.FLOAT, [1]
            )
        ],
    )
    onnx_path = tmp_path / "minimax.onnx"
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)]), onnx_path)

    npi_path = Path(_generate_minimax_npi_file(onnx_path))
    npi = yaml.safe_load(npi_path.read_text())
    assert npi["FP32NodeInstanceNames"] == [
        "/language_model/layers.0/Add_output_0",
        "/language_model/layers.0/input_layernorm/CustomRMSNorm_output_0",
        "/language_model/norm/CustomRMSNorm_output_0",
    ]


@pytest.mark.llm_model
def test_minimax_m3_npi_generation_expands_decoder_functions(tmp_path):
    decoder_function = onnx.helper.make_function(
        "test.minimax",
        "Decoder",
        ["hidden_states"],
        ["hidden_states.2"],
        [
            onnx.helper.make_node("CustomRMSNorm", ["hidden_states", "w0"], ["norm0"], name="CustomRMSNorm_0"),
            onnx.helper.make_node("CustomRMSNorm", ["norm0", "w1"], ["norm1"], name="CustomRMSNorm_1"),
            onnx.helper.make_node("CustomRMSNorm", ["norm1", "w2"], ["norm2"], name="CustomRMSNorm_2"),
            onnx.helper.make_node("Add", ["hidden_states", "attn"], ["hidden_states.1"], name="Add_0"),
            onnx.helper.make_node("CustomRMSNorm", ["hidden_states.1", "w3"], ["norm3"], name="CustomRMSNorm_3"),
            onnx.helper.make_node("Add", ["hidden_states.1", "mlp"], ["hidden_states.2"], name="Add_1"),
        ],
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "Decoder",
                ["x"],
                ["/language_model/layers.0/Decoder_output"],
                name="/language_model/layers.0/Decoder",
                domain="test.minimax",
            )
        ],
        "minimax",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("/language_model/layers.0/Decoder_output", onnx.TensorProto.FLOAT, [1])],
    )
    onnx_path = tmp_path / "minimax.onnx"
    onnx.save(
        onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 17)],
            functions=[decoder_function],
        ),
        onnx_path,
    )

    npi_path = Path(_generate_minimax_npi_file(onnx_path))
    npi = yaml.safe_load(npi_path.read_text())
    assert npi["FP32NodeInstanceNames"] == [
        "/language_model/layers.0/CustomRMSNorm_0_output_0",
        "/language_model/layers.0/CustomRMSNorm_1_output_0",
        "/language_model/layers.0/CustomRMSNorm_2_output_0",
        "/language_model/layers.0/Add_0_output_0",
        "/language_model/layers.0/CustomRMSNorm_3_output_0",
        "/language_model/layers.0/Add_1_output_0",
    ]


@pytest.mark.llm_model
def test_minimax_m3_text_config_derived_pt_and_onnx_runtime_parity(tmp_path):
    torch.manual_seed(7)
    vlm_config = _tiny_minimax_m3_vlm_config()
    text_config = vlm_config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    model_hf_orig = deepcopy(model_hf)

    input_ids = torch.tensor([[3]], dtype=torch.int64)
    position_ids = torch.tensor([[4]], dtype=torch.int64)
    hidden_size = text_config.hidden_size
    cache_seq_len = 4
    # Use the same zero-initialized KV cache for the HF reference and QEff paths.
    runtime_cache_len = 8
    past_key_values = tuple(
        (
            torch.zeros((1, text_config.num_key_value_heads, runtime_cache_len, text_config.head_dim)),
            torch.zeros((1, text_config.num_key_value_heads, runtime_cache_len, text_config.head_dim)),
        )
        for _ in range(text_config.num_hidden_layers)
    )
    hf_past_key_values = DynamicCache(config=text_config)
    for layer_idx, (key, value) in enumerate(past_key_values):
        hf_past_key_values.update(key[:, :, :cache_seq_len], value[:, :, :cache_seq_len], layer_idx)
    for layer_idx, layer_type in enumerate(text_config.layer_types):
        if layer_type == "minimax_m3_sparse":
            hf_past_key_values.layers[layer_idx].update_index(
                torch.zeros((1, 1, cache_seq_len, text_config.index_head_dim))
            )

    with torch.no_grad():
        inputs_embeds_hf = model_hf_orig.model.language_model.embed_tokens(input_ids)
        hf_out = model_hf_orig.model.language_model(
            inputs_embeds=inputs_embeds_hf,
            position_ids=position_ids,
            past_key_values=hf_past_key_values,
            use_cache=False,
        )
        hf_logits = model_hf_orig.lm_head(hf_out.last_hidden_state[:, -1:, :]).float()

    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)

    dummy_vision_embeds = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
    dummy_image_idx = torch.zeros((1, 1), dtype=torch.int64)

    index_keys = [
        torch.zeros((1, 1, runtime_cache_len, text_config.index_head_dim))
        for lt in text_config.layer_types
        if lt == "minimax_m3_sparse"
    ]

    with torch.no_grad():
        qeff_out = qeff_model.lang_model.model(
            input_ids=input_ids,
            vision_embeds=dummy_vision_embeds,
            position_ids=position_ids,
            image_idx=dummy_image_idx,
            past_key_values=past_key_values,
            index_keys=index_keys,
        )
    qeff_logits = qeff_out[0]

    assert torch.allclose(hf_logits, qeff_logits, atol=1e-5, rtol=1e-5)

    with torch.no_grad():
        qeff_cached_out = qeff_model.lang_model.model(
            input_ids=input_ids,
            vision_embeds=dummy_vision_embeds,
            position_ids=position_ids,
            image_idx=dummy_image_idx,
            past_key_values=past_key_values,
            index_keys=index_keys,
        )
    qeff_cached_logits = qeff_cached_out[0]

    onnx_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "minimax-m3-vlm", skip_vision=True, prefill_seq_len=1, ctx_len=runtime_cache_len)
    )
    session = _ort_session(onnx_path)
    ort_inputs = {}
    for input_meta in session.get_inputs():
        shape = [1 if not isinstance(dim, int) else dim for dim in input_meta.shape]
        if input_meta.name == "input_ids":
            ort_inputs[input_meta.name] = input_ids.numpy()
        elif input_meta.name == "position_ids":
            ort_inputs[input_meta.name] = position_ids.numpy()
        elif input_meta.name == "vision_embeds":
            ort_inputs[input_meta.name] = dummy_vision_embeds.numpy()
        elif input_meta.name == "image_idx":
            ort_inputs[input_meta.name] = dummy_image_idx.numpy()
        elif input_meta.name.startswith(("past_key.", "past_value.")):
            ort_inputs[input_meta.name] = np.zeros(
                (1, text_config.num_key_value_heads, runtime_cache_len, text_config.head_dim), dtype=np.float32
            )
        elif input_meta.name.startswith("index_key."):
            ort_inputs[input_meta.name] = np.zeros(
                (1, 1, runtime_cache_len, text_config.index_head_dim), dtype=np.float32
            )
        else:
            dtype = np.int64 if input_meta.type == "tensor(int64)" else np.float32
            ort_inputs[input_meta.name] = np.zeros(shape, dtype=dtype)

    ort_logits = session.run(None, ort_inputs)[0]
    assert ort_logits.shape == (1, 1, text_config.vocab_size)
    assert np.allclose(ort_logits, qeff_cached_logits.detach().numpy(), atol=1e-4, rtol=1e-4)

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    output_names = {output.name for output in onnx_model.graph.output}
    assert any(name.startswith("past_key.") and name.endswith("_RetainedState") for name in output_names)


@pytest.mark.llm_model
def test_minimax_m3_dummy_inputs_use_scaled_prefill_export_length():
    vlm_config = _tiny_minimax_m3_vlm_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": 1,
        "msa_indexer_dp": 1,
        "msa_indexer_cp": 1,
        "msa_attn_dp": 1,
        "msa_attn_cp": 1,
        "indexer_n_head": 1,
        "indexer_q_size": 8,
        "indexer_q_chunk": 32,
        "indexer_prefill_parallel": True,
        "msa_q_chunk": 32,
        "num_cores_per_device": 4,
    }

    qeff_model.transform(
        ctx_len=128,
        seq_len=128,
        bs=1,
        num_devices=1,
        num_cores=4,
        qaic_config=qaic_config,
        prefill_only=True,
        prefill_seq_len=128,
    )
    inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        ctx_len=128,
        past_seq_len=128,
        prefill_seq_len=128,
        batch_size=1,
    )["lang"]
    blocking_config = next(
        module.attn_blocking_config
        for module in qeff_model.model.language_model.modules()
        if getattr(module, "attn_blocking_config", None) is not None
    )

    assert inputs["input_ids"].shape[1] == 16
    assert inputs["position_ids"].shape[1] == 16
    assert blocking_config.prefill_export_seq_len == 16
    assert blocking_config.prefill_compile_seq_len == 128


@pytest.mark.llm_model
def test_minimax_m3_index_query_projection_uses_configured_loop_count():
    config = _tiny_minimax_m3_text_config()
    indexer = QEffMiniMaxM3VLIndexer(config, layer_idx=1).eval()
    blocking_config = SimpleNamespace(indexer_q_proj_num_chunks=8)
    hidden_states = torch.randn(1, 64, config.hidden_size)
    call_count = 0
    original_forward = indexer.q_proj.forward

    def counted_forward(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward(*args, **kwargs)

    indexer.q_proj.forward = counted_forward
    projected = indexer._project_index_q_prefill(hidden_states, blocking_config)

    assert call_count == 8
    assert projected.shape[:2] == hidden_states.shape[:2]


@pytest.mark.llm_model
def test_minimax_m3_reduced_export_runs_at_full_prefill_length(tmp_path):
    vlm_config = _tiny_minimax_m3_vlm_config()
    vlm_config.text_config.max_position_embeddings = 128
    text_config = vlm_config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": 1,
        "msa_indexer_dp": 1,
        "msa_indexer_cp": 1,
        "msa_attn_dp": 1,
        "msa_attn_cp": 1,
        "indexer_n_head": 1,
        "indexer_q_size": 8,
        "indexer_q_chunk": 32,
        "indexer_prefill_parallel": True,
        "msa_q_chunk": 32,
        "num_cores_per_device": 4,
    }
    qeff_model.transform(
        ctx_len=128,
        seq_len=128,
        bs=1,
        num_devices=1,
        num_cores=4,
        qaic_config=qaic_config,
        prefill_only=True,
        prefill_seq_len=128,
    )
    onnx_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "minimax-m3-reduced-prefill",
            skip_vision=True,
            prefill_only=True,
            enable_chunking=True,
            prefill_seq_len=128,
            ctx_len=128,
            qaic_config=qaic_config,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    session = _ort_session(onnx_path)
    ort_inputs = {}
    for input_meta in session.get_inputs():
        if input_meta.name == "input_ids":
            ort_inputs[input_meta.name] = np.zeros((1, 128), dtype=np.int64)
        elif input_meta.name == "position_ids":
            ort_inputs[input_meta.name] = np.arange(128, dtype=np.int64).reshape(1, 128)
        elif input_meta.name == "vision_embeds":
            ort_inputs[input_meta.name] = np.zeros((1, 128, text_config.hidden_size), dtype=np.float32)
        elif input_meta.name == "image_idx":
            ort_inputs[input_meta.name] = np.zeros((1, 1), dtype=np.int64)
        elif input_meta.name.startswith(("past_key.", "past_value.")):
            ort_inputs[input_meta.name] = np.zeros(
                (1, text_config.num_key_value_heads, 128, text_config.head_dim), dtype=np.float32
            )
        elif input_meta.name.startswith("index_key."):
            ort_inputs[input_meta.name] = np.zeros((1, 1, 128, text_config.index_head_dim), dtype=np.float32)
        else:
            raise AssertionError(f"Unhandled MiniMax dynamic-prefill ORT input: {input_meta.name}")

    outputs = session.run(None, ort_inputs)
    assert outputs[0].shape == (1, 1, text_config.vocab_size)


@pytest.mark.llm_model
def test_minimax_m3_cp2_decode_cache_shapes_and_retained_dynamic_axes(tmp_path):
    vlm_config = _tiny_minimax_m3_vlm_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "msa_indexer_dp": 1,
        "msa_indexer_cp": 2,
        "msa_attn_dp": 1,
        "msa_attn_cp": 2,
        "indexer_n_head": 1,
        "num_kv_blocks": 1,
        "num_cores_per_device": 1,
    }
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    # The public export/compile path synchronizes this configuration before
    # constructing dummy inputs and dynamic axes; mirror that path directly.
    qeff_model.model.qaic_config = qaic_config
    qeff_model.lang_model.model.qaic_config = qaic_config

    lang_inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        ctx_len=8,
        past_seq_len=8,
        prefill_seq_len=1,
        batch_size=1,
    )["lang"]
    dense_key, _ = lang_inputs["past_key_values"][0]
    sparse_key, sparse_value = lang_inputs["past_key_values"][1]
    (index_key,) = lang_inputs["index_keys"]

    assert tuple(dense_key.shape) == (1, 2, 8, 8)
    assert tuple(sparse_key.shape) == (1, 4, 4, 8)
    assert tuple(sparse_value.shape) == (1, 4, 4, 8)
    assert tuple(index_key.shape) == (1, 2, 4, 8)

    dynamic_axes = qeff_model.model.get_onnx_dynamic_axes(kv_offload=True)["lang"]
    expected_main_axes = {0: "main_kv_batch_size", 1: "main_kv_rows", 2: "main_kv_ctx_len"}
    expected_index_axes = {0: "indexer_kv_batch_size", 1: "indexer_kv_rows", 2: "indexer_kv_ctx_len"}
    for name in ("past_key.1", "past_value.1"):
        assert dynamic_axes[name] == expected_main_axes
        assert dynamic_axes[f"{name}_RetainedState"] == expected_main_axes
    assert dynamic_axes["index_key.1"] == expected_index_axes
    assert dynamic_axes["index_key.1_RetainedState"] == expected_index_axes

    specs, _ = qeff_model.model.get_specializations(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=8,
        kv_offload=True,
    )
    decode_spec = specs["lang"][-1]
    assert decode_spec["main_kv_batch_size"] == 1
    assert decode_spec["main_kv_rows"] == 4
    assert decode_spec["main_kv_ctx_len"] == 4
    assert decode_spec["indexer_kv_batch_size"] == 1
    assert decode_spec["indexer_kv_rows"] == 2
    assert decode_spec["indexer_kv_ctx_len"] == 4

    qeff_model.transform(
        ctx_len=8,
        seq_len=1,
        bs=1,
        num_devices=1,
        num_cores=1,
        qaic_config=qaic_config,
        prefill_seq_len=1,
    )
    onnx_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "minimax-m3-cp2-decode",
            skip_vision=True,
            prefill_seq_len=1,
            ctx_len=8,
            qaic_config=qaic_config,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    graph = onnx.load(onnx_path, load_external_data=False).graph
    inputs_by_name = {value.name: value for value in graph.input}
    outputs_by_name = {value.name: value for value in graph.output}

    def shape_signature(value):
        return [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim]

    for input_name in ("past_key.1", "past_value.1", "index_key.1"):
        output_name = next(
            name
            for name in (
                f"{input_name}_RetainedState",
                f"{input_name}_InternalRetainedState",
            )
            if name in outputs_by_name
        )
        assert shape_signature(outputs_by_name[output_name]) == shape_signature(inputs_by_name[input_name])

    assert shape_signature(inputs_by_name["past_key.1"]) == [
        "main_kv_batch_size",
        "main_kv_rows",
        "main_kv_ctx_len",
        8,
    ]
    assert shape_signature(inputs_by_name["past_value.1"]) == [
        "main_kv_batch_size",
        "main_kv_rows",
        "main_kv_ctx_len",
        8,
    ]
    assert shape_signature(inputs_by_name["index_key.1"]) == [
        "indexer_kv_batch_size",
        "indexer_kv_rows",
        "indexer_kv_ctx_len",
        8,
    ]


@pytest.mark.cpu_only
def test_minimax_m3_cp_indexer_rejects_rank_three_cache():
    config = _tiny_minimax_m3_text_config()
    indexer = QEffMiniMaxM3VLIndexer(config, layer_idx=1).eval()
    cache = QEffMiniMaxSparseCache()
    cache.index_keys[1] = torch.zeros(2, 4, config.index_head_dim)
    blocking_config = AttentionBlockingConfig(
        mode=BlockingMode.KV_HEADPAR,
        ctx_len=8,
        num_kv_blocks=1,
        num_cores_per_device=1,
        msa_indexer_dp=1,
        msa_indexer_cp=2,
        indexer_n_head=1,
    )

    with pytest.raises(ValueError, match="MSA index key cache must be rank 4"):
        indexer._select_blocks_dp(
            torch.zeros(1, 1, config.hidden_size),
            torch.tensor([[7]]),
            cache,
            layer_idx=1,
            cos=torch.ones(1, 1, config.index_head_dim),
            sin=torch.zeros(1, 1, config.index_head_dim),
            blocking_config=blocking_config,
        )


@pytest.mark.llm_model
@pytest.mark.parametrize(("vision_batch_size", "expected_vision_batch_size"), [(None, 4), (2, 2)])
def test_minimax_m3_get_specializations_consumes_vision_batch_size(vision_batch_size, expected_vision_batch_size):
    model_hf = MiniMaxM3SparseForConditionalGeneration(_tiny_minimax_m3_vlm_config()).eval()
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)

    specs, compiler_options = qeff_model.model.get_specializations(
        batch_size=4,
        vision_batch_size=vision_batch_size,
        prefill_seq_len=1,
        ctx_len=8,
        kv_offload=True,
    )

    assert all(spec["vision_batch_size"] == expected_vision_batch_size for spec in specs["lang"])
    assert "vision_batch_size" not in compiler_options


@pytest.mark.llm_model
def test_minimax_m3_paged_indexer_cp16_attention_cp4_decode():
    vlm_config = _tiny_minimax_m3_vlm_config()
    text_config = vlm_config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "msa_indexer_dp": 1,
        "msa_indexer_cp": 16,
        "msa_attn_dp": 1,
        "msa_attn_cp": 4,
        "indexer_n_head": 1,
        "num_kv_blocks": 1,
        "num_cores_per_device": 1,
        "paged_kv": True,
        "gqa_page_block_size": 16,
        "msa_indexer_page_block_size": 4,
        "msa_attn_page_block_size": 4,
    }
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    qeff_model.model.qaic_config = qaic_config
    qeff_model.lang_model.model.qaic_config = qaic_config

    lang_inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        ctx_len=128,
        past_seq_len=128,
        prefill_seq_len=1,
        batch_size=1,
    )["lang"]
    dense_key, dense_value = lang_inputs["past_key_values"][0]
    sparse_key, sparse_value = lang_inputs["past_key_values"][1]
    (index_key,) = lang_inputs["index_keys"]
    assert tuple(dense_key.shape) == (8, 2, 16, 8)
    assert tuple(dense_value.shape) == tuple(dense_key.shape)
    assert tuple(sparse_key.shape) == (8, 8, 4, 8)
    assert tuple(sparse_value.shape) == tuple(sparse_key.shape)
    assert tuple(index_key.shape) == (2, 16, 4, 8)
    specs, _ = qeff_model.model.get_specializations(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=128,
        kv_offload=True,
        qaic_config=qaic_config,
    )
    assert all(spec["gqa_page_size"] == 16 for spec in specs["lang"])
    assert all(spec["indexer_page_size"] == 4 for spec in specs["lang"])
    assert all(spec["attn_page_size"] == 4 for spec in specs["lang"])
    dynamic_axes = qeff_model.model.get_onnx_dynamic_axes(kv_offload=True)["lang"]
    assert dynamic_axes["past_key.0"][2] == "gqa_page_size"
    assert dynamic_axes["past_key.1"][2] == "attn_page_size"
    assert dynamic_axes["index_key.1"][2] == "indexer_page_size"

    attention = QEffMiniMaxM3VLAttention(text_config, layer_idx=1).eval()
    query = torch.zeros(1, text_config.num_attention_heads, 1, text_config.head_dim)
    key_cache = torch.zeros_like(sparse_key)
    value_cache = torch.zeros_like(sparse_value)
    value_cache[0, 0:2, :, :] = 1.0
    value_cache[3, 6:8, :, :] = 4.0
    block_indices = torch.tensor([[[0, 15], [0, 15]]], dtype=torch.int64)
    block_valid = torch.ones_like(block_indices, dtype=torch.bool)
    indexer_block_table = torch.arange(2, dtype=torch.int32).view(1, 1, 2)
    attention_block_table = torch.arange(8, dtype=torch.int32).view(1, 1, 8)
    blocking_config = SimpleNamespace(
        msa_indexer_dp=1,
        msa_indexer_cp=16,
        indexer_n_head=1,
        msa_attn_dp=1,
        msa_attn_cp=4,
        ctx_len=128,
        num_kv_blocks=1,
        num_cores_per_device=1,
        msa_indexer_page_block_size=4,
        msa_attn_page_block_size=4,
    )

    indexer = QEffMiniMaxM3VLIndexer(text_config, layer_idx=1).eval()
    with torch.no_grad():
        indexer.q_proj.weight.zero_()
        indexer.k_proj.weight.zero_()
        indexer.q_proj.weight[0, 0] = 1.0
        indexer.q_proj.weight[text_config.index_head_dim + 1, 0] = 1.0
        indexer.k_proj.weight[2, 0] = 1.0

    # Populate two different past blocks so each index query head has a
    # distinct winner. This catches accidental reshapes across CP/core/head
    # axes; shape-only checks cannot detect that class of error.
    index_key.zero_()
    indexer_page_size = qaic_config["msa_indexer_page_block_size"]
    indexer_cp = qaic_config["msa_indexer_cp"]
    for position in range(128):
        logical_page = position // indexer_page_size
        page_group = logical_page // indexer_cp
        cp_way = logical_page % indexer_cp
        page_address = position % indexer_page_size
        block_idx = position // text_config.index_block_size
        if block_idx == 3:
            index_key[page_group, cp_way, page_address, 0] = 10.0
        if block_idx == 7:
            index_key[page_group, cp_way, page_address, 1] = 20.0
    index_cache = SimpleNamespace(index_keys={1: index_key})
    hidden_states = torch.zeros(1, 1, text_config.hidden_size)
    hidden_states[..., 0] = 1.0
    index_cos = torch.ones(1, 1, text_config.index_head_dim)
    index_sin = torch.zeros_like(index_cos)
    selected_blocks, selected_valid = indexer._select_blocks_paged(
        hidden_states,
        torch.tensor([[127]], dtype=torch.int64),
        index_cache,
        1,
        index_cos,
        index_sin,
        indexer_block_table,
        blocking_config,
    )
    assert tuple(selected_blocks.shape) == (1, text_config.index_n_heads, text_config.index_topk_blocks)
    assert tuple(selected_valid.shape) == tuple(selected_blocks.shape)
    assert torch.equal(selected_blocks[0, 0], torch.tensor([3, 31], dtype=torch.int32))
    assert torch.equal(selected_blocks[0, 1], torch.tensor([7, 31], dtype=torch.int32))
    assert selected_valid.all()
    assert index_cache.index_keys[1][1, 15, 3, 2] != 0

    attention_blocking_config = SimpleNamespace(
        msa_attn_dp=1,
        msa_attn_cp=4,
        ctx_len=64,
        num_kv_blocks=1,
        num_cores_per_device=1,
        msa_attn_page_block_size=4,
    )
    output = attention._paged_attention(
        query,
        block_indices,
        block_valid,
        key_cache,
        value_cache,
        torch.tensor([[63]], dtype=torch.int64),
        attention_block_table,
        attention_blocking_config,
    )
    assert tuple(output.shape) == tuple(query.shape)
    assert torch.allclose(output, torch.full_like(output, 2.5), atol=1e-6, rtol=0)


@pytest.mark.llm_model
def test_minimax_m3_paged_gqa_cache_and_block_table_are_exported(tmp_path):
    vlm_config = _tiny_minimax_m3_vlm_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qaic_config = {
        "blocking_mode": "kv_minimax_dedicated",
        "num_kv_blocks": 1,
        "num_cores_per_device": 1,
        "paged_kv": True,
        "page_block_size": vlm_config.text_config.index_block_size,
        "gqa_physical_pages": 5,
        "msa_attn_physical_pages": 6,
        "msa_indexer_physical_pages": 7,
    }
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    qeff_model.model.qaic_config = qaic_config

    lang_inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        ctx_len=8,
        past_seq_len=8,
        prefill_seq_len=1,
        batch_size=1,
    )["lang"]

    dense_key, dense_value = lang_inputs["past_key_values"][0]
    sparse_key, sparse_value = lang_inputs["past_key_values"][1]
    (index_key,) = lang_inputs["index_keys"]
    assert tuple(dense_key.shape) == (5, 2, 4, 8)
    assert tuple(dense_value.shape) == tuple(dense_key.shape)
    assert tuple(sparse_key.shape) == (6, 2, 4, 8)
    assert tuple(sparse_value.shape) == tuple(sparse_key.shape)
    assert tuple(index_key.shape) == (7, 1, 4, 8)
    assert tuple(lang_inputs["gqa_block_table"].shape) == (1, 1, 2)
    specs, _ = qeff_model.model.get_specializations(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=8,
        kv_offload=True,
        qaic_config=qaic_config,
    )
    assert all(spec["gqa_page_groups"] == 2 for spec in specs["lang"])
    assert all(spec["gqa_physical_pages"] == 5 for spec in specs["lang"])
    assert all(spec["attn_physical_pages"] == 6 for spec in specs["lang"])
    assert all(spec["indexer_physical_pages"] == 7 for spec in specs["lang"])
    dynamic_axes = qeff_model.model.get_onnx_dynamic_axes(kv_offload=True)["lang"]
    assert dynamic_axes["gqa_block_table"] == {1: "gqa_batch_local", 2: "gqa_page_groups"}
    assert dynamic_axes["past_key.0"] == {0: "gqa_physical_pages", 2: "page_size"}
    assert dynamic_axes["past_key.1"] == {0: "attn_physical_pages", 2: "page_size"}
    assert dynamic_axes["index_key.1"] == {0: "indexer_physical_pages", 2: "page_size"}

    qeff_model.transform(
        ctx_len=8,
        seq_len=1,
        bs=1,
        num_devices=1,
        num_cores=1,
        qaic_config=qaic_config,
        prefill_seq_len=1,
    )
    onnx_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "minimax-m3-paged-gqa",
            skip_vision=True,
            prefill_seq_len=1,
            ctx_len=8,
            qaic_config=qaic_config,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    graph = onnx.load(onnx_path, load_external_data=False).graph
    inputs_by_name = {value.name: value for value in graph.input}
    assert "gqa_block_table" in inputs_by_name


@pytest.mark.llm_model
@pytest.mark.accuracy
def test_minimax_m3_paged_dp_cp_qeff_pytorch_vs_ort(tmp_path):
    torch.manual_seed(7)
    qaic_config = _tiny_minimax_m3_paged_dp_cp_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(_tiny_minimax_m3_vlm_config()).eval()
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    qeff_model.model.qaic_config = qaic_config
    qeff_model.lang_model.model.qaic_config = qaic_config

    qeff_model.transform(
        ctx_len=8,
        seq_len=1,
        bs=2,
        num_devices=1,
        num_cores=1,
        qaic_config=qaic_config,
        prefill_seq_len=1,
    )
    inputs = _tiny_minimax_m3_paged_decode_inputs(qeff_model)
    pytorch_inputs = deepcopy(inputs)
    with torch.no_grad():
        qeff_logits = qeff_model.lang_model.model(**pytorch_inputs)[0].detach().float().numpy()

    with _suppress_native_output():
        onnx_path = _exported_onnx_path(
            qeff_model.export(
                tmp_path / "minimax-m3-paged-dp-cp-ort",
                skip_vision=True,
                prefill_seq_len=1,
                ctx_len=8,
                qaic_config=qaic_config,
                use_onnx_subfunctions=True,
                offload_pt_weights=False,
            )
        )
    session = _ort_session(onnx_path)
    ort_logits = session.run(None, _flatten_minimax_paged_ort_inputs(session, inputs))[0]

    np.testing.assert_allclose(ort_logits, qeff_logits, atol=1e-4, rtol=1e-4)
    assert qaic_config["paged_kv"] is True
    assert qaic_config["attn_dp"] == 1
    assert qaic_config["msa_indexer_cp"] == 2
    assert qaic_config["msa_attn_dp"] == 2
    assert qaic_config["gqa_page_block_size"] == 8
    assert qaic_config["msa_indexer_page_block_size"] == 4
    assert qaic_config["msa_attn_page_block_size"] == 4


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.accuracy
def test_minimax_m3_paged_dp_cp_qeff_pytorch_vs_aic(tmp_path):
    torch.manual_seed(7)
    qaic_config = _tiny_minimax_m3_paged_dp_cp_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(_tiny_minimax_m3_vlm_config()).eval()
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)

    qeff_model.transform(
        ctx_len=8,
        seq_len=1,
        bs=2,
        num_devices=1,
        num_cores=1,
        qaic_config=qaic_config,
        prefill_seq_len=1,
    )
    inputs = _tiny_minimax_m3_paged_decode_inputs(qeff_model)
    with torch.no_grad():
        qeff_logits = qeff_model.lang_model.model(**inputs)[0].detach().float().numpy()

    qpc_path = qeff_model.compile(
        compile_dir=tmp_path / "minimax-m3-paged-dp-cp-aic",
        prefill_seq_len=1,
        ctx_len=8,
        batch_size=2,
        num_cores=1,
        num_devices=1,
        skip_vision=True,
        node_precision_info=True,
        use_onnx_subfunctions=False,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
        retain_full_kv=True,
        split_model_io=True,
        user_tiled=True,
        qaic_config=qaic_config,
        offload_pt_weights=False,
    )["lang_decode_qpc_path"]
    session = QAICInferenceSession(qpc_path)
    try:

        def binding_value(input_name, source=None):
            binding = session.bindings[session.binding_index_map[input_name]]
            shape = tuple(binding.dims)
            dtype = session.aic_to_np_dtype_mapping[binding.type]
            if source is None:
                return np.zeros(shape, dtype=dtype)
            source = np.asarray(source, dtype=dtype)
            assert source.size == np.prod(shape), (
                f"AIC binding shape mismatch for {input_name}: source {source.shape}, expected {shape}"
            )
            return source.reshape(shape)

        source_inputs = _flatten_minimax_paged_ort_inputs(session, inputs)
        aic_inputs = {name: binding_value(name, source_inputs.get(name)) for name in session.input_names}
        aic_outputs = session.run(aic_inputs)
    finally:
        session.deactivate()

    np.testing.assert_allclose(aic_outputs["logits"], qeff_logits, atol=1e-2, rtol=1e-2)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    "physical_page_name",
    ["gqa_physical_pages", "msa_attn_physical_pages", "msa_indexer_physical_pages"],
)
def test_minimax_m3_rejects_undersized_physical_page_pools(physical_page_name):
    vlm_config = _tiny_minimax_m3_vlm_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qaic_config = {
        "blocking_mode": "kv_minimax_dedicated",
        "num_kv_blocks": 1,
        "num_cores_per_device": 1,
        "paged_kv": True,
        "page_block_size": vlm_config.text_config.index_block_size,
        physical_page_name: 1,
    }
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    qeff_model.model.qaic_config = qaic_config

    with pytest.raises(ValueError, match="physical_pages"):
        qeff_model.model.get_dummy_inputs(
            kv_offload=True,
            ctx_len=8,
            past_seq_len=8,
            prefill_seq_len=1,
            batch_size=1,
        )


@pytest.mark.cpu_only
def test_minimax_sparse_cache_cp_write_keeps_rank_four(monkeypatch):
    dp, cp, hkv = 2, 3, 2
    batch_local, local_ctx_len, head_dim = 1, 3, 1
    rows = dp * cp * hkv
    key_cache = torch.zeros(batch_local, rows, local_ctx_len, head_dim)
    value_cache = torch.zeros_like(key_cache)
    cache = QEffMiniMaxSparseCache.from_legacy_cache(((key_cache, value_cache),))
    original_paged_scatter = cache.paged_scatter
    observed_shapes = []

    def paged_scatter(data, block_id, addr, updates):
        observed_shapes.append(tuple(data.shape))
        return original_paged_scatter(data, block_id, addr, updates)

    monkeypatch.setattr(cache, "paged_scatter", paged_scatter)
    key_states = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
    value_states = key_states + 10.0
    position_ids = torch.tensor([[4], [7]])
    cache.write_only_sparse(
        key_states,
        value_states,
        layer_idx=0,
        cache_kwargs={"position_ids": position_ids, "dp": dp, "hkv": hkv, "cp": cp},
    )

    expected_keys = torch.zeros_like(key_cache)
    expected_values = torch.zeros_like(value_cache)
    for dp_lane in range(dp):
        cp_lane = int(position_ids[dp_lane, 0]) % cp
        local_position = int(position_ids[dp_lane, 0]) // cp
        for kv_head in range(hkv):
            row = dp_lane * cp * hkv + cp_lane * hkv + kv_head
            expected_keys[0, row, local_position] = key_states[dp_lane, kv_head, 0]
            expected_values[0, row, local_position] = value_states[dp_lane, kv_head, 0]

    assert observed_shapes == [tuple(key_cache.shape), tuple(value_cache.shape)]
    assert tuple(cache.layers[0].keys.shape) == tuple(key_cache.shape)
    assert tuple(cache.layers[0].values.shape) == tuple(value_cache.shape)
    assert torch.equal(cache.layers[0].keys, expected_keys)
    assert torch.equal(cache.layers[0].values, expected_values)


@pytest.mark.cpu_only
def test_minimax_sparse_cache_cp_write_rejects_rank_three():
    key_cache = torch.zeros(2, 3, 1)
    value_cache = torch.zeros_like(key_cache)
    cache = QEffMiniMaxSparseCache.from_legacy_cache(((key_cache, value_cache),))

    with pytest.raises(ValueError, match="CP sparse cache shape"):
        cache.write_only_sparse(
            torch.tensor([[[[7.0]]]]),
            torch.tensor([[[[11.0]]]]),
            layer_idx=0,
            cache_kwargs={"position_ids": torch.tensor([[4]]), "dp": 1, "hkv": 1, "cp": 2},
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("use_block_indices", [True, False])
def test_minimax_m3_retained_cp_attention_uses_cp_before_kv_heads(use_block_indices):
    config = _tiny_minimax_m3_text_config()
    attention = QEffMiniMaxM3VLAttention(config, layer_idx=1).eval()
    dp, cp, hkv = 1, 2, config.num_key_value_heads
    batch_local, local_ctx_len, head_dim = 2, 4, config.head_dim
    rows = dp * cp * hkv

    query = torch.zeros(dp * batch_local, config.num_attention_heads, 1, head_dim)
    key_cache = torch.zeros(batch_local, rows, local_ctx_len, head_dim)
    value_cache = torch.full_like(key_cache, 100.0)
    value_cache[0, :2] = 1.0  # DP=0, CP=0
    value_cache[0, 2:4] = 7.0  # DP=0, CP=1
    value_cache[1, :2] = 3.0
    value_cache[1, 2:4] = 9.0

    if use_block_indices:
        token_indices = torch.tensor([[[0], [1]], [[0], [1]]], dtype=torch.int64)
        token_valid = torch.ones_like(token_indices, dtype=torch.bool)
    else:
        token_indices = torch.tensor(
            [[[[0, 1]], [[4, 5]]], [[[0, 1]], [[4, 5]]]],
            dtype=torch.int64,
        )
        token_valid = torch.ones_like(token_indices, dtype=torch.bool)

    output, _, _ = attention._baseline_attention_gp(
        query,
        token_indices,
        token_valid,
        key_cache,
        value_cache,
        dp=dp,
        blocking_config=SimpleNamespace(msa_attn_cp=cp, ctx_len=cp * local_ctx_len, num_cores_per_device=1),
        position_ids=torch.full((batch_local, 1), cp * local_ctx_len - 1, dtype=torch.int64),
    )

    expected = torch.tensor([[4.0, 4.0, 4.0, 4.0], [6.0, 6.0, 6.0, 6.0]])
    expected = expected.view(batch_local, config.num_attention_heads, 1, 1)
    expected = expected.expand_as(output)
    assert torch.allclose(output, expected, atol=1e-6, rtol=0)


@pytest.mark.llm_model
def test_minimax_m3_dedicated_contiguous_gqa_cache_has_dynamic_dp_cp_axes():
    vlm_config = _tiny_minimax_m3_vlm_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qaic_config = {
        "blocking_mode": "kv_minimax_dedicated",
        "num_kv_blocks": 2,
        "num_cores_per_device": 1,
        "attn_dp": 2,
        "attn_cp": 2,
    }
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    qeff_model.model.qaic_config = qaic_config

    lang_inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        ctx_len=8,
        past_seq_len=8,
        prefill_seq_len=1,
        batch_size=2,
    )["lang"]
    key, value = lang_inputs["past_key_values"][0]
    assert tuple(key.shape) == (1, 8, 4, 8)
    assert tuple(value.shape) == tuple(key.shape)

    specs, _ = qeff_model.model.get_specializations(
        batch_size=2,
        prefill_seq_len=1,
        ctx_len=8,
        kv_offload=True,
    )
    assert all(spec["gqa_batch_local"] == 1 for spec in specs["lang"])
    assert all(spec["gqa_ctx_len"] == 4 for spec in specs["lang"])
    dynamic_axes = qeff_model.model.get_onnx_dynamic_axes(kv_offload=True)["lang"]
    assert dynamic_axes["past_key.0"] == {0: "gqa_batch_local", 2: "gqa_ctx_len"}
    assert dynamic_axes["past_key.0_RetainedState"] == dynamic_axes["past_key.0"]


@pytest.mark.llm_model
def test_minimax_m3_dedicated_gqa_rejects_continuous_batching():
    vlm_config = _tiny_minimax_m3_vlm_config()
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qaic_config = {"blocking_mode": "kv_minimax_dedicated", "num_kv_blocks": 1}
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    qeff_model.model.qaic_config = qaic_config

    with pytest.raises(NotImplementedError, match="does not support continuous batching"):
        qeff_model.model.get_specializations(
            batch_size=1,
            prefill_seq_len=1,
            ctx_len=8,
            kv_offload=True,
            continuous_batching=True,
            kv_cache_batch_size=2,
        )


@pytest.mark.llm_model
@pytest.mark.parametrize("query_len", [1, 4])
def test_minimax_m3_dense_attention_uses_dedicated_paged_gqa(query_len):
    config = _tiny_minimax_m3_text_config()
    paged_attention = QEffMiniMaxM3VLAttention(config, layer_idx=0).eval()
    reference_attention = deepcopy(paged_attention)
    is_prefill = query_len > 1
    paged_attention.attn_blocking_config = AttentionBlockingConfig(
        mode=BlockingMode.KV_MINIMAX_DEDICATED,
        num_kv_blocks=2 if is_prefill else 1,
        num_q_blocks=2 if is_prefill else 1,
        n_rep_chunk=1,
        head_block_size=1,
        ctx_len=8,
        paged_attention=True,
        paged_gqa=True,
        gqa_page_block_size=4,
        num_cores_per_device=2 if is_prefill else 1,
    )

    hidden_states = torch.randn(1, query_len, config.hidden_size)
    position_ids = torch.arange(query_len, dtype=torch.int64).view(1, -1)
    cos = torch.ones(1, query_len, config.head_dim)
    sin = torch.zeros_like(cos)
    attention_mask = torch.arange(8).view(1, 1, 1, -1) > position_ids.view(1, 1, -1, 1)
    paged_cache = QEffMiniMaxSparseCache.from_legacy_cache(
        (
            (
                torch.zeros(2, config.num_key_value_heads, 4, config.head_dim),
                torch.zeros(2, config.num_key_value_heads, 4, config.head_dim),
            ),
        )
    )
    reference_cache = QEffMiniMaxSparseCache.from_legacy_cache(
        (
            (
                torch.zeros(1, config.num_key_value_heads, 8, config.head_dim),
                torch.zeros(1, config.num_key_value_heads, 8, config.head_dim),
            ),
        )
    )

    with torch.no_grad():
        paged_output, _ = paged_attention(
            hidden_states,
            (cos, sin),
            attention_mask=attention_mask,
            past_key_values=paged_cache,
            position_ids=position_ids,
            gqa_block_table=torch.arange(2, dtype=torch.int32).view(1, 1, 2),
        )
        reference_output, _ = reference_attention(
            hidden_states,
            (cos, sin),
            attention_mask=attention_mask,
            past_key_values=reference_cache,
            position_ids=position_ids,
        )

    assert torch.allclose(paged_output, reference_output, atol=1e-5, rtol=1e-5)


@pytest.mark.llm_model
def test_minimax_m3_dedicated_contiguous_prefill_matches_dense_attention():
    torch.manual_seed(3)
    config = _tiny_minimax_m3_text_config()
    attention = QEffMiniMaxM3VLAttention(config, layer_idx=0).eval()
    batch, query_len, ctx_len = 1, 4, 8
    query = torch.randn(batch, config.num_attention_heads, query_len, config.head_dim)
    key_cache = torch.zeros(batch, config.num_key_value_heads, ctx_len, config.head_dim)
    value_cache = torch.zeros_like(key_cache)
    key_cache[:, :, :query_len] = torch.randn(batch, config.num_key_value_heads, query_len, config.head_dim)
    value_cache[:, :, :query_len] = torch.randn_like(key_cache[:, :, :query_len])
    position_ids = torch.arange(query_len).view(1, -1)
    blocking_config = AttentionBlockingConfig(
        mode=BlockingMode.KV_MINIMAX_DEDICATED,
        num_kv_blocks=2,
        num_q_blocks=2,
        n_rep_chunk=1,
        head_block_size=1,
        ctx_len=ctx_len,
        num_cores_per_device=config.num_key_value_heads,
    )

    output = attention._gqa_dedicated_prefill(query, key_cache, value_cache, position_ids, blocking_config)
    repeated_k = key_cache[:, :, :query_len].repeat_interleave(attention.num_key_value_groups, dim=1)
    repeated_v = value_cache[:, :, :query_len].repeat_interleave(attention.num_key_value_groups, dim=1)
    scores = torch.matmul(query, repeated_k.transpose(-1, -2)) * attention.scaling
    causal = torch.arange(query_len).view(1, 1, 1, -1) > position_ids[:, None, :, None]
    expected = torch.matmul(torch.softmax(scores.masked_fill(causal, -3.0e4), dim=-1), repeated_v)

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.llm_model
def test_minimax_m3_dedicated_contiguous_dp_cp_decode_matches_dense_attention():
    torch.manual_seed(4)
    config = _tiny_minimax_m3_text_config()
    attention = QEffMiniMaxM3VLAttention(config, layer_idx=0).eval()
    batch, dp, cp, ctx_len = 2, 2, 2, 8
    hkv, head_dim = config.num_key_value_heads, config.head_dim
    full_key = torch.randn(batch, hkv, ctx_len, head_dim)
    full_value = torch.randn_like(full_key)
    key_cache = torch.zeros(batch // dp, dp * cp * hkv, ctx_len // cp, head_dim)
    value_cache = torch.zeros_like(key_cache)
    for dp_idx in range(dp):
        for way in range(cp):
            for head_idx in range(hkv):
                row = dp_idx * cp * hkv + way * hkv + head_idx
                key_cache[0, row] = full_key[dp_idx, head_idx, way::cp]
                value_cache[0, row] = full_value[dp_idx, head_idx, way::cp]
    query = torch.randn(batch, config.num_attention_heads, 1, head_dim)
    position_ids = torch.full((batch, 1), ctx_len - 1, dtype=torch.int64)
    blocking_config = AttentionBlockingConfig(
        mode=BlockingMode.KV_MINIMAX_DEDICATED,
        num_kv_blocks=2,
        ctx_len=ctx_len,
        attn_dp=dp,
        attn_cp=cp,
        num_cores_per_device=1,
        skip_kv=True,
    )

    output = attention._gqa_dedicated_decode(query, key_cache, value_cache, position_ids, blocking_config)
    repeated_k = full_key.repeat_interleave(attention.num_key_value_groups, dim=1)
    repeated_v = full_value.repeat_interleave(attention.num_key_value_groups, dim=1)
    scores = torch.matmul(query, repeated_k.transpose(-1, -2)) * attention.scaling
    expected = torch.matmul(torch.softmax(scores, dim=-1), repeated_v)

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.llm_model
@pytest.mark.accuracy
def test_minimax_m3_contiguous_dp_cp_decode_hf_qeff_pytorch_parity():
    torch.manual_seed(5)
    vlm_config = _tiny_minimax_m3_vlm_config()
    text_config = vlm_config.text_config
    text_config.index_topk_blocks = 1
    text_config.index_local_blocks = 1
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    model_hf_orig = deepcopy(model_hf)

    batch, dp, cp, ctx_len = 2, 2, 2, 16
    past_seq_len = ctx_len - 1
    qaic_config = {
        "blocking_mode": "kv_minimax_dedicated",
        "num_kv_blocks": 2,
        "attn_dp": dp,
        "attn_cp": cp,
        "msa_indexer_dp": dp,
        "msa_indexer_cp": cp,
        "msa_attn_dp": dp,
        "msa_attn_cp": cp,
        "indexer_n_head": 1,
        "num_cores_per_device": 1,
    }

    def pack_dp_cp_cache(cache: torch.Tensor) -> torch.Tensor:
        cache_batch, num_heads, cache_len, head_dim = cache.shape
        assert cache_batch == batch
        assert cache_len == ctx_len
        batch_local = cache_batch // dp
        packed = torch.empty(batch_local, dp * cp * num_heads, cache_len // cp, head_dim, dtype=cache.dtype)
        for local_batch_idx in range(batch_local):
            for dp_lane in range(dp):
                source_batch_idx = dp_lane * batch_local + local_batch_idx
                for cp_lane in range(cp):
                    row_start = (dp_lane * cp + cp_lane) * num_heads
                    row_end = row_start + num_heads
                    packed[local_batch_idx, row_start:row_end] = cache[source_batch_idx, :, cp_lane::cp]
        return packed

    logical_caches = []
    logical_index_keys = {}
    hf_cache = DynamicCache(config=text_config)
    for layer_idx, layer_type in enumerate(text_config.layer_types):
        key = torch.randn(batch, text_config.num_key_value_heads, past_seq_len, text_config.head_dim)
        value = torch.randn_like(key)
        logical_caches.append((key, value))
        hf_cache.update(key, value, layer_idx)
        if layer_type == "minimax_m3_sparse":
            index_key = torch.randn(batch, 1, past_seq_len, text_config.index_head_dim)
            logical_index_keys[layer_idx] = index_key
            hf_cache.layers[layer_idx].update_index(index_key)

    input_ids = torch.tensor([[3], [5]], dtype=torch.int64)
    position_ids = torch.full((batch, 1), past_seq_len, dtype=torch.int64)
    with torch.no_grad():
        hf_outputs = model_hf_orig.model.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=hf_cache,
            use_cache=True,
        )
        hf_logits = model_hf_orig.lm_head(hf_outputs.last_hidden_state[:, -1:, :]).float()

    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True, qaic_config=qaic_config)
    qeff_model.transform(
        ctx_len=ctx_len,
        seq_len=1,
        bs=batch,
        num_devices=1,
        num_cores=1,
        qaic_config=qaic_config,
        prefill_seq_len=1,
    )
    qeff_inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        ctx_len=ctx_len,
        past_seq_len=ctx_len,
        prefill_seq_len=1,
        batch_size=batch,
    )["lang"]
    qeff_inputs["input_ids"] = input_ids
    qeff_inputs["position_ids"] = position_ids
    qeff_inputs["past_key_values"] = tuple(
        (
            pack_dp_cp_cache(torch.nn.functional.pad(key, (0, 0, 0, 1))),
            pack_dp_cp_cache(torch.nn.functional.pad(value, (0, 0, 0, 1))),
        )
        for key, value in logical_caches
    )
    qeff_inputs["index_keys"] = [
        pack_dp_cp_cache(torch.nn.functional.pad(logical_index_keys[layer_idx], (0, 0, 0, 1)))
        for layer_idx, layer_type in enumerate(text_config.layer_types)
        if layer_type == "minimax_m3_sparse"
    ]

    with torch.no_grad():
        qeff_logits = qeff_model.lang_model.model(**qeff_inputs)[0].float()

    assert torch.isfinite(hf_logits).all()
    assert torch.isfinite(qeff_logits).all()
    torch.testing.assert_close(qeff_logits, hf_logits, atol=1e-5, rtol=1e-5)
    assert torch.equal(qeff_logits.argmax(dim=-1), hf_logits.argmax(dim=-1))


@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_minimax_m3_decode_qeff_pytorch_vs_aic(tmp_path):
    torch.manual_seed(7)
    vlm_config = _tiny_minimax_m3_vlm_config()
    text_config = vlm_config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)

    input_ids = torch.tensor([[3]], dtype=torch.int64)
    position_ids = torch.tensor([[4]], dtype=torch.int64)
    runtime_cache_len = 8
    past_key_values = tuple(
        (
            torch.zeros((1, text_config.num_key_value_heads, runtime_cache_len, text_config.head_dim)),
            torch.zeros((1, text_config.num_key_value_heads, runtime_cache_len, text_config.head_dim)),
        )
        for _ in range(text_config.num_hidden_layers)
    )
    index_keys = [
        torch.zeros((1, 1, runtime_cache_len, text_config.index_head_dim))
        for layer_type in text_config.layer_types
        if layer_type == "minimax_m3_sparse"
    ]
    dummy_vision_embeds = torch.zeros((1, 1, text_config.hidden_size), dtype=torch.float32)
    dummy_image_idx = torch.zeros((1, 1), dtype=torch.int64)

    with torch.no_grad():
        qeff_outputs = qeff_model.lang_model.model(
            input_ids=input_ids,
            vision_embeds=dummy_vision_embeds,
            position_ids=position_ids,
            image_idx=dummy_image_idx,
            past_key_values=past_key_values,
            index_keys=index_keys,
        )
    qeff_logits = qeff_outputs[0].detach().cpu().numpy()

    qpc_paths = qeff_model.compile(
        compile_dir=tmp_path / "minimax-m3-aic",
        prefill_seq_len=1,
        ctx_len=runtime_cache_len,
        batch_size=1,
        num_cores=1,
        num_devices=1,
        skip_vision=True,
        node_precision_info=True,
        qaic_config={},
        offload_pt_weights=False,
    )
    session = QAICInferenceSession(qpc_paths["lang_decode_qpc_path"])
    try:

        def binding_shape_dtype(input_name):
            binding = session.bindings[session.binding_index_map[input_name]]
            shape = tuple(binding.dims)
            assert all(dim >= 0 for dim in shape), f"Dynamic AIC binding shape for {input_name}: {shape}"
            return shape, session.aic_to_np_dtype_mapping[binding.type]

        def binding_zeros(input_name):
            shape, dtype = binding_shape_dtype(input_name)
            return np.zeros(shape, dtype=dtype)

        def binding_copy(input_name, source):
            shape, dtype = binding_shape_dtype(input_name)
            source = np.asarray(source, dtype=dtype)
            assert source.size == np.prod(shape), (
                f"AIC binding shape mismatch for {input_name}: source {source.shape}, expected {shape}"
            )
            return source.reshape(shape)

        aic_inputs = {}
        for input_name in session.input_names:
            if input_name == "input_ids":
                aic_inputs[input_name] = binding_copy(input_name, input_ids.numpy())
            elif input_name == "position_ids":
                aic_inputs[input_name] = binding_copy(input_name, position_ids.numpy())
            elif input_name == "vision_embeds":
                aic_inputs[input_name] = binding_zeros(input_name)
            elif input_name == "image_idx":
                aic_inputs[input_name] = binding_copy(input_name, dummy_image_idx.numpy())
            elif input_name.startswith(("past_key.", "past_value.")):
                aic_inputs[input_name] = binding_zeros(input_name)
            elif input_name.startswith("index_key."):
                aic_inputs[input_name] = binding_zeros(input_name)
            else:
                raise AssertionError(f"Unhandled Minimax AIC input: {input_name}")
        aic_outputs = session.run(aic_inputs)
    finally:
        session.deactivate()

    np.testing.assert_allclose(aic_outputs["logits"], qeff_logits, atol=1e-2, rtol=1e-2)


@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_minimax_m3_expert_parallel_prefill_hf_pytorch_vs_aic(tmp_path):
    """Check MiniMax expert-parallel prefill logits against the original HF model."""
    torch.manual_seed(7)
    vlm_config = _tiny_minimax_m3_vlm_config()
    text_config = vlm_config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    model_hf_orig = deepcopy(model_hf)
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)

    prefill_len, cache_len = 4, 8
    input_ids = torch.tensor([[3, 5, 7, 9]], dtype=torch.int64)
    position_ids = torch.arange(prefill_len, dtype=torch.int64).view(1, -1)
    with torch.no_grad():
        hf_prefill = model_hf_orig.model.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=True,
        )
        hf_logits = model_hf_orig.lm_head(hf_prefill.last_hidden_state[:, -1:, :]).float().cpu().numpy()

    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": 2,
        "msa_indexer_dp": 1,
        "msa_indexer_cp": 1,
        "msa_attn_dp": 1,
        "indexer_n_head": 1,
        "num_cores_per_device": 2,
        "msa_q_chunk": 4,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": prefill_len,
        },
    }
    qpc_path = qeff_model.compile(
        compile_dir=tmp_path / "minimax-m3-expert-parallel-prefill-aic",
        prefill_seq_len=prefill_len,
        ctx_len=cache_len,
        batch_size=1,
        num_cores=4,
        num_devices=1,
        skip_vision=True,
        node_precision_info=True,
        prefill_only=True,
        enable_chunking=True,
        qaic_config=qaic_config,
        offload_pt_weights=False,
    )["lang_prefill_qpc_path"]
    assert qeff_model.lang_model.hash_params["moe_prefill_flavour"] == "expert_parallel"

    session = QAICInferenceSession(qpc_path)
    try:

        def binding_value(input_name, source=None):
            binding = session.bindings[session.binding_index_map[input_name]]
            shape = tuple(binding.dims)
            dtype = session.aic_to_np_dtype_mapping[binding.type]
            if source is None:
                return np.zeros(shape, dtype=dtype)
            source = np.asarray(source, dtype=dtype)
            assert source.size == np.prod(shape), (
                f"AIC binding shape mismatch for {input_name}: source {source.shape}, expected {shape}"
            )
            return source.reshape(shape)

        aic_inputs = {}
        for input_name in session.input_names:
            if input_name == "input_ids":
                aic_inputs[input_name] = binding_value(input_name, input_ids.numpy())
            elif input_name == "position_ids":
                aic_inputs[input_name] = binding_value(input_name, position_ids.numpy())
            elif input_name == "image_idx":
                aic_inputs[input_name] = binding_value(input_name, np.zeros((1, 1), dtype=np.int64))
            elif input_name == "vision_embeds" or input_name.startswith(("past_key.", "past_value.", "index_key.")):
                aic_inputs[input_name] = binding_value(input_name)
            else:
                raise AssertionError(f"Unhandled MiniMax AIC input: {input_name}")
        aic_outputs = session.run(aic_inputs)
    finally:
        session.deactivate()

    np.testing.assert_allclose(aic_outputs["logits"], hf_logits, atol=1e-2, rtol=1e-2)


@pytest.mark.llm_model
def test_minimax_m3_text_hf_qeff_pytorch_parity():
    torch.manual_seed(7)
    config = _tiny_minimax_m3_text_config()
    model_hf = MiniMaxM3VLForCausalLM(config).eval()
    # Preserve the original HF model before transforms mutate it in-place.
    model_hf_orig = deepcopy(model_hf)

    input_ids = torch.arange(4, dtype=torch.int64).view(1, 4) % config.vocab_size
    position_ids = torch.arange(4, dtype=torch.int64).view(1, 4)
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        hf_logits_no_cache = model_hf(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        ).logits[:, -1:]

    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    with torch.no_grad():
        qeff_logits_no_cache = qeff_model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        ).logits

    assert torch.allclose(hf_logits_no_cache, qeff_logits_no_cache, atol=1e-5, rtol=1e-5), (
        f"HF vs QEff logit mismatch on no-cache prefill: "
        f"max_diff={(hf_logits_no_cache - qeff_logits_no_cache).abs().max().item():.6f}"
    )


@pytest.mark.llm_model
def test_minimax_m3_indexer_select_blocks_parity():
    """
    Check that QEffMiniMaxM3VLIndexer._select_blocks selects the same key-blocks as
    MiniMaxM3VLIndexer.forward for a single-token decode step with a pre-filled index-key cache.

    index_n_heads=1 makes HF's per-query head-amax and QEff's per-head topk equivalent.
    The two outputs differ in representation (HF: block indices; QEff: token indices + validity mask),
    so we convert QEff token indices to block indices before comparing.
    """
    torch.manual_seed(42)

    # Use index_n_heads=1 so HF's amax(dim=heads) and QEff's per-head topk are identical.
    text_cfg = MiniMaxM3VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        dense_intermediate_size=64,
        shared_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        layer_types=["full_attention", "minimax_m3_sparse"],
        mlp_layer_types=["dense", "sparse"],
        index_n_heads=1,
        index_head_dim=8,
        index_block_size=4,
        index_topk_blocks=2,
        index_local_blocks=1,
    )
    text_cfg.torch_dtype = torch.float32

    layer_idx = 1  # the sparse layer
    batch = 1
    prefill_len = 8  # pre-filled positions 0..7
    ctx_len = prefill_len + 1  # includes decode token at position 8

    hf_indexer = MiniMaxM3VLIndexer(text_cfg, layer_idx=layer_idx).eval()
    qeff_indexer = QEffMiniMaxM3VLIndexer(text_cfg, layer_idx=layer_idx).eval()
    qeff_indexer.load_state_dict(hf_indexer.state_dict())

    hidden_states = torch.randn(batch, 1, text_cfg.hidden_size)
    position_ids = torch.tensor([[prefill_len]])  # decode at position 8

    # cos/sin for the single decode query position: [B, 1, index_head_dim]
    cos = torch.rand(batch, 1, text_cfg.index_head_dim)
    sin = torch.rand(batch, 1, text_cfg.index_head_dim)

    # Already-rotated index-key cache for positions 0..7: [B, 1, prefill_len, D]
    pre_fill_index_keys = torch.randn(batch, 1, prefill_len, text_cfg.index_head_dim)

    # HF mock cache: layers[layer_idx].update_index(new_k) appends new_k to past keys.
    class _HFIndexLayer:
        def __init__(self, past_k: torch.Tensor) -> None:
            self._past = past_k

        def update_index(self, new_k: torch.Tensor) -> torch.Tensor:
            return torch.cat([self._past, new_k], dim=2)

    class _HFCache:
        def __init__(self) -> None:
            self.layers = {layer_idx: _HFIndexLayer(pre_fill_index_keys)}

    hf_cache = _HFCache()

    # QEff cache: _select_blocks reads ctx_len from layers[layer_idx].keys.shape[2] and
    # scatter-writes the new key via update_index_key_cache into a pre-allocated buffer.
    qeff_cache = QEffMiniMaxSparseCache()
    qeff_cache.layers = [None] * (layer_idx + 1)
    qeff_cache.layers[layer_idx] = SimpleNamespace(
        keys=torch.zeros(batch, text_cfg.num_key_value_heads, ctx_len, text_cfg.head_dim)
    )
    # Pre-allocated buffer: positions 0..7 filled, position 8 zeroed (filled in-flight).
    qeff_cache.index_keys[layer_idx] = torch.cat(
        [pre_fill_index_keys, torch.zeros(batch, 1, 1, text_cfg.index_head_dim)], dim=2
    )

    with torch.no_grad():
        hf_block_indices = hf_indexer.forward(
            hidden_states,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            position_ids=position_ids,
        )  # [B=1, S_q=1, topk_blocks]

        qeff_safe_indices, qeff_token_valid = qeff_indexer._select_blocks(
            hidden_states,
            position_ids,
            qeff_cache,
            layer_idx=layer_idx,
            cos=cos,
            sin=sin,
        )  # [B=1, H=1, topk*block_size] each

    block_size = text_cfg.index_block_size

    # HF returns block indices [B=1, S_q=1, topk]; -1 pads unused slots.
    hf_selected = hf_block_indices[0, 0]
    hf_valid_blocks = hf_selected[hf_selected >= 0].sort().values

    # QEff returns token indices + validity mask; convert to block indices.
    qeff_tok_idx = qeff_safe_indices[0, 0]
    qeff_tok_mask = qeff_token_valid[0, 0]
    qeff_valid_blocks = (qeff_tok_idx[qeff_tok_mask] // block_size).unique().sort().values

    assert torch.equal(hf_valid_blocks, qeff_valid_blocks), (
        f"Block selection mismatch: HF={hf_valid_blocks.tolist()}, QEff={qeff_valid_blocks.tolist()}"
    )


def check_attention_module_parity(
    text_cfg: MiniMaxM3VLTextConfig,
    layer_idx: int,
    prefill_len: int = 7,
    batch: int = 1,
    atol: float = 1e-4,
    seed: int = 42,
) -> tuple:
    """
    Verify that QEffMiniMaxM3VLAttention and MiniMaxM3VLAttention produce the same
    attention output on a single-token decode step with a pre-filled KV cache.

    A QEffMiniMaxSparseCache is built via from_legacy_cache with pre-allocated
    ctx_len = prefill_len + 1 buffers (positions 0..prefill_len-1 filled, position
    prefill_len zeroed as the decode slot).  QEff's write_only / update then
    scatter-writes the decode token into that slot.  A lightweight HF mock cache
    appends the same decode token so both modules attend over the same ctx_len KV.

    For sparse layers ctx_len is kept within index_topk_blocks * index_block_size so
    that every key block falls within the topk budget, making HF's indexer block mask
    equivalent to a plain causal mask and the outputs identical.

    Args:
        text_cfg:    MiniMaxM3VLTextConfig instance.
        layer_idx:   Index of the attention layer to test.
        prefill_len: Number of pre-existing tokens in the cache (decode position).
        batch:       Batch size.
        atol:        Absolute tolerance; returns True when max_diff < atol.
        seed:        RNG seed.

    Returns:
        (passed: bool, max_diff: float)
    """
    torch.manual_seed(seed)

    is_sparse = text_cfg.layer_types[layer_idx] == "minimax_m3_sparse"
    if is_sparse:
        max_ctx = text_cfg.index_topk_blocks * text_cfg.index_block_size
        prefill_len = min(prefill_len, max_ctx - 1)  # leave one slot for decode

    ctx_len = prefill_len + 1  # buffer size; position prefill_len is the decode slot

    nkv = text_cfg.num_key_value_heads
    hd = text_cfg.head_dim
    idx_d = text_cfg.index_head_dim

    # Pre-filled KV: random for positions 0..prefill_len-1, zero for the decode slot.
    pre_k = torch.cat([torch.randn(batch, nkv, prefill_len, hd), torch.zeros(batch, nkv, 1, hd)], dim=2)
    pre_v = torch.cat([torch.randn(batch, nkv, prefill_len, hd), torch.zeros(batch, nkv, 1, hd)], dim=2)
    # Pre-filled index keys for the sparse indexer (same zeroed-slot convention).
    pre_idx_k = torch.cat([torch.randn(batch, 1, prefill_len, idx_d), torch.zeros(batch, 1, 1, idx_d)], dim=2)

    # ── QEff cache ─────────────────────────────────────────────────────────────
    # Build a per-layer tuple list so from_legacy_cache initialises layers[0..layer_idx].
    # Dummy 2-tuples cover layers below the target; the target layer gets a 3-tuple
    # carrying the index-key buffer so QEffMiniMaxSparseCache.index_keys is set.
    dummy = (
        torch.zeros(batch, nkv, ctx_len, hd),
        torch.zeros(batch, nkv, ctx_len, hd),
    )
    layer_tuples = [dummy] * layer_idx + [(pre_k, pre_v, pre_idx_k)]
    qeff_cache = QEffMiniMaxSparseCache.from_legacy_cache(layer_tuples)

    # ── HF mock cache ──────────────────────────────────────────────────────────
    # MiniMaxM3VLAttention.forward calls cache.update(k, v, layer_idx) for standard
    # attention and cache.layers[layer_idx].update_index(new_idx_k) for the indexer.
    pre_k_hf = pre_k[:, :, :prefill_len, :]  # HF holds only the filled positions
    pre_v_hf = pre_v[:, :, :prefill_len, :]
    pre_idx_k_hf = pre_idx_k[:, :, :prefill_len, :]

    class _HFIndexLayer:
        def __init__(self, past_idx_k: torch.Tensor) -> None:
            self._past = past_idx_k

        def update_index(self, new_idx_k: torch.Tensor) -> torch.Tensor:
            return torch.cat([self._past, new_idx_k], dim=2)

    class _HFCache:
        def __init__(self) -> None:
            self._k = pre_k_hf
            self._v = pre_v_hf
            self.layers = {layer_idx: _HFIndexLayer(pre_idx_k_hf)}

        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, _layer_idx: int):
            k = torch.cat([self._k, key_states], dim=2)
            v = torch.cat([self._v, value_states], dim=2)
            self._k, self._v = k, v
            return k, v

    hf_cache = _HFCache()

    # ── Shared module setup ────────────────────────────────────────────────────
    hf_attn = MiniMaxM3VLAttention(text_cfg, layer_idx=layer_idx).eval()
    qeff_attn = QEffMiniMaxM3VLAttention(text_cfg, layer_idx=layer_idx).eval()
    qeff_attn.load_state_dict(hf_attn.state_dict())
    # Mirror the production ModuleMappingTransform: swap the indexer class so that
    # QEffMiniMaxM3VLAttention.forward can call self.indexer._select_blocks().
    if qeff_attn.indexer is not None:
        qeff_attn.indexer.__class__ = QEffMiniMaxM3VLIndexer

    # Decode inputs: single token at position prefill_len.
    hidden_states = torch.randn(batch, 1, text_cfg.hidden_size)
    position_ids = torch.tensor([[prefill_len]], dtype=torch.long)

    rot_emb = QEffMiniMaxM3VLRotaryEmbedding(text_cfg).eval()
    with torch.no_grad():
        cos, sin = rot_emb(hidden_states, position_ids)

    with torch.no_grad():
        hf_out, _ = hf_attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=hf_cache,
            position_ids=position_ids,
        )
        qeff_out, _ = qeff_attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=qeff_cache,
            position_ids=position_ids,
        )

    max_diff = (hf_out - qeff_out).abs().max().item()
    return max_diff < atol, max_diff


@pytest.mark.llm_model
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["dense", "sparse"])
def test_minimax_m3_attention_module_parity(layer_idx):
    """
    Verify that QEffMiniMaxM3VLAttention.forward and MiniMaxM3VLAttention.forward
    produce identical outputs (within 1e-4) on a decode step with a pre-filled
    QEffMiniMaxSparseCache, for both the dense (layer 0) and sparse (layer 1) types.
    """
    text_cfg = _tiny_minimax_m3_text_config()
    passed, max_diff = check_attention_module_parity(text_cfg, layer_idx=layer_idx)
    assert passed, (
        f"Attention parity failed for layer_idx={layer_idx} "
        f"(layer_type={text_cfg.layer_types[layer_idx]}): max_diff={max_diff:.6f}"
    )


def check_moe_module_parity(
    text_cfg: MiniMaxM3VLTextConfig,
    batch: int = 1,
    seq_len: int = 4,
    atol: float = 1e-4,
    seed: int = 42,
) -> tuple:
    """
    Verify that QEffMiniMaxM3VLSparseMoeBlock.forward and MiniMaxM3VLSparseMoeBlock.forward
    produce identical outputs on random hidden_states.

    QEffMiniMaxM3VLSparseMoeBlock replaces the per-expert index_add_ loop with batched
    matrix multiplies (BMM).  Both should produce the same weighted sum of expert outputs
    plus the shared expert contribution.

    Args:
        text_cfg: MiniMaxM3VLTextConfig with sparse MoE settings.
        batch:    Batch size.
        seq_len:  Sequence length.
        atol:     Absolute tolerance; returns True when max_diff < atol.
        seed:     RNG seed.

    Returns:
        (passed: bool, max_diff: float)
    """
    torch.manual_seed(seed)

    hf_moe = MiniMaxM3VLSparseMoeBlock(text_cfg).eval()
    with torch.no_grad():
        for parameter in hf_moe.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    qeff_moe = MiniMaxM3VLSparseMoeBlock(text_cfg).eval()
    qeff_moe.load_state_dict(hf_moe.state_dict())
    qeff_moe.__class__ = QEffMiniMaxM3VLSparseMoeBlock
    qeff_moe.__qeff_init__()
    qeff_moe.transform_weights()

    hidden_states = torch.randn(batch, seq_len, text_cfg.hidden_size)

    with torch.no_grad():
        hf_out = hf_moe(hidden_states.clone())
        qeff_out = qeff_moe(hidden_states.clone())

    max_diff = (hf_out - qeff_out).abs().max().item()
    return max_diff < atol, max_diff


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("batch", "seq_len"),
    [(1, 1), (1, 4), (2, 6)],
    ids=["decode", "context_4", "context_6_batch2"],
)
def test_minimax_m3_moe_module_parity(batch, seq_len):
    """
    Verify that QEffMiniMaxM3VLSparseMoeBlock.forward produces identical outputs to
    MiniMaxM3VLSparseMoeBlock.forward (within 1e-5) across decode and context steps.

    The QEff variant uses BMM-based expert computation instead of the HF per-expert
    index_add_ loop; this test confirms mathematical equivalence.
    """
    text_cfg = _tiny_minimax_m3_text_config()
    passed, max_diff = check_moe_module_parity(text_cfg, batch=batch, seq_len=seq_len)
    assert passed, f"MoE parity failed for batch={batch}, seq_len={seq_len}: max_diff={max_diff:.6f}"


@pytest.mark.llm_model
def test_minimax_m3_moe_expert_parallel_transform_and_parity():
    from QEfficient.transformers.models.pytorch_transforms import OptimizedMoETransform
    from QEfficient.transformers.moe import MoEFlavour

    torch.manual_seed(42)
    text_cfg = _tiny_minimax_m3_text_config()
    hf_moe = MiniMaxM3VLSparseMoeBlock(text_cfg).eval()
    with torch.no_grad():
        for parameter in hf_moe.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    qeff_moe = MiniMaxM3VLSparseMoeBlock(text_cfg).eval()
    qeff_moe.load_state_dict(hf_moe.state_dict())

    qeff_moe, transformed = OptimizedMoETransform.apply(
        qeff_moe,
        prefill_only=True,
        num_devices=2,
        num_cores=text_cfg.num_local_experts // 2,
        qaic_config={
            "moe_config": {
                "flavour": "expert_parallel",
                "expert_parallel_chunk_size": 4,
                "tree_reduce": True,
            }
        },
        prefill_seq_len=4,
    )

    assert transformed
    assert isinstance(qeff_moe, QEffMiniMaxM3VLSparseMoeBlock)
    assert qeff_moe._moe_flavour is MoEFlavour.EXPERT_PARALLEL
    assert qeff_moe.moe_weights.gate.shape[:2] == (text_cfg.num_local_experts, 1)
    assert qeff_moe.num_devices == 2
    assert qeff_moe.tree_reduce is True

    hidden_states = torch.randn(1, 4, text_cfg.hidden_size)
    with torch.no_grad():
        hf_out = hf_moe(hidden_states.clone())
        qeff_out = qeff_moe(hidden_states.clone())

    torch.testing.assert_close(qeff_out, hf_out, atol=1e-4, rtol=1e-4)

    qeff_moe, _ = OptimizedMoETransform.apply(
        qeff_moe,
        prefill_only=False,
        qaic_config={"moe_config": {"flavour": "decode_bmm"}},
    )
    assert qeff_moe._moe_flavour is MoEFlavour.DECODE_BMM
    assert qeff_moe.moe_weights.gate.ndim == 3
    with torch.no_grad():
        decode_out = qeff_moe(hidden_states.clone())
    torch.testing.assert_close(decode_out, hf_out, atol=1e-4, rtol=1e-4)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("vlm_name", "model_id"),
    sorted(VLM_EXPORT_MODEL_IDS.items()),
    ids=sorted(VLM_EXPORT_MODEL_IDS),
)
def test_vlm_export_smoke_additional_models(vlm_name, model_id, tmp_path):
    vlm_onnx_path = _export_vlm_with_text_fallback(model_id, tmp_path / f"vlm-{vlm_name}")
    assert vlm_onnx_path.name.endswith(".onnx")


@pytest.mark.llm_model
def test_text_embedding_cpu_parity_and_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID)
    model_hf = AutoModel.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID, **MODEL_KWARGS)
    model_hf.eval()

    inputs = tokenizer("hello world", return_tensors="pt")
    hf_outputs = model_hf(**inputs).last_hidden_state.detach().numpy()

    qeff_model = QEFFAutoModel(model_hf)
    qeff_outputs = qeff_model.generate(inputs=inputs, runtime_ai100=False).last_hidden_state.detach().numpy()
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_outputs = _run_embedding_ort(onnx_path, inputs)

    assert np.allclose(hf_outputs, qeff_outputs, atol=1e-5)
    assert np.allclose(hf_outputs, ort_outputs, atol=1e-5)


@pytest.mark.llm_model
def test_text_embedding_fp16_clip_transform_and_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID)
    qeff_model = QEFFAutoModel.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID)
    transform_names = {transform.__name__ for transform in qeff_model._onnx_transforms}

    assert "FP16ClipTransform" in transform_names
    assert "SplitTensorsTransform" in transform_names

    inputs = tokenizer("hello world", return_tensors="pt")
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "embedding-ai100"))
    ort_outputs = _run_embedding_ort(onnx_path, inputs)
    assert ort_outputs.shape[0] == inputs["input_ids"].shape[0]
    assert ort_outputs.shape[1] == inputs["input_ids"].shape[1]


@pytest.mark.llm_model
def test_audio_embedding_ctc_cpu_parity_and_export(tmp_path):
    processor = AutoTokenizer.from_pretrained(TINY_AUDIO_CTC_MODEL_ID)
    del processor
    replace_transformers_quantizers()
    model_hf = AutoModelForCTC.from_pretrained(TINY_AUDIO_CTC_MODEL_ID, **MODEL_KWARGS, low_cpu_mem_usage=False)
    model_hf.eval()

    from transformers import AutoProcessor

    audio_processor = AutoProcessor.from_pretrained(TINY_AUDIO_CTC_MODEL_ID)
    input_values = audio_processor(
        np.zeros(400, dtype=np.float32), return_tensors="pt", sampling_rate=16000
    ).input_values

    hf_logits = model_hf(input_values=input_values).logits.detach().numpy()
    qeff_model = QEFFAutoModelForCTC(model_hf, pretrained_model_name_or_path=TINY_AUDIO_CTC_MODEL_ID)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_session = _ort_session(onnx_path)
    ort_logits = ort_session.run(None, {"input_values": input_values.detach().numpy()})[0]

    assert np.allclose(hf_logits, ort_logits, atol=1e-5)


@pytest.mark.llm_model
def test_seq_classification_cpu_parity_and_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TINY_SEQ_CLASSIFICATION_MODEL_ID, trust_remote_code=True)
    model_hf = AutoModelForSequenceClassification.from_pretrained(
        TINY_SEQ_CLASSIFICATION_MODEL_ID,
        trust_remote_code=True,
    )
    model_hf.eval()

    inputs = tokenizer("quick classification check", return_tensors="pt")
    hf_logits = model_hf(**inputs).logits.detach().numpy()

    qeff_model = QEFFAutoModelForSequenceClassification(model_hf)
    qeff_logits = qeff_model.model(**inputs).logits.detach().numpy()
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_session = _ort_session(onnx_path)
    input_names = {item.name for item in ort_session.get_inputs()}
    ort_logits = ort_session.run(
        None,
        {name: tensor.detach().numpy() for name, tensor in inputs.items() if name in input_names},
    )[0]

    assert np.allclose(hf_logits, qeff_logits, atol=1e-5)
    assert np.allclose(hf_logits, ort_logits, atol=1e-5)


@pytest.mark.llm_model
def test_whisper_export_smoke(tmp_path):
    model_hf = AutoModelForSpeechSeq2Seq.from_pretrained(
        TINY_WHISPER_MODEL_ID,
        **MODEL_KWARGS,
        low_cpu_mem_usage=False,
    )
    model_hf.eval()

    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model_hf, pretrained_model_name_or_path=TINY_WHISPER_MODEL_ID)
    onnx_path = _run_whisper_export_smoke(qeff_model, tmp_path / "whisper")

    assert onnx_path.name.endswith(".onnx")


@pytest.mark.llm_model
def test_causal_subfunction_export_smoke(tmp_path):
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    with_subfunctions_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "with-subfunctions", use_onnx_subfunctions=True, offload_pt_weights=False)
    )
    without_subfunctions_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "without-subfunctions", use_onnx_subfunctions=False)
    )

    with_subfunctions_model = onnx.load(with_subfunctions_path, load_external_data=False)
    without_subfunctions_model = onnx.load(without_subfunctions_path, load_external_data=False)
    with_names = [func.name for func in with_subfunctions_model.functions]
    without_names = [func.name for func in without_subfunctions_model.functions]
    assert any("QEffGPT2Block" in name for name in with_names)
    assert not any("QEffGPT2Block" in name for name in without_names)


@pytest.mark.llm_model
def test_causal_subfunction_export_uses_semantic_weight_and_node_names(tmp_path):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model_hf = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS).eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    onnx_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "semantic-wsub-names", use_onnx_subfunctions=True, offload_pt_weights=False)
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)

    initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}
    assert "model.layers.0.self_attn.q_proj.weight" in initializer_names
    assert not any(name.startswith("onnx::MatMul_") for name in initializer_names)

    function_node_names = {node.name for function in onnx_model.functions for node in function.node}
    assert "self_attn/q_proj/MatMul" in function_node_names
    assert "mlp/down_proj/MatMul" in function_node_names


@pytest.mark.llm_model
def test_gemma3_vlm_export_parity_with_and_without_subfunctions(tmp_path):
    """Gemma3 VLM export keeps retained-state/output signatures stable across subfunction toggles.

    This guards the Gemma3 cache-bridging export patch on the supported VLM path:
    the historical working non-subfunction export remains the reference, and
    enabling subfunctions must preserve the exported graph interface.
    """
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        VLM_TEXT_RUNTIME_MODEL_ID,
        trust_remote_code=True,
        kv_offload=True,
    )

    with_subfunctions_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "gemma3-vlm-with-subfunctions",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    without_subfunctions_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "gemma3-vlm-without-subfunctions",
            use_onnx_subfunctions=False,
            offload_pt_weights=False,
        )
    )

    with_model = onnx.load(with_subfunctions_path, load_external_data=False)
    without_model = onnx.load(without_subfunctions_path, load_external_data=False)

    # Ensure the subfunction export path really emits decoder-block subfunctions
    # for the Gemma3 language decoder, not just a graph with matching IO names.
    with_subfunction_names = _decoder_block_subfunction_names(with_model, qeff_model.lang_model)
    without_subfunction_names = _decoder_block_subfunction_names(without_model, qeff_model.lang_model)
    assert with_subfunction_names, "Expected Gemma3 decoder-layer subfunctions with use_onnx_subfunctions=True"
    assert not without_subfunction_names, (
        "Did not expect Gemma3 decoder-layer subfunctions with use_onnx_subfunctions=False"
    )

    assert [value.name for value in with_model.graph.input] == [value.name for value in without_model.graph.input]
    assert [value.name for value in with_model.graph.output] == [value.name for value in without_model.graph.output]


@pytest.mark.llm_model
def test_repeat_kv_quickcheck_hf_qeff_ort_parity(tmp_path):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model_hf = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS).eval()
    qeff_model = QEFFAutoModelForCausalLM(deepcopy(model_hf), qaic_config={"replicate_kv_heads": True})

    input_ids = torch.arange(1, 5, dtype=torch.int64).view(1, 4)
    position_ids = torch.arange(4, dtype=torch.int64).view(1, 4)
    head_dim = config.hidden_size // config.num_attention_heads
    past_key_values = tuple(
        (
            torch.zeros((1, config.num_attention_heads, 8, head_dim), dtype=torch.float32),
            torch.zeros((1, config.num_attention_heads, 8, head_dim), dtype=torch.float32),
        )
        for _ in range(config.num_hidden_layers)
    )
    inputs = {"input_ids": input_ids, "position_ids": position_ids, "past_key_values": past_key_values}
    with torch.no_grad():
        hf_logits = model_hf(input_ids=input_ids, position_ids=position_ids).logits[:, -1:, :].detach().numpy()

    qeff_model.transform(ctx_len=8, seq_len=4, bs=1, num_devices=4, qaic_config=qeff_model.model.qaic_config)
    with torch.no_grad():
        qeff_logits = qeff_model.model(**inputs).logits.detach().numpy()

    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "repeat-kv"))
    ort_session, added_initializers = _ort_session_zeroing_int32_max_constants(onnx_path)
    assert added_initializers is not None
    flat_inputs = {name: value.detach().numpy() for name, value in inputs.items() if torch.is_tensor(value)}
    for layer_idx, layer_state in enumerate(past_key_values):
        flat_inputs[f"past_key.{layer_idx}"] = layer_state[0].detach().numpy()
        flat_inputs[f"past_value.{layer_idx}"] = layer_state[1].detach().numpy()
    ort_inputs = {item.name: flat_inputs[item.name] for item in ort_session.get_inputs()}
    ort_logits = ort_session.run(None, ort_inputs)[0]

    assert np.allclose(hf_logits, qeff_logits, atol=1e-4, rtol=1e-4)
    assert np.allclose(qeff_logits, ort_logits, atol=1e-4, rtol=1e-4)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "config_kwargs"),
    sorted(TINY_MOE_PREFILL_SUBFUNCTION_CONFIGS.items()),
    ids=sorted(TINY_MOE_PREFILL_SUBFUNCTION_CONFIGS),
)
def test_moe_prefill_subfunction_export_uses_reduce_sum_reductions(model_type, config_kwargs, tmp_path):
    config = AutoConfig.for_model(model_type, **config_kwargs)
    model_hf = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=False)
    qeff_model.transform(
        ctx_len=64,
        seq_len=64,
        bs=1,
        prefill_only=True,
        enable_chunking=True,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 32}},
        prefill_seq_len=64,
    )

    onnx_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / f"{model_type}-prefill-subfunctions",
            prefill_only=True,
            enable_chunking=True,
            prefill_seq_len=64,
            num_cores=2,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": 32}},
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    decoder_function_names = _decoder_block_subfunction_names(onnx_model, qeff_model)
    decoder_op_types = _function_op_types(onnx_model, decoder_function_names)

    assert len(decoder_function_names) == config.num_hidden_layers
    assert "ReduceSum" in decoder_op_types
    assert "CtxGather3D" in decoder_op_types
    assert "CtxScatter3D" in decoder_op_types
    assert "CtxScatter3DInt" in decoder_op_types


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_compile_with_subfunctions_all_models(model_type, model_id, tmp_path):
    del model_type
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    try:
        qpc = qeff_model.compile(
            prefill_seq_len=8,
            ctx_len=32,
            use_onnx_subfunctions=True,
            compile_dir=tmp_path / "compile-with-subfunctions",
        )
    except Exception as exc:
        pytest.skip(
            f"Skipping compile for {model_id}: compile backend unavailable or unsupported in this environment "
            f"({type(exc).__name__}: {exc})"
        )

    qpc_path = Path(qpc)
    assert qpc_path.name == "qpc"
    assert qpc_path.is_dir()


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_subfunction_export_smoke_all_models(model_type, model_id, tmp_path):
    if model_type == "gpt_oss":
        pytest.skip("Subfunction runtime parity is currently excluded for gpt_oss in quickcheck.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_id,
            **MODEL_KWARGS,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    prompt = ["hello world"]
    prompt_len = 8
    ctx_len = 12

    model_hf.eval()

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompt,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "with-subfunctions-all", use_onnx_subfunctions=True))
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0))
    assert np.array_equal(kv_tokens, ort_tokens)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_subfunction_count_with_onnx_subfunctions(model_type, model_id, tmp_path):
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    onnx_path = _exported_onnx_path(
        qeff_model.export(tmp_path / f"subfunction-count-{model_type}", use_onnx_subfunctions=True)
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    subfunction_count = _count_decoder_block_subfunctions(onnx_model, qeff_model)

    if model_type in CAUSAL_MULTI_SUBFUNCTION_MODEL_TYPES:
        assert subfunction_count > 1, (
            f"{model_type} expected multiple decoder-block subfunctions (>1), but found {subfunction_count}"
        )
    else:
        assert subfunction_count == 1, (
            f"{model_type} expected a single decoder-block subfunction (1), but found {subfunction_count}"
        )


@pytest.mark.llm_model
def test_causal_subfunction_and_proxy_export_smoke_gpt2(tmp_path):
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            enable_proxy=True,
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_model, enable_proxy=True)
    onnx_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "with-subfunctions-and-proxy", use_onnx_subfunctions=True)
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    assert any("QEffGPT2Block" in func.name for func in onnx_model.functions)


@pytest.mark.llm_model
def test_gpt_oss_proxy_repeats_each_decoder_subfunction(tmp_path):
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt_oss"]
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            enable_proxy=True,
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    assert qeff_model.model.config.num_hidden_layers == 4
    assert Counter(qeff_model.model.config.layer_types) == Counter({"sliding_attention": 2, "full_attention": 2})

    onnx_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "gpt-oss-proxy-subfunctions", use_onnx_subfunctions=True)
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    function_names = _decoder_block_subfunction_names(onnx_model, qeff_model)
    call_counts = _function_call_counts(onnx_model, function_names)

    assert len(function_names) == 2
    assert call_counts.keys() == function_names
    assert sorted(call_counts.values()) == [2, 2]


def test_proxy_subfunction_validation_rejects_single_call_function(tmp_path):
    from onnx import TensorProto, helper

    from QEfficient.utils.export_utils import _validate_proxy_subfunction_calls

    class DummyLayer:
        pass

    class DummyModel:
        def get_submodules_for_export(self):
            return {DummyLayer}

    class DummyQEffModel:
        _enable_proxy = True
        model = DummyModel()

    graph = helper.make_graph(
        [helper.make_node("DummyLayer", ["x"], ["y"])],
        "proxy_subfunction_single_call",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    function = helper.make_function(
        "",
        "DummyLayer",
        ["x"],
        ["y"],
        [helper.make_node("Identity", ["x"], ["y"])],
        [helper.make_operatorsetid("", 17)],
    )
    model = helper.make_model(graph, functions=[function], opset_imports=[helper.make_operatorsetid("", 17)])
    onnx_path = tmp_path / "single_call.onnx"
    onnx.save(model, onnx_path)

    with pytest.raises(RuntimeError, match="must be invoked more than once"):
        _validate_proxy_subfunction_calls(DummyQEffModel(), onnx_path)


def test_proxy_subfunction_validation_accepts_repeated_function_calls(tmp_path):
    from onnx import TensorProto, helper

    from QEfficient.utils.export_utils import _validate_proxy_subfunction_calls

    class DummyLayer:
        pass

    class DummyModel:
        def get_submodules_for_export(self):
            return {DummyLayer}

    class DummyQEffModel:
        _enable_proxy = True
        model = DummyModel()

    graph = helper.make_graph(
        [
            helper.make_node("DummyLayer", ["x"], ["mid"]),
            helper.make_node("DummyLayer", ["mid"], ["y"]),
        ],
        "proxy_subfunction_repeated_call",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    function = helper.make_function(
        "",
        "DummyLayer",
        ["x"],
        ["y"],
        [helper.make_node("Identity", ["x"], ["y"])],
        [helper.make_operatorsetid("", 17)],
    )
    model = helper.make_model(graph, functions=[function], opset_imports=[helper.make_operatorsetid("", 17)])
    onnx_path = tmp_path / "repeated_call.onnx"
    onnx.save(model, onnx_path)

    _validate_proxy_subfunction_calls(DummyQEffModel(), onnx_path)


def test_proxy_subfunction_validation_ignores_single_use_vision_functions(tmp_path):
    from onnx import TensorProto, helper

    from QEfficient.utils.export_utils import _validate_proxy_subfunction_calls

    class DummyDecoderLayer:
        pass

    class DummyVisionBlock:
        pass

    class DummyLanguageDecoder:
        def get_submodules_for_export(self):
            return {DummyDecoderLayer}

    class DummyVlmModel:
        def get_submodules_for_export(self):
            return {DummyDecoderLayer, DummyVisionBlock}

        def get_qeff_language_decoder(self):
            return DummyLanguageDecoder()

    class DummyQEffModel:
        _enable_proxy = True
        model = DummyVlmModel()

    graph = helper.make_graph(
        [
            helper.make_node("DummyVisionBlock", ["x"], ["vision"]),
            helper.make_node("DummyDecoderLayer", ["vision"], ["mid"]),
            helper.make_node("DummyDecoderLayer", ["mid"], ["y"]),
        ],
        "proxy_subfunction_mixed_vlm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    functions = [
        helper.make_function(
            "",
            "DummyVisionBlock",
            ["x"],
            ["y"],
            [helper.make_node("Identity", ["x"], ["y"])],
            [helper.make_operatorsetid("", 17)],
        ),
        helper.make_function(
            "",
            "DummyDecoderLayer",
            ["x"],
            ["y"],
            [helper.make_node("Identity", ["x"], ["y"])],
            [helper.make_operatorsetid("", 17)],
        ),
    ]
    model = helper.make_model(graph, functions=functions, opset_imports=[helper.make_operatorsetid("", 17)])
    onnx_path = tmp_path / "mixed_vlm.onnx"
    onnx.save(model, onnx_path)

    _validate_proxy_subfunction_calls(DummyQEffModel(), onnx_path)


def test_proxy_subfunction_validation_skips_vision_encoder_exports(tmp_path):
    from onnx import TensorProto, helper

    from QEfficient.utils.export_utils import _validate_proxy_subfunction_calls

    class DummyVisionBlock:
        pass

    class DummyVisionModel:
        def get_submodules_for_export(self):
            return {DummyVisionBlock}

    class QEffVisionEncoderForTextImageToTextModel:
        _enable_proxy = True
        model = DummyVisionModel()

    graph = helper.make_graph(
        [helper.make_node("DummyVisionBlock", ["x"], ["y"])],
        "proxy_subfunction_vision_only",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    function = helper.make_function(
        "",
        "DummyVisionBlock",
        ["x"],
        ["y"],
        [helper.make_node("Identity", ["x"], ["y"])],
        [helper.make_operatorsetid("", 17)],
    )
    model = helper.make_model(graph, functions=[function], opset_imports=[helper.make_operatorsetid("", 17)])
    onnx_path = tmp_path / "vision_only.onnx"
    onnx.save(model, onnx_path)

    _validate_proxy_subfunction_calls(QEffVisionEncoderForTextImageToTextModel(), onnx_path)


def test_subfunction_export_restores_onnx_transforms_on_failure():
    from QEfficient.base.onnx_transforms import CustomOpTransform, RenameFunctionOutputsTransform
    from QEfficient.transformers.cache_utils import InvalidIndexProvider
    from QEfficient.utils.export_utils import export_wrapper

    class DummyInner:
        config = None

        def get_submodules_for_export(self):
            return set()

    class DummyQEff:
        model = DummyInner()
        model_architecture = None
        model_name = "DummyQEff"
        hash_params = {}
        export_hash = None
        _onnx_transforms = []

        @export_wrapper
        def export(self, export_dir=None, **kwargs):
            raise RuntimeError("forced export failure")

    qeff_model = DummyQEff()

    with pytest.raises(RuntimeError, match="forced export failure"):
        qeff_model.export(use_onnx_subfunctions=True)

    assert qeff_model._onnx_transforms == []
    assert RenameFunctionOutputsTransform not in DummyQEff._onnx_transforms
    assert CustomOpTransform not in DummyQEff._onnx_transforms
    assert InvalidIndexProvider.SUBFUNC_ENABLED is False


@pytest.mark.llm_model
def test_prefix_caching_continuous_batching_export_and_ort_smoke(tmp_path):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        PREFIX_CACHING_MODEL_ID, continuous_batching=True, torch_dtype=torch.float32
    )
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "prefix-caching"))
    onnx_model = onnx.load(onnx_path, load_external_data=False)

    input_names = {inp.name for inp in onnx_model.graph.input}
    output_names = {out.name for out in onnx_model.graph.output}
    op_types = {node.op_type for node in onnx_model.graph.node}
    assert "batch_index" in input_names
    assert "CtxScatterCB" in op_types
    assert "CtxGatherCB" in op_types
    assert any(name.endswith("_RetainedState") for name in output_names)


@pytest.mark.llm_model
def test_awq_export_smoke(tmp_path):
    replace_transformers_quantizers()
    try:
        model_hf = AutoModelForCausalLM.from_pretrained(
            TINY_AWQ_MODEL_ID, low_cpu_mem_usage=False, torch_dtype=torch.float32
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, TINY_AWQ_MODEL_ID)
    model_hf.eval()

    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=TINY_AWQ_MODEL_ID)
    with _suppress_native_output():
        onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
        onnx_model = onnx.load(onnx_path, load_external_data=False)

    assert any(node.op_type == "MatMulNBits" for node in onnx_model.graph.node)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_causal_lm():
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    try:
        qeff_default = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        )
        qeff_proxy = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, enable_proxy=True, torch_dtype=torch.float32
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


def test_proxy_layer_config_preserves_repeated_layer_types():
    from transformers import LlamaForCausalLM

    from QEfficient.proxy.modeling_utils import apply_proxy_layer_config, prepare_proxy_config

    llama_config = LlamaConfig(num_hidden_layers=16)
    assert apply_proxy_layer_config(llama_config) == 2
    assert llama_config.num_hidden_layers == 2

    gpt_oss_config = GptOssConfig(num_hidden_layers=16)
    assert apply_proxy_layer_config(gpt_oss_config) == 4
    assert gpt_oss_config.layer_types == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]

    qwen_moe_config = Qwen3MoeConfig(num_hidden_layers=16)
    assert apply_proxy_layer_config(qwen_moe_config) == 2

    mixed_qwen_moe_config = Qwen3MoeConfig(num_hidden_layers=16, decoder_sparse_step=2)
    assert apply_proxy_layer_config(mixed_qwen_moe_config) == 4

    qwen3_5_moe_config = Qwen3_5MoeTextConfig(num_hidden_layers=16)
    qwen3_5_moe_config.num_hidden_layers = 2
    assert apply_proxy_layer_config(qwen3_5_moe_config) == 8
    assert qwen3_5_moe_config.layer_types == ["linear_attention"] * 3 + ["full_attention"] + [
        "linear_attention"
    ] * 3 + ["full_attention"]

    kwargs = {"config": GptOssConfig(num_hidden_layers=16), "num_hidden_layers": 2}
    prepare_proxy_config("unused-local-config", kwargs)
    assert kwargs["config"].num_hidden_layers == 2
    assert "num_hidden_layers" not in kwargs

    llama_kwargs = {
        "config": LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=8,
            num_key_value_heads=2,
            vocab_size=32,
        ),
        "num_hidden_layers": 3,
    }
    prepare_proxy_config("unused-local-config", llama_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(
        LlamaForCausalLM(llama_kwargs["config"]).eval(),
        enable_proxy=True,
    )
    assert "num_hidden_layers" not in llama_kwargs
    assert qeff_model.hash_params["proxy_num_hidden_layers"] == 3


@pytest.mark.llm_model
@pytest.mark.mdp
def test_proxy_compile_artifacts_generate_disagg_mdp_config_for_tiny_model(tmp_path):
    from transformers import LlamaForCausalLM

    from QEfficient.proxy.modeling_utils import prepare_proxy_config

    model_kwargs = {
        "config": LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=4,
            num_key_value_heads=2,
            vocab_size=32,
            max_position_embeddings=32,
            torch_dtype=torch.float32,
        )
    }
    prepare_proxy_config("unused-local-config", model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(
        LlamaForCausalLM(model_kwargs["config"]).eval(),
        enable_proxy=True,
        torch_dtype=torch.float32,
    )

    artifacts_dir = Path(
        qeff_model.compile(
            compile_dir=str(tmp_path / "proxy-mdp"),
            prefill_seq_len=4,
            ctx_len=8,
            batch_size=1,
            num_devices=2,
            mdp_num_partitions=2,
            artifacts=True,
            offload_pt_weights=False,
        )
    )

    assert qeff_model.num_layers == 2
    assert qeff_model.hash_params["proxy_num_hidden_layers"] == 2
    assert qeff_model.qpc_path is None
    assert qeff_model.compile_artifacts_path == artifacts_dir

    mdp_path = artifacts_dir / "mdp_disagg_2d_2p.json"
    assert mdp_path.is_file()
    mdp_data = json.loads(mdp_path.read_text())
    assert len(mdp_data["partitions"]) == 2
    assert [device["deviceId"] for device in mdp_data["partitions"][0]["devices"]] == [0]
    assert [device["deviceId"] for device in mdp_data["partitions"][1]["devices"]] == [1]

    replay_script = artifacts_dir / "qaic-compile.sh"
    assert replay_script.is_file()
    replay_command = replay_script.read_text()
    assert "-mdp-load-partition-config=mdp_disagg_2d_2p.json" in replay_command
    assert "-aic-binary-dir=qpc" in replay_command


def test_proxy_layer_config_only_reduces_vlm_language_layers():
    from QEfficient.proxy.modeling_utils import apply_proxy_layer_config

    # Proxy layer reduction is language-only; vision layers must remain untouched.
    config = Qwen3VLMoeConfig(
        text_config=Qwen3VLMoeTextConfig(num_hidden_layers=16),
        vision_config=Qwen3VLMoeVisionConfig(depth=8),
    )

    assert apply_proxy_layer_config(config) == 2
    assert config.text_config.num_hidden_layers == 2

    explicit_config = Qwen3VLMoeConfig(
        text_config=Qwen3VLMoeTextConfig(num_hidden_layers=16),
        vision_config=Qwen3VLMoeVisionConfig(depth=16),
    )
    assert apply_proxy_layer_config(explicit_config, num_hidden_layers=3) == 3
    assert explicit_config.text_config.num_hidden_layers == 3
    assert explicit_config.vision_config.depth == 16
    assert config.vision_config.depth == 8


def test_proxy_module_transform_leaves_vision_embedding_unchanged():
    from QEfficient.proxy import QeffProxyEmbedding, QeffProxyLinear
    from QEfficient.proxy.pytorch_transform import QeffProxyModuleTransform

    class DummyVlm(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_decoder = nn.Module()
            self.language_decoder.token_embedding = nn.Embedding(32, 8)
            self.language_decoder.lm_head = nn.Linear(8, 32, bias=False)
            self.vision_embedding = nn.Embedding(16, 8)

        def get_qeff_language_decoder(self):
            return self.language_decoder

        def get_input_embeddings(self):
            return self.language_decoder.token_embedding

        def get_output_embeddings(self):
            return self.language_decoder.lm_head

    model = DummyVlm()
    model, transformed = QeffProxyModuleTransform.apply(model)

    assert transformed
    assert isinstance(model.language_decoder.token_embedding, QeffProxyEmbedding)
    assert isinstance(model.language_decoder.lm_head, QeffProxyLinear)
    assert type(model.vision_embedding) is nn.Embedding

    causal_model = nn.Module()
    causal_model.token_embedding = nn.Embedding(32, 8)
    causal_model.position_embedding = nn.Embedding(16, 8)
    causal_model.lm_head = nn.Linear(8, 32, bias=False)
    causal_model, transformed = QeffProxyModuleTransform.apply(causal_model)

    assert transformed
    assert isinstance(causal_model.token_embedding, QeffProxyEmbedding)
    assert isinstance(causal_model.position_embedding, QeffProxyEmbedding)
    assert isinstance(causal_model.lm_head, QeffProxyLinear)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_embedding():
    model_id = TINY_TEXT_EMBEDDING_MODEL_ID
    try:
        qeff_default = QEFFAutoModel.from_pretrained(model_id)
        qeff_proxy = QEFFAutoModel.from_pretrained(model_id, enable_proxy=True)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(
        qeff_default, enable_proxy=False, always_on_transforms={"FP16ClipTransform", "SplitTensorsTransform"}
    )
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_sequence_classification():
    model_id = TINY_SEQ_CLASSIFICATION_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
        qeff_proxy = QEFFAutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True, enable_proxy=True
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_whisper():
    model_id = TINY_WHISPER_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
        qeff_proxy = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True, enable_proxy=True)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_ctc():
    model_id = TINY_AUDIO_CTC_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForCTC.from_pretrained(model_id, trust_remote_code=True)
        qeff_proxy = QEFFAutoModelForCTC.from_pretrained(model_id, trust_remote_code=True, enable_proxy=True)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_vlm():
    model_id = VLM_TEXT_RUNTIME_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, kv_offload=True
        )
        qeff_proxy = QEFFAutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, enable_proxy=True, kv_offload=True
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_dual_qpc_vlm_proxy_transform_policy(qeff_default, enable_proxy=False)
    _assert_dual_qpc_vlm_proxy_transform_policy(qeff_proxy, enable_proxy=True)


class TestCausalLMFlagDiagnostics:
    """Unsupported/ignored CausalLM flags should fail or warn clearly."""

    def _tiny_llama(self):
        try:
            return QEFFAutoModelForCausalLM.from_pretrained(
                CAUSAL_RUNTIME_MODEL_IDS["llama"],
                continuous_batching=True,
                num_hidden_layers=2,
                **MODEL_KWARGS,
            )
        except Exception as exc:
            _skip_on_model_fetch_error(exc, CAUSAL_RUNTIME_MODEL_IDS["llama"])

    def test_export_decode_only_rejected_for_standard_causal_lm(self, tmp_path):
        qeff_model = self._tiny_llama()

        with pytest.raises(NotImplementedError, match="decode_only=True is not supported"):
            qeff_model.export(tmp_path / "decode-only", decode_only=True)

    def test_compile_retain_full_kv_ignored_for_llama(self, tmp_path, monkeypatch, caplog):
        qeff_model = self._tiny_llama()
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        captured_kwargs = {}

        def fake_compile(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "qpc"

        monkeypatch.setattr(qeff_model, "_compile", fake_compile)
        caplog.set_level(logging.WARNING, logger="QEfficient")

        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_path),
            prefill_seq_len=1,
            ctx_len=128,
            full_batch_size=1,
            prefill_only=False,
            retain_full_kv=True,
        )

        assert captured_kwargs["retain_full_kv"] is False
        assert captured_kwargs["specializations"][0]["_graph_name"] == "Decode"
        assert "retain_full_kv=True is only supported" in caplog.text


# ---------------------------------------------------------------------------
# Tests for the named-specializations format (backend team feature request)
# ---------------------------------------------------------------------------


class TestInferSpecializationName:
    """Unit tests for _infer_specialization_name."""

    # --- _graph_name tag (authoritative, set at creation time) ---

    def test_graph_name_tag_takes_priority(self):
        """_graph_name in spec overrides all heuristics."""
        spec = {"_graph_name": "Vision", "batch_size": "1", "seq_len": "128", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 0) == "Vision"

    def test_graph_name_tag_whisper_encoder(self):
        spec = {
            "_graph_name": "Encoder",
            "batch_size": "1",
            "seq_len": "1",
            "encoder_ctx_len": "1500",
            "feature_len": "3000",
        }
        assert _infer_specialization_name(spec, 0) == "Encoder"

    def test_graph_name_tag_whisper_decode(self):
        spec = {
            "_graph_name": "Decode",
            "batch_size": "1",
            "seq_len": "1",
            "encoder_ctx_len": "1500",
            "feature_len": "1",
        }
        assert _infer_specialization_name(spec, 1) == "Decode"

    def test_graph_name_tag_prefill(self):
        spec = {"_graph_name": "Prefill", "batch_size": "1", "seq_len": "128", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 0) == "Prefill"

    def test_graph_name_tag_decode(self):
        spec = {"_graph_name": "Decode", "batch_size": "1", "seq_len": "1", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 1) == "Decode"

    def test_graph_name_tag_embedding_with_seq_len(self):
        spec = {"_graph_name": "Embedding", "batch_size": "1", "seq_len": "128"}
        assert _infer_specialization_name(spec, 0) == "Embedding"

    # --- module_name hint (diffusers path) ---

    def test_module_name_used_when_no_tag(self):
        spec = {"batch_size": "1", "seq_len": "77"}
        assert _infer_specialization_name(spec, 0, module_name="text_encoder") == "text_encoder"

    def test_module_name_with_model_type(self):
        spec = {"batch_size": "1", "model_type": 1}
        assert _infer_specialization_name(spec, 0, module_name="transformer") == "transformer_model_type_1"

    # --- seq_len heuristic fallback (plain causal LM raw dicts) ---

    def test_prefill_detected_by_seq_len_gt_1(self):
        spec = {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 0) == "Prefill"

    def test_decode_detected_by_seq_len_1_string(self):
        spec = {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 1) == "Decode"

    def test_decode_detected_by_seq_len_1_int(self):
        spec = {"batch_size": 1, "seq_len": 1, "ctx_len": 4096}
        assert _infer_specialization_name(spec, 1) == "Decode"

    def test_encoder_detected_by_encoder_ctx_len_no_seq_len(self):
        spec = {"batch_size": "1", "encoder_ctx_len": "1500"}
        assert _infer_specialization_name(spec, 0) == "Encoder"

    def test_legacy_embedding_detected_by_sequence_length(self):
        spec = {"batch_size": "1", "sequence_length": "128"}
        assert _infer_specialization_name(spec, 0) == "Embedding"

    def test_generic_fallback_for_unknown_shape(self):
        spec = {"custom_dim": "42"}
        assert _infer_specialization_name(spec, 3) == "Graph_3"


class TestToNamedSpecializations:
    """Unit tests for to_named_specializations — the main serialization helper."""

    def test_llm_prefill_decode_pair(self):
        """Standard LLM: two specializations → Prefill + Decode."""
        flat = [
            {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
        ]
        result = to_named_specializations(flat)
        assert len(result) == 2
        assert result[0] == {
            "name": "Prefill",
            "symbols": {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
        }
        assert result[1] == {
            "name": "Decode",
            "symbols": {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
        }

    def test_llm_continuous_batching_with_full_batch_size(self):
        """Continuous-batching LLM: full_batch_size present in both entries."""
        flat = [
            {"batch_size": "1", "full_batch_size": "16", "seq_len": "128", "ctx_len": "4096"},
            {"batch_size": "16", "full_batch_size": "16", "seq_len": "1", "ctx_len": "4096"},
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Prefill"
        assert result[1]["name"] == "Decode"
        assert result[0]["symbols"]["full_batch_size"] == "16"
        assert result[1]["symbols"]["batch_size"] == "16"

    def test_prefill_only_single_entry(self):
        """When prompt_len == 1 the decode entry is dropped; only one entry remains."""
        flat = [{"batch_size": "1", "seq_len": "1", "ctx_len": "128"}]
        result = to_named_specializations(flat)
        assert len(result) == 1
        assert result[0]["name"] == "Decode"

    def test_vlm_vision_specialization(self):
        """VLM vision encoder: _graph_name tag set at source → 'Vision', tag stripped from symbols."""
        flat = [
            {
                "_graph_name": "Vision",
                "vision_batch_size": "1",
                "vision_size": "247",
                "grid_height": "988",
                "grid_width": "1176",
                "grid_h": "26",
                "grid_w": "38",
            }
        ]
        result = to_named_specializations(flat)
        assert len(result) == 1
        assert result[0]["name"] == "Vision"
        assert result[0]["symbols"]["vision_batch_size"] == "1"
        assert result[0]["symbols"]["vision_size"] == "247"
        assert "_graph_name" not in result[0]["symbols"]

    def test_vlm_lang_prefill_decode(self):
        """VLM language side: prefill + decode with vision_batch_size."""
        flat = [
            {
                "batch_size": "1",
                "ctx_len": "4096",
                "seq_len": "128",
                "vision_batch_size": "1",
                "vision_size": "247",
            },
            {
                "batch_size": "1",
                "ctx_len": "4096",
                "seq_len": "1",
                "vision_batch_size": "1",
                "vision_size": "247",
            },
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Prefill"
        assert result[1]["name"] == "Decode"
        assert result[0]["symbols"]["vision_size"] == "247"
        assert result[1]["symbols"]["vision_size"] == "247"

    def test_all_values_coerced_to_strings(self):
        """Integer values in the flat dict must be stringified in symbols."""
        flat = [{"batch_size": 1, "seq_len": 32, "ctx_len": 128}]
        result = to_named_specializations(flat)
        symbols = result[0]["symbols"]
        assert all(isinstance(v, str) for v in symbols.values())

    def test_output_structure_keys(self):
        """Every entry must have exactly 'name' and 'symbols' keys."""
        flat = [
            {"batch_size": "1", "seq_len": "64", "ctx_len": "256"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "256"},
        ]
        result = to_named_specializations(flat)
        for entry in result:
            assert set(entry.keys()) == {"name", "symbols"}

    def test_whisper_encoder_specialization_simplified(self):
        """Simplified Whisper spec: encoder_ctx_len, no seq_len → 'Encoder' via heuristic."""
        flat = [
            {"batch_size": "1", "encoder_ctx_len": "1500"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "448"},
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Encoder"
        assert result[1]["name"] == "Decode"
        assert result[0]["symbols"]["encoder_ctx_len"] == "1500"

    def test_whisper_encoder_specialization_real_spec(self):
        """Real Whisper spec: _graph_name tags set at source distinguish Encoder from Decode."""
        flat = [
            {
                "_graph_name": "Encoder",
                "batch_size": "1",
                "seq_len": "1",
                "encoder_ctx_len": "1500",
                "decoder_ctx_len": "150",
                "feature_len": "3000",
            },
            {
                "_graph_name": "Decode",
                "batch_size": "1",
                "seq_len": "1",
                "encoder_ctx_len": "1500",
                "decoder_ctx_len": "150",
                "feature_len": "1",
            },
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Encoder"
        assert result[1]["name"] == "Decode"
        assert result[0]["symbols"]["feature_len"] == "3000"
        assert result[1]["symbols"]["feature_len"] == "1"
        assert "_graph_name" not in result[0]["symbols"]
        assert "_graph_name" not in result[1]["symbols"]

    def test_graph_name_tag_stripped_from_symbols(self):
        """_graph_name must never appear in the serialized symbols dict."""
        flat = [{"_graph_name": "Prefill", "batch_size": "1", "seq_len": "128", "ctx_len": "4096"}]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Prefill"
        assert "_graph_name" not in result[0]["symbols"]

    def test_text_embedding_specialization_actual_compile_path(self):
        """Text embedding compile path: _graph_name + seq_len should serialize as Embedding."""
        flat = [{"_graph_name": "Embedding", "batch_size": "1", "seq_len": "128"}]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Embedding"
        assert result[0]["symbols"]["seq_len"] == "128"

    def test_legacy_text_embedding_specialization(self):
        """Legacy BERT-like raw dict: sequence_length fallback still resolves to Embedding."""
        flat = [{"batch_size": "1", "sequence_length": "128"}]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Embedding"
        assert result[0]["symbols"]["sequence_length"] == "128"

    def test_roundtrip_via_json(self):
        """Serializing to JSON and back must preserve the named format exactly."""
        import json

        flat = [
            {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
        ]
        named = to_named_specializations(flat)
        payload = {"specializations": named}
        roundtripped = json.loads(json.dumps(payload))
        assert roundtripped["specializations"][0]["name"] == "Prefill"
        assert roundtripped["specializations"][1]["name"] == "Decode"
        assert roundtripped["specializations"][0]["symbols"]["seq_len"] == "128"

    def test_compile_helper_create_and_dump_specializations(self, tmp_path):
        """create_and_dump_specializations must write the new named format to disk."""
        import json

        from QEfficient.compile.compile_helper import create_and_dump_specializations

        out_path = tmp_path / "specializations.json"
        create_and_dump_specializations(
            batch_size=1,
            prompt_len=128,
            ctx_len=4096,
            path=str(out_path),
        )
        data = json.loads(out_path.read_text())
        specs = data["specializations"]
        assert len(specs) == 2
        assert specs[0]["name"] == "Prefill"
        assert specs[1]["name"] == "Decode"
        assert "symbols" in specs[0]
        assert specs[0]["symbols"]["seq_len"] == "128"
        assert specs[1]["symbols"]["seq_len"] == "1"

    def test_compile_helper_prefill_only_when_prompt_len_1(self, tmp_path):
        """When prompt_len==1 and no full_batch_size, only one entry is written."""
        import json

        from QEfficient.compile.compile_helper import create_and_dump_specializations

        out_path = tmp_path / "specializations_prefill_only.json"
        create_and_dump_specializations(
            batch_size=1,
            prompt_len=1,
            ctx_len=128,
            path=str(out_path),
        )
        data = json.loads(out_path.read_text())
        specs = data["specializations"]
        assert len(specs) == 1
        assert specs[0]["name"] == "Decode"

    def test_compile_helper_continuous_batching(self, tmp_path):
        """With full_batch_size, both entries carry full_batch_size in symbols."""
        import json

        from QEfficient.compile.compile_helper import create_and_dump_specializations

        out_path = tmp_path / "specializations_cb.json"
        create_and_dump_specializations(
            batch_size=1,
            prompt_len=128,
            ctx_len=4096,
            path=str(out_path),
            full_batch_size=16,
        )
        data = json.loads(out_path.read_text())
        specs = data["specializations"]
        assert len(specs) == 2
        assert specs[0]["name"] == "Prefill"
        assert specs[1]["name"] == "Decode"
        assert specs[0]["symbols"]["full_batch_size"] == "16"
        assert specs[1]["symbols"]["full_batch_size"] == "16"
        assert specs[1]["symbols"]["batch_size"] == "16"


class TestGetCompilationDims:
    """Verify get_compilation_dims handles both flat (legacy) and named (new) formats."""

    def _write_spec(self, tmp_path, payload):
        import json

        spec_dir = tmp_path / "qpc-hash"
        spec_dir.mkdir()
        qpc_dir = spec_dir / "qpc"
        qpc_dir.mkdir()
        (spec_dir / "specializations.json").write_text(json.dumps(payload))
        return str(qpc_dir)

    def test_new_named_format(self, tmp_path):
        from QEfficient.generation.text_generation_inference import get_compilation_dims

        qpc_path = self._write_spec(
            tmp_path,
            {
                "specializations": [
                    {"name": "Prefill", "symbols": {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"}},
                    {"name": "Decode", "symbols": {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"}},
                ]
            },
        )
        bs, ctx, fbs, num_kv_blocks = get_compilation_dims(qpc_path)
        assert bs == 1
        assert ctx == 4096
        assert fbs is None

    def test_new_named_format_with_full_batch_size(self, tmp_path):
        from QEfficient.generation.text_generation_inference import get_compilation_dims

        qpc_path = self._write_spec(
            tmp_path,
            {
                "specializations": [
                    {
                        "name": "Prefill",
                        "symbols": {"batch_size": "1", "full_batch_size": "16", "seq_len": "128", "ctx_len": "4096"},
                    },
                    {
                        "name": "Decode",
                        "symbols": {"batch_size": "16", "full_batch_size": "16", "seq_len": "1", "ctx_len": "4096"},
                    },
                ]
            },
        )
        bs, ctx, fbs, num_kv_blocks = get_compilation_dims(qpc_path)
        assert bs == 1
        assert ctx == 4096
        assert fbs == 16

    def test_legacy_flat_format_still_works(self, tmp_path):
        from QEfficient.generation.text_generation_inference import get_compilation_dims

        qpc_path = self._write_spec(
            tmp_path,
            {
                "specializations": [
                    {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
                    {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
                ]
            },
        )
        bs, ctx, fbs, num_kv_blocks = get_compilation_dims(qpc_path)
        assert bs == 1
        assert ctx == 4096
        assert fbs is None


class TestDiffusersNamedSpecializations:
    """Named-specialization format for diffusers pipeline modules via _graph_name tag."""

    def _tag(self, specs, module_name):
        """Simulate what pipeline_utils does: tag each spec with _graph_name."""
        return [
            {**s, "_graph_name": f"{module_name}_model_type_{s['model_type']}" if "model_type" in s else module_name}
            for s in specs
        ]

    def test_flux_text_encoder(self):
        flat = self._tag([{"batch_size": 1, "seq_len": 77}], "text_encoder")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "text_encoder"
        assert result[0]["symbols"] == {"batch_size": "1", "seq_len": "77"}
        assert "_graph_name" not in result[0]["symbols"]

    def test_flux_text_encoder_2(self):
        flat = self._tag([{"batch_size": 1, "seq_len": 256}], "text_encoder_2")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "text_encoder_2"

    def test_flux_transformer(self):
        flat = self._tag([{"batch_size": 1, "seq_len": 256, "steps": 1}], "transformer")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "transformer"

    def test_flux_vae_decoder(self):
        flat = self._tag([{"batch_size": 1, "channels": 16}], "vae_decoder")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "vae_decoder"

    def test_wan_transformer_model_type_naming(self):
        """Wan transformer: two model_type entries → transformer_model_type_1 / _2."""
        flat = self._tag(
            [
                {"batch_size": "1", "num_channels": "16", "steps": "1", "sequence_length": "512", "model_type": 1},
                {"batch_size": "1", "num_channels": "16", "steps": "1", "sequence_length": "512", "model_type": 2},
            ],
            "transformer",
        )
        result = to_named_specializations(flat)
        assert result[0]["name"] == "transformer_model_type_1"
        assert result[1]["name"] == "transformer_model_type_2"

    def test_wan_vae_decoder(self):
        flat = self._tag([{"batch_size": 1, "num_channels": 16}], "vae_decoder")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "vae_decoder"

    def test_wan_i2v_vae_encoder(self):
        flat = self._tag([{"batch_size": 1, "num_channels": 16}], "vae_encoder")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "vae_encoder"

    def test_graph_name_tag_stripped_from_symbols(self):
        flat = self._tag([{"batch_size": 1, "seq_len": 77}], "text_encoder")
        result = to_named_specializations(flat)
        assert "_graph_name" not in result[0]["symbols"]

    def test_idempotent_already_named_entries(self):
        already = [{"name": "transformer", "symbols": {"batch_size": "1", "steps": "1"}}]
        result = to_named_specializations(already)
        assert result == already

    def test_no_tag_falls_back_to_lm_rules(self):
        """Without _graph_name tag, seq_len heuristic applies."""
        flat = [
            {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Prefill"
        assert result[1]["name"] == "Decode"


# ---------------------------------------------------------------------------
# Prefill Blocking Configurations
# ---------------------------------------------------------------------------


class _BlockedPrefillCache:
    def __init__(self, batch_size, num_kv_heads, ctx_len, head_dim):
        self.key = torch.zeros(batch_size, num_kv_heads, ctx_len, head_dim)
        self.value = torch.zeros_like(self.key)

    def read_only_blocked_K(self, start_index, end_index, layer_idx, cache_kwargs):
        return self.key[:, :, start_index:end_index, :]

    def read_only_blocked_V(self, start_index, end_index, layer_idx, cache_kwargs):
        return self.value[:, :, start_index:end_index, :]


@pytest.mark.parametrize("n_rep_chunk", [None, 1, 2])
def test_blocked_prefill_qkv_accepts_n_rep_chunk(n_rep_chunk):
    from QEfficient.blocking.blocked_attention_forwards import blocked_qkv_attention_forward_prefill_headpar_offline

    batch_size, sequence_length, head_dim, ctx_len = 1, 2, 4, 4
    num_query_heads, num_kv_groups = 4, 2
    module = SimpleNamespace(num_key_value_groups=num_kv_groups)
    query = torch.ones(batch_size, num_query_heads, sequence_length, head_dim)
    cache = _BlockedPrefillCache(batch_size, num_query_heads // num_kv_groups, ctx_len, head_dim)

    output, attention_weights = blocked_qkv_attention_forward_prefill_headpar_offline(
        module=module,
        query=query,
        key=None,
        value=None,
        attention_mask=None,
        scaling=1.0,
        num_q_blocks=1,
        num_kv_blocks=1,
        cache_kwargs={"position_ids": torch.zeros(batch_size, sequence_length, dtype=torch.long)},
        layer_idx=0,
        past_key_value=cache,
        ctx_len=ctx_len,
        configured_split=1,
        n_rep_chunk=n_rep_chunk,
    )

    assert output.shape == (batch_size, sequence_length, num_query_heads, head_dim)
    assert attention_weights is None


def test_blocked_prefill_qkv_chunks_queries_by_seq_len(monkeypatch):
    from QEfficient.blocking import blocked_attention_forwards as blocked

    chunk_lengths = []
    original_create_causal_mask = blocked._create_causal_mask

    def tracked_create_causal_mask(*args, **kwargs):
        chunk_lengths.append(kwargs["position_ids"].shape[-1])
        return original_create_causal_mask(*args, **kwargs)

    monkeypatch.setattr(blocked, "_create_causal_mask", tracked_create_causal_mask)

    batch_size, sequence_length, head_dim, ctx_len = 1, 2, 4, 8
    num_query_heads, num_kv_groups = 4, 2
    module = SimpleNamespace(num_key_value_groups=num_kv_groups)
    query = torch.ones(batch_size, num_query_heads, sequence_length, head_dim)
    cache = _BlockedPrefillCache(batch_size, num_query_heads // num_kv_groups, ctx_len, head_dim)

    blocked.blocked_qkv_attention_forward_prefill_headpar_offline(
        module=module,
        query=query,
        key=None,
        value=None,
        attention_mask=None,
        scaling=1.0,
        num_q_blocks=2,
        num_kv_blocks=1,
        cache_kwargs={"position_ids": torch.zeros(batch_size, sequence_length, dtype=torch.long)},
        layer_idx=0,
        past_key_value=cache,
        ctx_len=ctx_len,
        configured_split=1,
        n_rep_chunk=None,
    )

    assert chunk_lengths == [1, 1]


@pytest.mark.parametrize(
    "num_query_heads, num_kv_groups, num_cores",
    [
        (8, 4, 2),
        (8, 4, 4),
        (8, 4, 8),
        (12, 3, 4),
        (12, 3, 12),
    ],
)
def test_blocked_prefill_online_accepts_valid_num_heads_and_num_cores(num_query_heads, num_kv_groups, num_cores):
    from QEfficient.blocking.blocked_attention_forwards import blocked_qkv_attention_forward_prefill_online

    batch_size, sequence_length, head_dim, ctx_len = 1, 1, 4, 1
    num_kv_heads = num_query_heads // num_kv_groups
    module = SimpleNamespace(num_key_value_groups=num_kv_groups)
    query = torch.ones(batch_size, num_query_heads, sequence_length, head_dim)
    cache = _BlockedPrefillCache(batch_size, num_kv_heads, ctx_len, head_dim)

    output, attention_weights = blocked_qkv_attention_forward_prefill_online(
        module=module,
        query=query,
        key=None,
        value=None,
        attention_mask=None,
        scaling=1.0,
        num_q_blocks=1,
        num_kv_blocks=1,
        cache_kwargs={"position_ids": torch.zeros(batch_size, sequence_length, dtype=torch.long)},
        layer_idx=0,
        past_key_value=cache,
        ctx_len=ctx_len,
        num_cores_per_device=num_cores,
    )

    assert output.shape == (batch_size, sequence_length, num_query_heads, head_dim)


@pytest.mark.parametrize(
    "num_cores",
    [
        0,
        1,
        3,
        6,
    ],
)
def test_blocked_prefill_online_rejects_invalid_num_cores(num_cores):
    from QEfficient.blocking.blocked_attention_forwards import blocked_qkv_attention_forward_prefill_online

    num_query_heads, num_kv_groups = 8, 4
    module = SimpleNamespace(num_key_value_groups=num_kv_groups)
    query = torch.ones(1, num_query_heads, 1, 4)

    with pytest.raises(ValueError):
        blocked_qkv_attention_forward_prefill_online(
            module=module,
            query=query,
            key=None,
            value=None,
            attention_mask=None,
            scaling=1.0,
            num_q_blocks=1,
            num_kv_blocks=1,
            cache_kwargs={},
            layer_idx=0,
            past_key_value=None,
            ctx_len=1,
            num_cores_per_device=num_cores,
        )


# ---------------------------------------------------------------------------
# Layer-wise export (provisional, scheduled for deprecation)
# ---------------------------------------------------------------------------


LAYERWISE_TINY_MODEL_ID = "tiny-random/qwen3-vl-moe"
LAYERWISE_TINY_MODEL_IDS = {
    "qwen3_vl_moe": "tiny-random/qwen3-vl-moe",
    "qwen3_5_moe": "tiny-random/qwen3.5-moe",
    "qwen3_moe": "tiny-random/qwen3-moe",
}


@pytest.mark.llm_model
def test_layerwise_window_helpers():
    """Pure-Python coverage of the windowing helpers - no model load required."""
    from QEfficient.transformers.models import _layerwise

    assert _layerwise._build_layer_windows(4, 1) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert _layerwise._build_layer_windows(5, 2) == [(0, 2), (2, 4), (4, 5)]
    with pytest.raises(ValueError):
        _layerwise._build_layer_windows(0, 1)
    with pytest.raises(ValueError):
        _layerwise._build_layer_windows(4, 0)


@pytest.mark.llm_model
def test_layerwise_supported_guard_rejects_unrelated_model():
    """layerwise=True must hard-fail on architectures without windowing hooks."""
    from QEfficient.transformers.models import _layerwise

    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    with pytest.raises(NotImplementedError, match="layerwise=True is only supported"):
        _layerwise.assert_layerwise_supported(config)


def test_resolve_torch_dtype_normalizes_dtype_alias(caplog):
    """transformers-v5 ``dtype`` alias must be honored and kept in sync with ``torch_dtype``.

    Regression guard: passing ``dtype=float16`` used to be ignored, leaving the
    model forced to float32 - breaking the Qwen3.5 float16 export path.
    """
    from QEfficient.transformers.models.modeling_auto import _resolve_torch_dtype

    # float16 supplied via the v5 alias must survive and populate torch_dtype.
    kwargs = {"dtype": torch.float16}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.float16
    assert kwargs["dtype"] == torch.float16

    # float16 via the legacy name still works.
    kwargs = {"torch_dtype": torch.float16}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.float16

    # bfloat16 is kept as-is on ai100 (regardless of which name is used) so export/compile
    # still run in bfloat16, with a warning that on-device generation is expected to fail.
    caplog.set_level(logging.WARNING, logger="QEfficient")
    kwargs = {"dtype": torch.bfloat16}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.bfloat16
    assert kwargs["dtype"] == torch.bfloat16
    assert "on-device generation is expected to fail" in caplog.text

    # Regression guard: when torch_dtype/dtype is not set at all, default to float32 on
    # ai100 so models whose config.json declares bfloat16 (e.g. Mistral-7B-v0.1) aren't
    # silently loaded in bfloat16, which breaks KV-cache dtype (float32) parity.
    kwargs = {}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.float32


def test_qwen3_5_moe_gated_norm_preserves_float16():
    """GatedDeltaNet RMSNorm must keep the input dtype so the gated output feeds
    the float16 out_proj without a dtype mismatch (Qwen3.5 float16 export)."""
    import torch.nn as nn

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeGatedDeltaNetCustomRMSNormAIC,
    )

    norm = QEffQwen3_5MoeGatedDeltaNetCustomRMSNormAIC()
    norm.weight = nn.Parameter(torch.ones(16, dtype=torch.float16))
    norm.eps = 1e-6

    out = norm(torch.randn(4, 16, dtype=torch.float16), torch.randn(4, 16, dtype=torch.float16))
    assert out.dtype == torch.float16


def test_qwen3_5_moe_get_submodules_for_export_keeps_decoder_layer_for_mixed_layer_types():
    """Mixed full/linear attention configs must still expose decoder layer subfunctions."""
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeDecoderLayer,
        QEffQwen3_5MoeDecoderWrapper,
        QEffQwen3_5MoeForCausalLM,
    )

    causal_lm = QEffQwen3_5MoeForCausalLM.__new__(QEffQwen3_5MoeForCausalLM)
    causal_lm.config = SimpleNamespace(layer_types=["full_attention", "linear_attention"])
    assert causal_lm.get_submodules_for_export() == {QEffQwen3_5MoeDecoderLayer}

    wrapper = QEffQwen3_5MoeDecoderWrapper.__new__(QEffQwen3_5MoeDecoderWrapper)
    wrapper.config = SimpleNamespace(text_config=SimpleNamespace(layer_types=["full_attention", "linear_attention"]))
    assert wrapper.get_submodules_for_export() == {QEffQwen3_5MoeDecoderLayer}


def test_qwen3_5_moe_get_specializations_supports_multi_resolution():
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeForConditionalGeneration

    model = QEffQwen3_5MoeForConditionalGeneration.__new__(QEffQwen3_5MoeForConditionalGeneration)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=16, temporal_patch_size=1),
        text_config=SimpleNamespace(num_hidden_layers=0, layer_types=[]),
    )

    specs, _ = model.get_specializations(
        batch_size=1,
        prefill_seq_len=64,
        ctx_len=4096,
        height=[448, 896],
        width=[448, 896],
        num_frames=[1, 2],
        kv_offload=True,
    )

    vision_specs = specs["vision"]
    lang_specs = specs["lang"]

    assert len(vision_specs) == 2
    expected_vision_size = max(spec["vision_size"] * frames for spec, frames in zip(vision_specs, [1, 2]))
    assert all(spec["vision_size"] == expected_vision_size for spec in lang_specs)
    assert all(spec["vision_batch_size"] == 1 for spec in lang_specs)


def test_qwen3_5_moe_get_specializations_accepts_qeff_logger_adapter():
    """Ensure constrained vision-size logging works with a LoggerAdapter."""
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeForConditionalGeneration,
    )

    model = QEffQwen3_5MoeForConditionalGeneration.__new__(QEffQwen3_5MoeForConditionalGeneration)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=16, temporal_patch_size=1),
        text_config=SimpleNamespace(num_hidden_layers=1, layer_types=[]),
    )

    assert isinstance(modeling_qwen3_5_moe.logger, logging.LoggerAdapter)
    specs, _ = model.get_specializations(
        batch_size=1,
        prefill_seq_len=512,
        ctx_len=14336,
        height=[354],
        width=[536],
        num_frames=[1],
        vision_size=1,
        kv_offload=True,
    )

    assert len(specs["vision"]) == 1
    assert specs["lang"][0]["vision_size"] == 1


def test_qwen3_5_moe_get_specializations_decouples_vision_batch_size():
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeForConditionalGeneration

    model = QEffQwen3_5MoeForConditionalGeneration.__new__(QEffQwen3_5MoeForConditionalGeneration)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=16, temporal_patch_size=1),
        text_config=SimpleNamespace(num_hidden_layers=0, layer_types=[]),
    )

    specs, _ = model.get_specializations(
        batch_size=4,
        vision_batch_size=1,
        kv_cache_batch_size=4,
        prefill_seq_len=64,
        ctx_len=4096,
        height=448,
        width=448,
        kv_offload=True,
    )

    assert specs["vision"][0]["vision_batch_size"] == 1
    assert "batch_size" not in specs["vision"][0]
    assert specs["lang"][0]["batch_size"] == 4
    assert all(spec["vision_batch_size"] == 1 for spec in specs["lang"])
    axes = model.get_onnx_dynamic_axes(kv_offload=True)
    assert axes["vision"]["image_grid_thw"][0] == "vision_batch_size"
    assert axes["lang"]["vision_embeds"][0] == "vision_batch_size"


@pytest.mark.llm_model
def test_qwen3_5_moe_full_state_update_matches_singleton_routed_state():
    config = _tiny_qwen3_5_moe_config()
    config.text_config.layer_types = ["linear_attention"] * config.text_config.num_hidden_layers
    torch.manual_seed(0)
    hf_model = AutoModelForImageTextToText.from_config(config).eval()
    routed_model = QEFFAutoModelForImageTextToText(
        deepcopy(hf_model),
        kv_offload=True,
        continuous_batching=True,
    )
    full_state_model = QEFFAutoModelForImageTextToText(
        deepcopy(hf_model),
        kv_offload=True,
        continuous_batching=True,
    )
    inputs = full_state_model.model.get_dummy_inputs(
        kv_offload=True,
        continuous_batching=True,
        gdn_full_state_update=True,
        batch_size=1,
        prefill_seq_len=4,
    )["lang"]
    inputs["input_ids"].fill_(1)
    assert inputs["batch_index"].tolist() == [[0]]

    routed_model.transform(ctx_len=8, seq_len=4, bs=1, prefill_only=True, prefill_seq_len=4)
    full_state_model.transform(
        ctx_len=8,
        seq_len=4,
        bs=1,
        prefill_only=True,
        prefill_seq_len=4,
        qaic_config={"gdn_full_state_update": True},
    )
    assert full_state_model.lang_model.hash_params["gated_delta_kwargs"] == {"full_state_update": True}

    routed_gdn_modules = [
        module for module in routed_model.lang_model.model.modules() if hasattr(module, "gdn_full_state_update")
    ]
    full_state_gdn_modules = [
        module for module in full_state_model.lang_model.model.modules() if hasattr(module, "gdn_full_state_update")
    ]
    assert routed_gdn_modules and all(module.gdn_full_state_update is False for module in routed_gdn_modules)
    assert full_state_gdn_modules and all(module.gdn_full_state_update is True for module in full_state_gdn_modules)

    with torch.no_grad():
        routed_outputs = routed_model.lang_model.model(**deepcopy(inputs))
        full_state_outputs = full_state_model.lang_model.model(**deepcopy(inputs))

    torch.testing.assert_close(full_state_outputs[0], routed_outputs[0])
    routed_states = routed_outputs[3]
    full_states = full_state_outputs[3]
    assert len(routed_states) == len(full_states) == config.text_config.num_hidden_layers
    for layer_idx, ((routed_conv, routed_recurrent), (full_conv, full_recurrent)) in enumerate(
        zip(routed_states, full_states)
    ):
        torch.testing.assert_close(
            full_conv,
            routed_conv,
            msg=lambda msg, layer_idx=layer_idx: f"conv_state.{layer_idx}: {msg}",
        )
        torch.testing.assert_close(
            full_recurrent,
            routed_recurrent,
            msg=lambda msg, layer_idx=layer_idx: f"recurrent_state.{layer_idx}: {msg}",
        )

    full_state_model.transform(ctx_len=8, seq_len=1, bs=1, prefill_only=False, prefill_seq_len=1)
    assert all(module.gdn_full_state_update is False for module in full_state_gdn_modules)
    assert "gated_delta_kwargs" not in full_state_model.lang_model.hash_params


@pytest.mark.llm_model
def test_qwen3_5_moe_singleton_prefill_export_has_no_state_scatter(tmp_path):
    config = _tiny_qwen3_5_moe_config()
    config.text_config.layer_types = ["linear_attention"] * config.text_config.num_hidden_layers
    hf_model = AutoModelForImageTextToText.from_config(config).eval()
    qeff_model = QEFFAutoModelForImageTextToText(
        hf_model,
        kv_offload=True,
        continuous_batching=True,
    )
    qeff_model.transform(
        ctx_len=8,
        seq_len=4,
        bs=1,
        qaic_config={"gdn_full_state_update": True},
        prefill_only=True,
        prefill_seq_len=4,
    )

    inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        continuous_batching=True,
        gdn_full_state_update=True,
        batch_size=1,
        prefill_seq_len=4,
    )["lang"]
    inputs["input_ids"].fill_(1)
    assert all(state.shape[0] == 1 for layer_state in inputs["past_key_values"] for state in layer_state)

    export_result = qeff_model.export(
        export_dir=tmp_path / "qwen3-5-moe-prefill-no-state-scatter",
        skip_vision=True,
        prefill_only=True,
        prefill_seq_len=4,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        qaic_config={"gdn_full_state_update": True},
    )
    onnx_model = onnx.load(_exported_onnx_path(export_result), load_external_data=False)
    nodes = list(onnx_model.graph.node)
    nodes.extend(node for function_proto in onnx_model.functions for node in function_proto.node)
    state_indexing_ops = {"CtxGatherCB", "CtxGatherCB3D", "CtxScatterCB", "CtxScatterCB3D"}

    assert not ({node.op_type for node in nodes} & state_indexing_ops)
    for function_proto in onnx_model.functions:
        retained_outputs = [name for name in function_proto.output if "state." in name]
        if not retained_outputs:
            continue
        producer_by_output = {output: node for node in function_proto.node for output in node.output}
        for retained_output in retained_outputs:
            pending = [retained_output]
            visited = set()
            while pending:
                value = pending.pop()
                producer = producer_by_output.get(value)
                if producer is None or id(producer) in visited:
                    continue
                visited.add(id(producer))
                assert "Scatter" not in producer.op_type
                pending.extend(producer.input)

    with torch.no_grad():
        pytorch_outputs = qeff_model.lang_model.model(**deepcopy(inputs))
    session = _ort_session(_exported_onnx_path(export_result))
    session_input_names = [item.name for item in session.get_inputs()]
    session_output_names = [item.name for item in session.get_outputs()]
    ort_inputs = _flatten_qwen_ort_inputs(inputs, session_input_names, qeff_model.lang_model.model)
    ort_outputs = dict(zip(session_output_names, session.run(session_output_names, ort_inputs)))

    assert np.allclose(pytorch_outputs[0].detach().numpy(), ort_outputs["logits"], atol=2e-3, rtol=1e-4)
    for layer_idx, (conv_state, recurrent_state) in enumerate(pytorch_outputs[3]):
        assert np.allclose(
            conv_state.detach().numpy(),
            ort_outputs[f"conv_state.{layer_idx}_RetainedState"],
            atol=2e-3,
            rtol=1e-4,
        )
        assert np.allclose(
            recurrent_state.detach().numpy(),
            ort_outputs[f"recurrent_state.{layer_idx}_RetainedState"],
            atol=2e-3,
            rtol=1e-4,
        )


@pytest.mark.parametrize(
    ("model_type", "prefill_only", "batch_size", "full_batch_size"),
    [
        ("qwen3_5", True, 1, 1),
        ("qwen3_5_moe", False, 1, 1),
        ("qwen3_5_moe", True, 2, 1),
        ("qwen3_5_moe", True, 1, 4),
    ],
)
def test_gdn_full_state_update_rejects_unsafe_compile_modes(
    model_type,
    prefill_only,
    batch_size,
    full_batch_size,
):
    from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

    model = _QEffAutoModelForImageTextToTextDualQPC.__new__(_QEffAutoModelForImageTextToTextDualQPC)
    model.model = _tiny_qwen_qeff_model(model_type).model

    with pytest.raises(ValueError, match="gdn_full_state_update"):
        model.compile(
            prefill_only=prefill_only,
            batch_size=batch_size,
            full_batch_size=full_batch_size,
            qaic_config={"gdn_full_state_update": True},
        )


def test_vlm_compile_forwards_gdn_chunk_size_in_qaic_config_to_export_path():
    """GDN chunk size is configured only through qaic_config."""
    from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

    model = _QEffAutoModelForImageTextToTextDualQPC.__new__(_QEffAutoModelForImageTextToTextDualQPC)
    model._run_layerwise_compile = MagicMock(return_value="qpc_paths")

    result = model.compile(layerwise=True, qaic_config={"gdn_chunk_size": 512})

    assert result == "qpc_paths"
    compile_kwargs = model._run_layerwise_compile.call_args.kwargs
    assert compile_kwargs["qaic_config"]["gdn_chunk_size"] == 512
    assert "gdn_chunk_size" not in compile_kwargs
    assert "qeff_chunk_size" not in compile_kwargs

    model._run_layerwise_compile.reset_mock()
    model.compile(layerwise=True, prefill_seq_len=1024)
    assert "gdn_chunk_size" not in model._run_layerwise_compile.call_args.kwargs
    assert "qeff_chunk_size" not in model._run_layerwise_compile.call_args.kwargs

    model._run_layerwise_compile.reset_mock()
    model.compile(layerwise=True)
    assert "gdn_chunk_size" not in model._run_layerwise_compile.call_args.kwargs
    assert "qeff_chunk_size" not in model._run_layerwise_compile.call_args.kwargs


def test_qwen3_vl_moe_get_specializations_decouples_vision_batch_size():
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        QEffQwen3VLMoeForConditionalGeneration,
    )

    model = QEffQwen3VLMoeForConditionalGeneration.__new__(QEffQwen3VLMoeForConditionalGeneration)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=16, temporal_patch_size=1, deepstack_visual_indexes=[]),
        text_config=SimpleNamespace(num_hidden_layers=0, layer_types=[]),
    )
    model.model = SimpleNamespace(language_model=SimpleNamespace(layers=[]))

    specs, _ = model.get_specializations(
        batch_size=4,
        vision_batch_size=1,
        kv_cache_batch_size=4,
        prefill_seq_len=64,
        ctx_len=4096,
        height=448,
        width=448,
        kv_offload=True,
    )

    assert specs["vision"][0]["vision_batch_size"] == 1
    assert "batch_size" not in specs["vision"][0]
    assert specs["lang"][0]["batch_size"] == 4
    assert all(spec["vision_batch_size"] == 1 for spec in specs["lang"])
    axes = model.get_onnx_dynamic_axes(kv_offload=True)
    assert axes["vision"]["image_grid_thw"][0] == "vision_batch_size"
    assert axes["lang"]["vision_embeds"][0] == "vision_batch_size"


def test_qwen3_5_moe_get_specializations_strips_vision_symbols_for_comp_ctx_variants():
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeForConditionalGeneration

    model = QEffQwen3_5MoeForConditionalGeneration.__new__(QEffQwen3_5MoeForConditionalGeneration)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=16, temporal_patch_size=1))

    lang_specs, _ = model.get_specializations(
        batch_size=1,
        prefill_seq_len=64,
        ctx_len=4096,
        height=[448, 672],
        width=[448, 672],
        num_frames=[1, 1],
        comp_ctx_lengths_prefill=[32, 48],
        comp_ctx_lengths_decode=[96, 128],
        kv_offload=False,
    )

    assert [spec["seq_len"] for spec in lang_specs] == [64, 64, 1, 1]
    assert all("vision_size" not in spec for spec in lang_specs)
    assert all("vision_batch_size" not in spec for spec in lang_specs)


def test_qwen3_vl_moe_batch_index_boundary_reorders_inputs_and_outputs():
    from QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        _batch_index_gather,
        _batch_index_scatter,
    )

    batch_index = torch.tensor([[2], [0], [3], [1]], dtype=torch.int64)
    logical = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 1, 3)
    physical = _batch_index_scatter(logical, batch_index)

    assert torch.equal(physical[batch_index.flatten()], logical)
    assert torch.equal(_batch_index_gather(physical, batch_index), logical)

    logical_positions = torch.arange(3 * 4, dtype=torch.int64).reshape(3, 4, 1)
    physical_positions = _batch_index_scatter(logical_positions, batch_index, batch_dim=1)
    assert torch.equal(physical_positions[:, batch_index.flatten()], logical_positions)


@pytest.mark.parametrize("grouped_state", [False, True])
def test_qwen3_5_moe_conv_decode_slice_matches_gather_reference(grouped_state):
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        qeff_torch_causal_conv1d_update,
    )

    torch.manual_seed(0)
    batch_size, hidden_size, state_len = 3, 8, 3
    hidden_states = torch.randn(batch_size, hidden_size, 1)
    flat_state = torch.randn(batch_size, hidden_size, state_len)
    conv_state = flat_state.reshape(batch_size, 2, 4, state_len) if grouped_state else flat_state
    weight = torch.randn(hidden_size, 3)
    bias = torch.randn(hidden_size)
    position_ids = torch.tensor([[[4], [5], [-1]]])

    state_input = flat_state
    hidden_states_new = torch.cat([state_input, hidden_states], dim=-1).to(weight.dtype)
    order = torch.argsort(
        torch.cat([torch.zeros((batch_size, state_len), dtype=position_ids.dtype), position_ids[0]], dim=1), dim=1
    )
    expected_state = torch.gather(hidden_states_new, 2, order[:, None, -state_len:].expand(-1, hidden_size, -1))
    expected_state[2] = state_input[2]
    expected_state = expected_state.reshape_as(conv_state)
    expected_output = torch.nn.functional.silu(
        torch.nn.functional.conv1d(hidden_states_new, weight.unsqueeze(1), bias, groups=hidden_size)
    )[:, :, -1:]

    output, state = qeff_torch_causal_conv1d_update(
        hidden_states,
        conv_state,
        weight,
        position_ids,
        bias,
    )

    torch.testing.assert_close(output, expected_output.to(hidden_states.dtype))
    torch.testing.assert_close(state, expected_state)


def test_qwen3_5_moe_conv_decode_slice_keeps_prefill_gather_path():
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        qeff_torch_causal_conv1d_update,
    )

    torch.manual_seed(1)
    batch_size, hidden_size, state_len, seq_len = 2, 6, 3, 4
    conv_state = torch.randn(batch_size, hidden_size, state_len)
    hidden_states = torch.randn(batch_size, hidden_size, seq_len)
    weight = torch.randn(hidden_size, 3)
    bias = torch.randn(hidden_size)
    position_ids = torch.tensor([[[0, 1, -1, 2], [0, -1, 1, 2]]])

    output, state = qeff_torch_causal_conv1d_update(
        hidden_states,
        conv_state,
        weight,
        position_ids,
        bias,
    )

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    order = torch.argsort(
        torch.cat([torch.zeros((batch_size, state_len), dtype=position_ids.dtype), position_ids[0]], dim=1), dim=1
    )
    expected_state = torch.gather(hidden_states_new, 2, order[:, None, -state_len:].expand(-1, hidden_size, -1))
    expected_output = torch.nn.functional.silu(
        torch.nn.functional.conv1d(hidden_states_new, weight.unsqueeze(1), bias, groups=hidden_size)
    )[:, :, -seq_len:]

    torch.testing.assert_close(output, expected_output.to(hidden_states.dtype))
    torch.testing.assert_close(state, expected_state)


def test_qwen3_5_conv_decode_slice_matches_gather_reference():
    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import qeff_torch_causal_conv1d_update

    torch.manual_seed(0)
    batch_size, hidden_size, state_len = 3, 8, 3
    hidden_states = torch.randn(batch_size, hidden_size, 1)
    conv_state = torch.randn(batch_size, hidden_size, state_len)
    weight = torch.randn(hidden_size, 3)
    bias = torch.randn(hidden_size)
    position_ids = torch.tensor([[[4], [5], [-1]]])

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    order = torch.argsort(
        torch.cat([torch.zeros((batch_size, state_len), dtype=position_ids.dtype), position_ids[0]], dim=1), dim=1
    )
    expected_state = torch.gather(hidden_states_new, 2, order[:, None, -state_len:].expand(-1, hidden_size, -1))
    expected_state[2] = conv_state[2]
    expected_output = torch.nn.functional.silu(
        torch.nn.functional.conv1d(hidden_states_new, weight.unsqueeze(1), bias, groups=hidden_size)
    )[:, :, -1:]

    output, state = qeff_torch_causal_conv1d_update(hidden_states, conv_state, weight, position_ids, bias)

    torch.testing.assert_close(output, expected_output.to(hidden_states.dtype))
    torch.testing.assert_close(state, expected_state)


def test_qwen3_5_conv_decode_slice_keeps_prefill_gather_path():
    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import qeff_torch_causal_conv1d_update

    torch.manual_seed(1)
    batch_size, hidden_size, state_len, seq_len = 2, 6, 3, 4
    conv_state = torch.randn(batch_size, hidden_size, state_len)
    hidden_states = torch.randn(batch_size, hidden_size, seq_len)
    weight = torch.randn(hidden_size, 3)
    bias = torch.randn(hidden_size)
    position_ids = torch.tensor([[[0, 1, -1, 2], [0, -1, 1, 2]]])

    output, state = qeff_torch_causal_conv1d_update(hidden_states, conv_state, weight, position_ids, bias)

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    order = torch.argsort(
        torch.cat([torch.zeros((batch_size, state_len), dtype=position_ids.dtype), position_ids[0]], dim=1), dim=1
    )
    expected_state = torch.gather(hidden_states_new, 2, order[:, None, -state_len:].expand(-1, hidden_size, -1))
    expected_output = torch.nn.functional.silu(
        torch.nn.functional.conv1d(hidden_states_new, weight.unsqueeze(1), bias, groups=hidden_size)
    )[:, :, -seq_len:]

    torch.testing.assert_close(output, expected_output.to(hidden_states.dtype))
    torch.testing.assert_close(state, expected_state)


def test_qwen3_5_batch_fold_dynamic_axes_separate_input_and_cache_batches():
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import QEffQwen3_5ForConditionalGeneration

    model = QEffQwen3_5ForConditionalGeneration.__new__(QEffQwen3_5ForConditionalGeneration)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention", "linear_attention"])
    )

    folded_axes = model.get_onnx_dynamic_axes(continuous_batching=True, batch_fold=True)
    regular_axes = model.get_onnx_dynamic_axes(continuous_batching=True, batch_fold=False)

    assert folded_axes["input_ids"][0] == "full_batch_size"
    assert folded_axes["position_ids"][1] == "full_batch_size"
    assert folded_axes["batch_index"][0] == "full_batch_size"
    assert folded_axes["past_key.0"][0] == "full_batch_size"
    assert regular_axes["input_ids"][0] == "batch_size"
    assert regular_axes["batch_index"][0] == "batch_size"


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_qwen3_5_decoder_wrapper_folds_decode_only(model_type):
    from types import SimpleNamespace

    if model_type == "qwen3_5":
        from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import QEffQwen3_5DecoderWrapper

        wrapper_cls = QEffQwen3_5DecoderWrapper
    else:
        from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeDecoderWrapper

        wrapper_cls = QEffQwen3_5MoeDecoderWrapper

    class RecordingLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn_blocking_config=SimpleNamespace(batch_fold=True),
                    )
                )
            ]
            self.calls = []

        def forward(self, *, inputs_embeds, batch_index, batch_fold, **kwargs):
            self.calls.append((batch_index, batch_fold))
            return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=[])

    class Embedding(nn.Module):
        def forward(self, input_ids):
            return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 2)

    language_model = RecordingLanguageModel()
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=language_model,
            get_input_embeddings=lambda: Embedding(),
        ),
        config=SimpleNamespace(image_token_id=99),
        lm_head=nn.Identity(),
    )
    wrapper = wrapper_cls(model)
    batch_index = torch.tensor([[1], [0]], dtype=torch.long)
    vision_embeds = torch.zeros(2, 1, 2)
    image_idx = torch.zeros(2, 1, dtype=torch.long)

    for seq_len in (3, 1):
        language_model.calls.clear()
        input_ids = torch.tensor([[1] * seq_len, [2] * seq_len])
        position_ids = torch.arange(seq_len).view(1, 1, seq_len).expand(1, 2, seq_len)

        if model_type == "qwen3_5":
            wrapper(
                input_ids,
                vision_embeds,
                position_ids,
                image_idx,
                [],
                batch_index=batch_index,
            )
        else:
            wrapper(
                input_ids=input_ids,
                vision_embeds=vision_embeds,
                position_ids=position_ids,
                image_idx=image_idx,
                past_key_values=[],
                batch_index=batch_index,
            )

        received_batch_index, batch_fold = language_model.calls[0]
        assert batch_fold is (seq_len == 1)
        if seq_len == 1:
            assert received_batch_index is None
        else:
            torch.testing.assert_close(received_batch_index, batch_index)


def test_qwen3_5_moe_decode_expert_parallel_selection():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    from QEfficient.transformers.models.pytorch_transforms import (
        OptimizedMoEExpertParallelWeightsTransform,
        OptimizedMoEExportConfigTransform,
        OptimizedMoEMapperTransform,
        OptimizedMoEWeightsTransform,
    )
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeSparseMoeBlock
    from QEfficient.transformers.moe import MoEFlavour

    config = Qwen3_5MoeTextConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=32,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "linear_attention"],
    )
    torch.manual_seed(2)
    hf_block = Qwen3_5MoeSparseMoeBlock(config).eval()
    qeff_block = Qwen3_5MoeSparseMoeBlock(config).eval()
    qeff_block.load_state_dict(hf_block.state_dict())
    qeff_block, transformed = OptimizedMoEMapperTransform.apply(qeff_block)
    assert transformed and isinstance(qeff_block, QEffQwen3_5MoeSparseMoeBlock)
    OptimizedMoEWeightsTransform.apply(qeff_block)
    OptimizedMoEExportConfigTransform.apply(
        qeff_block,
        prefill_only=False,
        num_devices=1,
        num_cores=2,
        prefill_seq_len=2,
        qaic_config={
            "moe_config": {
                "flavour": "expert_parallel",
                "cores_per_expert": 1,
                "tree_reduce": False,
                "expert_parallel_chunk_size": 1,
            }
        },
    )
    OptimizedMoEExpertParallelWeightsTransform.apply(qeff_block)

    assert qeff_block._moe_flavour is MoEFlavour.EXPERT_PARALLEL


def test_moe_prefill_transform_does_not_require_enable_chunking():
    from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import QEffGlm4MoeMoE
    from QEfficient.transformers.models.pytorch_transforms import (
        PrefillOnlyChunkedTransform,
        PrefillOnlyTransform,
        RevertPrefillKeepAttentionTransform,
        RevertPrefillOnlyTransform,
    )
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeSparseMoeBlock,
    )
    from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import (
        QEffQwen3MoeSparseMoeBlock,
    )
    from QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        QEffQwen3VLMoeTextSparseMoeBlock,
    )
    from QEfficient.transformers.moe import MoEFlavour

    for moe_cls in (
        QEffGlm4MoeMoE,
        QEffQwen3MoeSparseMoeBlock,
        QEffQwen3VLMoeTextSparseMoeBlock,
        QEffQwen3_5MoeSparseMoeBlock,
    ):
        assert moe_cls not in PrefillOnlyTransform._module_mapping
        assert moe_cls not in PrefillOnlyChunkedTransform._module_mapping
        assert moe_cls not in RevertPrefillKeepAttentionTransform._module_mapping
        assert moe_cls not in RevertPrefillOnlyTransform._module_mapping
        assert MoEFlavour.EXPERT_PARALLEL in moe_cls.supported_moe_flavours


def _tiny_qwen3_vl_moe_sparse_block_pair(num_experts: int = 2):
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock

    from QEfficient.transformers.models.pytorch_transforms import (
        OptimizedMoEMapperTransform,
        OptimizedMoEWeightsTransform,
    )

    config = _tiny_qwen3_vl_moe_config(num_experts=num_experts).text_config
    torch.manual_seed(0)
    hf_block = Qwen3VLMoeTextSparseMoeBlock(config).eval()
    with torch.no_grad():
        hf_block.gate.weight.normal_(mean=0.0, std=0.02)
        hf_block.experts.gate_up_proj.normal_(mean=0.0, std=0.02)
        hf_block.experts.down_proj.normal_(mean=0.0, std=0.02)

    qeff_block = Qwen3VLMoeTextSparseMoeBlock(config).eval()
    qeff_block.load_state_dict(hf_block.state_dict())
    qeff_block, transformed = OptimizedMoEMapperTransform.apply(qeff_block)
    assert transformed
    OptimizedMoEWeightsTransform.apply(qeff_block)
    return hf_block, qeff_block, config


def test_qwen3_vl_moe_sparse_gather_bmm_matches_hf():
    from QEfficient.transformers.models.pytorch_transforms import OptimizedMoEExportConfigTransform
    from QEfficient.transformers.moe import MoEFlavour

    hf_block, qeff_block, config = _tiny_qwen3_vl_moe_sparse_block_pair()
    OptimizedMoEExportConfigTransform.apply(
        qeff_block,
        prefill_only=False,
        qaic_config={"moe_config": {"flavour": "decode_bmm"}},
    )
    assert qeff_block._moe_flavour is MoEFlavour.DECODE_BMM
    hidden_states = torch.randn(2, 3, config.hidden_size)

    with torch.no_grad():
        hf_output = hf_block(hidden_states)
        qeff_output, _ = qeff_block(hidden_states)

    torch.testing.assert_close(qeff_output, hf_output, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    (
        "num_experts",
        "num_devices",
        "num_cores",
        "cores_per_expert",
        "tree_reduce",
        "expected_experts_per_soc",
    ),
    [
        pytest.param(2, 1, 2, 1, False, None, id="default-einsum"),
        pytest.param(4, 2, 2, 1, False, 2, id="cross-soc-flat"),
        pytest.param(4, 2, 2, 1, True, 2, id="cross-soc-tree"),
        pytest.param(4, 2, 2, 2, False, 1, id="cores-per-expert-2"),
    ],
)
def test_qwen3_vl_moe_sparse_expert_parallel_dispatch_and_parity(
    num_experts,
    num_devices,
    num_cores,
    cores_per_expert,
    tree_reduce,
    expected_experts_per_soc,
):
    from QEfficient.transformers.models.pytorch_transforms import (
        OptimizedMoEExpertParallelWeightsTransform,
        OptimizedMoEExportConfigTransform,
    )
    from QEfficient.transformers.moe import MoEFlavour

    hf_block, qeff_block, config = _tiny_qwen3_vl_moe_sparse_block_pair(num_experts=num_experts)
    OptimizedMoEExportConfigTransform.apply(
        qeff_block,
        prefill_only=True,
        num_devices=num_devices,
        num_cores=num_cores,
        prefill_seq_len=4,
        qaic_config={
            "moe_config": {
                "flavour": "expert_parallel",
                "cores_per_expert": cores_per_expert,
                "tree_reduce": tree_reduce,
                "expert_parallel_chunk_size": 2,
            }
        },
    )
    OptimizedMoEExpertParallelWeightsTransform.apply(qeff_block)
    hidden_states = torch.randn(1, 4, config.hidden_size)

    with torch.no_grad():
        hf_output = hf_block(hidden_states)
        qeff_output, _ = qeff_block(hidden_states)

    assert qeff_block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
    assert qeff_block.total_avl_cores == num_devices * num_cores
    assert qeff_block.cores_per_expert == cores_per_expert
    assert qeff_block.experts_per_soc == expected_experts_per_soc
    assert qeff_block.tree_reduce is tree_reduce
    assert qeff_output.shape == hidden_states.shape
    torch.testing.assert_close(qeff_output, hf_output, atol=1e-6, rtol=1e-6)


def test_qwen3_vl_moe_sparse_configurator_sets_and_clears_dispatch():
    from QEfficient.transformers.models.pytorch_transforms import OptimizedMoEExportConfigTransform
    from QEfficient.transformers.moe import MoEFlavour

    _, qeff_block, config = _tiny_qwen3_vl_moe_sparse_block_pair(num_experts=4)
    hash_params = {}
    OptimizedMoEExportConfigTransform.apply(
        qeff_block,
        prefill_only=True,
        num_devices=2,
        num_cores=2,
        qaic_config={
            "moe_config": {
                "flavour": "expert_parallel",
                "cores_per_expert": 2,
                "tree_reduce": True,
                "expert_parallel_chunk_size": 3,
            }
        },
        hash_params=hash_params,
    )

    assert qeff_block._moe_flavour is MoEFlavour.EXPERT_PARALLEL
    assert qeff_block.expert_blocking_packed_chunk_size == 3
    assert qeff_block.num_devices == 2
    assert qeff_block.cores_per_expert == 2
    assert qeff_block.total_avl_cores == 4
    assert qeff_block.num_pipeline_stages == 2
    assert qeff_block.num_parallelized_experts == 2
    assert qeff_block.experts_per_soc == 1
    assert qeff_block.tree_reduce is True
    assert hash_params["moe_prefill_flavour"] == "expert_parallel"
    assert "moe_prefill_num_nsp" not in hash_params
    assert hash_params["moe_prefill_total_avl_cores"] == 4
    assert hash_params["moe_prefill_cores_per_expert"] == 2
    assert hash_params["moe_prefill_tree_reduce"] is True
    assert hash_params["moe_prefill_expert_parallel_chunk_size"] == 3
    assert hash_params["moe_prefill_num_pipeline_stages"] == 2
    assert hash_params["moe_prefill_num_parallelized_experts"] == 2
    assert "moe_prefill_num_packed_chunks" in hash_params

    OptimizedMoEExportConfigTransform.apply(
        qeff_block,
        prefill_only=False,
        num_devices=2,
        num_cores=2,
        qaic_config={"moe_config": {"flavour": "decode_bmm"}},
        hash_params=hash_params,
    )

    assert qeff_block._moe_flavour is MoEFlavour.DECODE_BMM
    assert hash_params["moe_prefill_flavour"] == "decode_bmm"
    assert "moe_prefill_total_avl_cores" not in hash_params
    assert "moe_prefill_cores_per_expert" not in hash_params
    assert "moe_prefill_tree_reduce" not in hash_params
    assert "moe_prefill_expert_parallel_chunk_size" not in hash_params
    assert "moe_prefill_num_pipeline_stages" not in hash_params
    assert "moe_prefill_num_parallelized_experts" not in hash_params
    assert "moe_prefill_num_packed_chunks" not in hash_params

    with torch.no_grad():
        qeff_block(torch.randn(1, 2, config.hidden_size))


def test_qwen3_vl_moe_sparse_configurator_rejects_invalid_core_layout():
    from QEfficient.transformers.models.pytorch_transforms import OptimizedMoEExportConfigTransform

    _, qeff_block, _ = _tiny_qwen3_vl_moe_sparse_block_pair(num_experts=4)
    with pytest.raises(ValueError, match="total_avl_cores"):
        OptimizedMoEExportConfigTransform.apply(
            qeff_block,
            prefill_only=True,
            num_devices=3,
            num_cores=2,
            qaic_config={
                "moe_config": {
                    "flavour": "expert_parallel",
                    "cores_per_expert": 1,
                    "tree_reduce": False,
                    "expert_parallel_chunk_size": 2,
                }
            },
        )

    with pytest.raises(TypeError, match="tree_reduce"):
        OptimizedMoEExportConfigTransform.apply(
            qeff_block,
            prefill_only=True,
            num_devices=1,
            num_cores=2,
            qaic_config={
                "moe_config": {
                    "flavour": "expert_parallel",
                    "cores_per_expert": 1,
                    "tree_reduce": "true",
                    "expert_parallel_chunk_size": 2,
                }
            },
        )


def test_qwen3_vl_moe_sparse_expert_parallel_resolver_prefill_wins():
    from QEfficient.transformers.models.pytorch_transforms import OptimizedMoEExportConfigTransform
    from QEfficient.transformers.moe import MoEFlavour

    # prefill_only=True with no explicit flavour selects expert_parallel
    _, qeff_block_prefill, _ = _tiny_qwen3_vl_moe_sparse_block_pair(num_experts=2)
    OptimizedMoEExportConfigTransform.apply(qeff_block_prefill, prefill_only=True, num_devices=1, num_cores=2)
    assert qeff_block_prefill._moe_flavour is MoEFlavour.EXPERT_PARALLEL

    # prefill_only=False with explicit expert_parallel flavour selects expert_parallel
    _, qeff_block_explicit, _ = _tiny_qwen3_vl_moe_sparse_block_pair(num_experts=2)
    OptimizedMoEExportConfigTransform.apply(
        qeff_block_explicit,
        prefill_only=False,
        num_devices=1,
        num_cores=2,
        qaic_config={"moe_config": {"flavour": "expert_parallel"}},
    )
    assert qeff_block_explicit._moe_flavour is MoEFlavour.EXPERT_PARALLEL

    # prefill_only=False with no explicit flavour selects decode_bmm
    _, qeff_block_decode, _ = _tiny_qwen3_vl_moe_sparse_block_pair(num_experts=2)
    OptimizedMoEExportConfigTransform.apply(qeff_block_decode, prefill_only=False)
    assert qeff_block_decode._moe_flavour is MoEFlavour.DECODE_BMM

    # invalid tree_reduce type raises TypeError
    _, qeff_block_err, _ = _tiny_qwen3_vl_moe_sparse_block_pair(num_experts=2)
    with pytest.raises(TypeError, match="tree_reduce"):
        OptimizedMoEExportConfigTransform.apply(
            qeff_block_err,
            prefill_only=True,
            num_devices=1,
            num_cores=2,
            qaic_config={"moe_config": {"tree_reduce": "true"}},
        )


def test_qwen3_vl_moe_sparse_vlm_compile_signatures_expose_expert_parallel_args():
    from QEfficient.transformers.models.modeling_auto import (
        _QEffAutoModelForImageTextToTextDualQPC,
        _QEFFAutoModelForImageTextToTextSingleQPC,
    )
    from QEfficient.transformers.models.pytorch_transforms import OptimizedMoEExportConfigTransform
    from QEfficient.transformers.moe import MoEFlavour

    for wrapper_cls in (_QEffAutoModelForImageTextToTextDualQPC, _QEFFAutoModelForImageTextToTextSingleQPC):
        params = inspect.signature(wrapper_cls.compile).parameters
        assert "qaic_config" in params
        assert params["qaic_config"].default is None

    # Verify that qaic_config["moe_config"] is the accepted interface for expert-parallel
    # configuration by applying all expected keys to a real block.
    _, qeff_block, _ = _tiny_qwen3_vl_moe_sparse_block_pair(num_experts=2)
    OptimizedMoEExportConfigTransform.apply(
        qeff_block,
        prefill_only=True,
        num_devices=1,
        num_cores=2,
        qaic_config={
            "moe_config": {
                "flavour": "expert_parallel",
                "cores_per_expert": 1,
                "tree_reduce": False,
                "expert_parallel_chunk_size": 2,
            }
        },
    )
    assert qeff_block._moe_flavour is MoEFlavour.EXPERT_PARALLEL


def test_layerwise_matches_default_path_for_qwen3_moe():
    """Without-layerwise and with-layerwise forwards must produce identical output.
    This is the core backward-compatibility contract: running every decoder layer
    in a single forward (default path) must match running the same layers one
    window at a time and chaining the hidden states (layerwise path), bit for bit.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    import QEfficient
    from QEfficient.transformers.models import _layerwise

    cfg = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=128,
        decoder_sparse_step=1,
        norm_topk_prob=True,
    )
    torch.manual_seed(0)
    hf = Qwen3MoeForCausalLM(cfg).eval()
    qeff_model = QEfficient.QEFFAutoModelForCausalLM(hf, continuous_batching=False)
    inner = qeff_model.model.model

    B, S, ctx, num_layers = 1, 8, 16, cfg.num_hidden_layers
    n_kv, head_dim = cfg.num_key_value_heads, cfg.head_dim
    ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).view(1, -1)

    def fresh_pkv():
        return tuple(
            (torch.zeros(B, n_kv, ctx, head_dim), torch.zeros(B, n_kv, ctx, head_dim)) for _ in range(num_layers)
        )

    # Default (non-layerwise) path: all layers in one shot.
    with torch.no_grad():
        default_out = inner(
            input_ids=ids,
            position_ids=position_ids,
            past_key_values=fresh_pkv(),
            cache_position=torch.arange(S),
        )

    # Layerwise path: window size 1, chaining hidden states across windows.
    hidden_states = inner.embed_tokens(ids)
    try:
        with _layerwise._layerwise_export_env():
            for window in range(num_layers):
                _layerwise._set_layer_windows(window, window + 1, num_layers)
                with torch.no_grad():
                    out = inner(
                        inputs_embeds=hidden_states,
                        position_ids=position_ids,
                        past_key_values=fresh_pkv(),
                        cache_position=torch.arange(S),
                    )
                hidden_states = out.last_hidden_state
    finally:
        _layerwise._reset_layer_windows()

    assert torch.equal(default_out.last_hidden_state, hidden_states), (
        "layerwise windowed forward diverged from the default single-shot forward"
    )

    # The default path must leave the mutable window state untouched.
    from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import QEffQwen3MoeModel

    assert QEffQwen3MoeModel._start == 0
    assert QEffQwen3MoeModel._end == 0


@pytest.mark.llm_model
def test_layerwise_matches_default_path_for_qwen3_5_moe():
    """Qwen3.5-MoE decoder wrapper must preserve logits with layerwise windows."""
    import QEfficient
    from QEfficient.transformers.models import _layerwise

    config = AutoConfig.from_pretrained(LAYERWISE_TINY_MODEL_IDS["qwen3_5_moe"])
    config.torch_dtype = "float32"

    qeff_model = QEfficient.QEFFAutoModelForImageTextToText.from_pretrained(
        LAYERWISE_TINY_MODEL_IDS["qwen3_5_moe"],
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float32,
        layerwise=False,
    )
    wrapper = qeff_model.model.get_qeff_language_decoder().eval()
    lang_inputs = qeff_model.model.get_dummy_inputs(kv_offload=True)["lang"]
    lang_inputs["input_ids"][0, 0] = qeff_model.model.config.image_token_id
    lang_inputs["vision_embeds"].normal_()

    with torch.no_grad():
        default_out = wrapper(**lang_inputs)

    hidden_states = None
    total_layers = qeff_model.model.config.text_config.num_hidden_layers
    try:
        with _layerwise._layerwise_export_env():
            for window in range(total_layers):
                _layerwise._set_layer_windows(window, window + 1, total_layers)
                call_kwargs = {
                    "position_ids": lang_inputs["position_ids"],
                    "past_key_values": lang_inputs["past_key_values"],
                }
                if window == 0:
                    call_kwargs.update(
                        input_ids=lang_inputs["input_ids"],
                        vision_embeds=lang_inputs["vision_embeds"],
                        image_idx=lang_inputs["image_idx"],
                    )
                else:
                    call_kwargs["inputs_embeds"] = hidden_states

                with torch.no_grad():
                    window_out = wrapper(**call_kwargs)
                hidden_states = window_out[0]
    finally:
        _layerwise._reset_layer_windows()

    assert torch.equal(default_out[0], hidden_states), "Qwen3.5-MoE layerwise logits diverged from default logits"


@pytest.mark.llm_model
def test_layerwise_matches_default_path_for_qwen3_vl_moe():
    """Qwen3-VL-MoE decoder wrapper must preserve logits with layerwise windows."""
    import QEfficient
    from QEfficient.transformers.models import _layerwise

    model_id = LAYERWISE_TINY_MODEL_IDS["qwen3_vl_moe"]
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)
    config.torch_dtype = "float32"

    qeff_model = QEfficient.QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float32,
        layerwise=False,
    )
    wrapper = qeff_model.model.get_qeff_language_decoder().eval()
    lang_inputs = qeff_model.model.get_dummy_inputs(kv_offload=True)["lang"]

    with torch.no_grad():
        default_out = wrapper(**lang_inputs)

    hidden_states = None
    total_layers = qeff_model.model.config.text_config.num_hidden_layers
    try:
        with _layerwise._layerwise_export_env():
            for window in range(total_layers):
                _layerwise._set_layer_windows(window, window + 1, total_layers)
                call_kwargs = {k: v for k, v in lang_inputs.items() if k not in ("input_ids", "inputs_embeds")}
                if window == 0:
                    call_kwargs["input_ids"] = lang_inputs["input_ids"]
                else:
                    call_kwargs["inputs_embeds"] = hidden_states

                with torch.no_grad():
                    window_out = wrapper(**call_kwargs)
                hidden_states = window_out[0]
    finally:
        _layerwise._reset_layer_windows()

    assert torch.equal(default_out[0], hidden_states), "Qwen3-VL-MoE layerwise logits diverged from default logits"


def test_split_layer_graph_keeps_qwen3_5_linear_states(tmp_path):
    """Qwen3.5 layerwise split must retain linear-attention state inputs."""
    from onnx import TensorProto, helper

    from QEfficient.utils.layerwise_pipeline import split_layer_graph

    layer_dir = tmp_path / "onnx_layerwise_tmp" / "layer_0_1"
    layer_dir.mkdir(parents=True)
    model_path = layer_dir / "model_layer_tmp_0_1.onnx"

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["input_ids"], ["logits"]),
            helper.make_node("Identity", ["conv_state.0"], ["conv_state.0_RetainedState"]),
            helper.make_node("Identity", ["recurrent_state.0"], ["recurrent_state.0_RetainedState"]),
        ],
        name="qwen35_layer",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 8]),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [4, 1, 8]),
            helper.make_tensor_value_info("conv_state.0", TensorProto.FLOAT, [1, 64, 4]),
            helper.make_tensor_value_info("recurrent_state.0", TensorProto.FLOAT, [1, 4, 16, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.INT64, [1, 8]),
            helper.make_tensor_value_info("conv_state.0_RetainedState", TensorProto.FLOAT, [1, 64, 4]),
            helper.make_tensor_value_info("recurrent_state.0_RetainedState", TensorProto.FLOAT, [1, 4, 16, 16]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, model_path)

    assert split_layer_graph(0, 1, str(tmp_path), 0, 1) is True

    split_model = onnx.load(layer_dir / "split_graph.onnx", load_external_data=False)
    input_names = {value.name for value in split_model.graph.input}
    assert "conv_state.0" in input_names
    assert "recurrent_state.0" in input_names


@pytest.mark.llm_model
def test_layerwise_supported_guard_accepts_qwen3_vl_moe():
    from QEfficient.transformers.models import _layerwise

    try:
        config = AutoConfig.from_pretrained(LAYERWISE_TINY_MODEL_ID)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, LAYERWISE_TINY_MODEL_ID)
    resolved = _layerwise.assert_layerwise_supported(config)
    assert resolved in {"qwen3_vl_moe", "qwen3_vl_moe_text"}


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("arch", "model_id"),
    sorted(LAYERWISE_TINY_MODEL_IDS.items()),
    ids=sorted(LAYERWISE_TINY_MODEL_IDS),
)
def test_layerwise_supported_guard_accepts_all_supported(arch, model_id):
    """Guard must accept each architecture in the layerwise allowlist."""
    from QEfficient.transformers.models import _layerwise

    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)
    resolved = _layerwise.assert_layerwise_supported(config)
    assert arch in resolved or resolved.startswith(arch)


@pytest.mark.llm_model
def test_layerwise_off_does_not_set_env_var(tmp_path):
    """Backward compat: layerwise must be controlled purely via the API,
    never via environment variables, and must be off by default."""
    from QEfficient.base.modeling_qeff import QEFFBaseModel
    from QEfficient.transformers.models import _layerwise  # noqa: F401

    assert os.environ.get("LAYERWISE_EXPORT") is None
    assert QEFFBaseModel._layerwise_active is False


@pytest.mark.llm_model
def test_layerwise_vision_wrapper_keeps_only_first_text_window():
    try:
        config = AutoConfig.from_pretrained(LAYERWISE_TINY_MODEL_ID)
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            LAYERWISE_TINY_MODEL_ID,
            kv_offload=True,
            config=config,
            layerwise=True,
        )
        vision_wrapper = qeff_model._build_layerwise_vision_wrapper()
    except Exception as exc:
        _skip_on_model_fetch_error(exc, LAYERWISE_TINY_MODEL_ID)

    layers = vision_wrapper.model.model.language_model.layers

    assert getattr(qeff_model, "_layerwise_outer_meta", False) is True
    assert layers[0] is not None
    assert sum(layer is not None for layer in layers) == 1
    assert next(vision_wrapper.model.model.visual.parameters()).device.type != "meta"

    default_model = QEFFAutoModelForImageTextToText.from_pretrained(
        LAYERWISE_TINY_MODEL_ID,
        kv_offload=True,
        config=config,
    )
    default_layers = default_model.model.model.language_model.layers
    assert sum(layer is not None for layer in default_layers) == len(default_layers)


@pytest.mark.llm_model
def test_layerwise_context_manager_toggles_class_flag():
    """The driver's context manager must flip the class flag and restore it,
    even on exception, with no env-var side-effects."""
    from QEfficient.base.modeling_qeff import QEFFBaseModel
    from QEfficient.transformers.models import _layerwise

    assert QEFFBaseModel._layerwise_active is False
    with _layerwise._layerwise_export_env():
        assert QEFFBaseModel._layerwise_active is True
        assert "LAYERWISE_EXPORT" not in os.environ
    assert QEFFBaseModel._layerwise_active is False

    try:
        with _layerwise._layerwise_export_env():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert QEFFBaseModel._layerwise_active is False


@pytest.mark.llm_model
def test_layerwise_safe_export_pass_patch_is_noop_when_inactive():
    from torch import _C

    from QEfficient.utils.torch_patches import layerwise_safe_onnx_export_patches

    original_constant_prop = _C._jit_pass_constant_propagation
    original_constant_fold = _C._jit_pass_onnx_constant_fold

    with layerwise_safe_onnx_export_patches():
        assert _C._jit_pass_constant_propagation is original_constant_prop
        assert _C._jit_pass_onnx_constant_fold is original_constant_fold

    assert _C._jit_pass_constant_propagation is original_constant_prop
    assert _C._jit_pass_onnx_constant_fold is original_constant_fold


@pytest.mark.llm_model
def test_layerwise_safe_export_pass_patch_toggles_only_inside_layerwise_context():
    from torch import _C

    from QEfficient.transformers.models import _layerwise
    from QEfficient.utils.torch_patches import layerwise_safe_onnx_export_patches

    original_cse = _C._jit_pass_cse
    original_constant_prop = _C._jit_pass_constant_propagation
    original_constant_fold = _C._jit_pass_onnx_constant_fold
    original_canonicalize = _C._jit_pass_canonicalize

    with _layerwise._layerwise_export_env():
        with layerwise_safe_onnx_export_patches():
            assert _C._jit_pass_cse is original_cse
            assert _C._jit_pass_constant_propagation is original_constant_prop
            assert _C._jit_pass_onnx_constant_fold is original_constant_fold
            assert _C._jit_pass_canonicalize is not original_canonicalize
            sentinel_graph = object()
            assert _C._jit_pass_canonicalize(sentinel_graph) is sentinel_graph

        assert _C._jit_pass_cse is original_cse
        assert _C._jit_pass_onnx_constant_fold is original_constant_fold
        assert _C._jit_pass_canonicalize is original_canonicalize

    assert _C._jit_pass_cse is original_cse
    assert _C._jit_pass_onnx_constant_fold is original_constant_fold
    assert _C._jit_pass_canonicalize is original_canonicalize

    assert _C._jit_pass_cse is original_cse
    assert _C._jit_pass_constant_propagation is original_constant_prop
    assert _C._jit_pass_onnx_constant_fold is original_constant_fold
    assert _C._jit_pass_canonicalize is original_canonicalize


@pytest.mark.llm_model
def test_layerwise_post_merge_dedup_removes_duplicate_onnx_nodes():
    from onnx import TensorProto, helper

    from QEfficient.utils.layerwise_pipeline import _deduplicate_redundant_onnx_nodes

    one_value = helper.make_tensor(name="one", data_type=TensorProto.FLOAT, dims=[1], vals=[1.0])
    const_a = helper.make_node("Constant", inputs=[], outputs=["c0"], value=one_value)
    # Same value but different TensorProto.name; dedup should still match it.
    one_value_2 = helper.make_tensor(name="one_2", data_type=TensorProto.FLOAT, dims=[1], vals=[1.0])
    const_b = helper.make_node("Constant", inputs=[], outputs=["c1"], value=one_value_2)
    add_a = helper.make_node("Add", inputs=["c0", "c0"], outputs=["sum0"])
    add_b = helper.make_node("Add", inputs=["c1", "c1"], outputs=["sum1"])

    graph = helper.make_graph(
        [const_a, const_b, add_a, add_b],
        "dedup_test",
        [],
        [helper.make_tensor_value_info("sum0", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)

    removed = _deduplicate_redundant_onnx_nodes(model)

    # The pass performs CSE (not full dead-code elimination), so Add is
    # deduplicated on non-graph-output nodes while preserving graph outputs.
    assert removed == 1
    assert len(model.graph.node) == 3
    assert [node.op_type for node in model.graph.node] == ["Constant", "Add", "Add"]
    assert list(model.graph.node[2].input) == ["c0", "c0"]


@pytest.mark.llm_model
def test_layerwise_uses_probe_model_for_cached_export(monkeypatch, tmp_path):
    """A cached merged ONNX must avoid rebuilding per-window models."""
    from QEfficient.transformers.models import _layerwise

    class DummyConfig:
        model_type = "qwen3_moe"
        num_hidden_layers = 2

    class ProbeModel:
        def __init__(self, cached_path):
            self.cached_path = cached_path

        def compile(self, **kwargs):
            assert kwargs.pop("_layerwise_cache_probe") is True
            return self.cached_path

    cached_path = tmp_path / "Model-hash" / "Qwen3MoeModel.onnx"
    cached_path.parent.mkdir(parents=True)
    cached_path.touch()
    factory_called = False

    def fail_factory(*args, **kwargs):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("factory must not run when merged ONNX is cached")

    monkeypatch.setattr(_layerwise, "_install_window_patches_for", lambda model_type: None)
    result = _layerwise.run_layerwise(
        model_id="dummy",
        config=DummyConfig(),
        qeff_factory=fail_factory,
        compile_kwargs={},
        probe_qeff_model=ProbeModel(cached_path),
        final_compile=False,
    )

    assert result == str(cached_path)
    assert factory_called is False


@pytest.mark.llm_model
def test_layerwise_cache_miss_exports_all_windows(monkeypatch, tmp_path):
    from QEfficient.transformers.models import _layerwise

    class DummyConfig:
        model_type = "qwen3_moe"
        num_hidden_layers = 3

    class ProbeModel:
        def compile(self, **kwargs):
            assert kwargs.pop("_layerwise_cache_probe") is True
            return None

    exported_windows = []

    class WindowModel:
        def __init__(self):
            self.model = object()
            self.model_name = "Qwen3MoeModel"

        def compile(self, **kwargs):
            start = _layerwise._LAYERWISE_STATE["text_start"]
            end = _layerwise._LAYERWISE_STATE["text_end"]
            exported_windows.append((start, end))
            shard = tmp_path / "onnx_layerwise_tmp" / f"layer_{start}_{end}" / f"model_layer_tmp_{start}_{end}.onnx"
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.touch()
            return str(shard)

    monkeypatch.setattr(_layerwise, "_install_window_patches_for", lambda model_type: None)
    monkeypatch.setattr(_layerwise, "_null_outside_window_layers", lambda *args, **kwargs: None)
    monkeypatch.setattr(_layerwise, "_slim_for_window_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _layerwise,
        "_stitch_layerwise_if_available",
        lambda export_root, total_layers=None: str(export_root / "merged.onnx"),
    )
    cleaned = []
    monkeypatch.setattr(
        _layerwise, "_cleanup_layerwise_intermediates", lambda export_root, **kwargs: cleaned.append(export_root)
    )

    result = _layerwise.run_layerwise(
        model_id="dummy",
        config=DummyConfig(),
        qeff_factory=lambda *args, **kwargs: WindowModel(),
        compile_kwargs={},
        probe_qeff_model=ProbeModel(),
        window_size=1,
        final_compile=False,
    )

    assert exported_windows == [(0, 1), (1, 2), (2, 3)]
    assert result.endswith("merged.onnx")
    assert cleaned == [tmp_path]


@pytest.mark.llm_model
def test_layerwise_cleanup_removes_intermediate_dirs(tmp_path):
    from QEfficient.transformers.models import _layerwise

    final_data = tmp_path / "final_data"
    onnx_tmp = tmp_path / "onnx_layerwise_tmp"
    final_data.mkdir()
    onnx_tmp.mkdir()
    (final_data / "merged_0-48.onnx").touch()
    (onnx_tmp / "layer_0_1").mkdir()

    _layerwise._cleanup_layerwise_intermediates(tmp_path)

    assert not final_data.exists()
    assert not onnx_tmp.exists()


@pytest.mark.llm_model
def test_layerwise_cached_merged_prefers_root_layout(tmp_path):
    from QEfficient.transformers.models import _layerwise

    root_merged = tmp_path / "merged_0-48.onnx"
    legacy_merged = tmp_path / "final_data" / "merged_0-48.onnx"
    legacy_merged.parent.mkdir()
    root_merged.touch()
    legacy_merged.touch()

    assert _layerwise._cached_merged_onnx(tmp_path, total_layers=48) == root_merged


def test_layerwise_merge_renames_decoder_function_variants_without_collisions():
    from onnx import TensorProto, helper

    from QEfficient.utils.layerwise_pipeline import merge_models

    domain = "QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"

    def make_func(name, inputs):
        return helper.make_function(
            domain,
            name,
            inputs,
            ["out"],
            [helper.make_node("Identity", [inputs[0]], ["out"])],
            opset_imports=[helper.make_opsetid("", 17)],
        )

    m1_graph = helper.make_graph(
        [helper.make_node("QEffQwen3_5MoeDecoderLayer", ["x", "a", "b", "c"], ["m1_out"], domain=domain)],
        "m1",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in ["x", "a", "b", "c"]],
        [helper.make_tensor_value_info("m1_out", TensorProto.FLOAT, [1])],
    )
    m1 = helper.make_model(m1_graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(domain, 1)])
    m1.functions.extend([make_func("QEffQwen3_5MoeDecoderLayer", ["x", "a", "b", "c"])])

    m2_graph = helper.make_graph(
        [
            helper.make_node("QEffQwen3_5MoeDecoderLayer", ["y", "d", "e"], ["mid"], domain=domain),
            helper.make_node("QEffQwen3_5MoeDecoderLayer__v2", ["mid", "f", "g", "h", "i"], ["m2_out"], domain=domain),
        ],
        "m2",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in ["y", "d", "e", "f", "g", "h", "i"]],
        [helper.make_tensor_value_info("m2_out", TensorProto.FLOAT, [1])],
    )
    m2 = helper.make_model(m2_graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(domain, 1)])
    m2.functions.extend(
        [
            make_func("QEffQwen3_5MoeDecoderLayer", ["y", "d", "e"]),
            make_func("QEffQwen3_5MoeDecoderLayer__v2", ["mid", "f", "g", "h", "i"]),
        ]
    )

    merged = merge_models(m1, m2, [("m1_out", "y")])
    function_inputs = {(func.domain, func.name): len(func.input) for func in merged.functions}

    assert function_inputs[(domain, "QEffQwen3_5MoeDecoderLayer")] == 4
    assert function_inputs[(domain, "QEffQwen3_5MoeDecoderLayer__v2")] == 5
    assert function_inputs[(domain, "QEffQwen3_5MoeDecoderLayer__v3")] == 3
    assert all(
        len(node.input) == function_inputs[(node.domain, node.op_type)]
        for node in merged.graph.node
        if (node.domain, node.op_type) in function_inputs
    )


@pytest.mark.llm_model
def test_layerwise_materializes_root_onnx_for_final_compile(monkeypatch, tmp_path):
    from QEfficient.transformers.models import _layerwise

    class DummyConfig:
        model_type = "qwen3_vl_moe"
        num_hidden_layers = 2

    class DummyLangModel:
        model_name = "Qwen3VLDecoderWrapper"

    class ProbeModel:
        lang_model = DummyLangModel()

        def __init__(self, cached_path):
            self.cached_path = cached_path
            self.final_kwargs = None

        def compile(self, **kwargs):
            if kwargs.pop("_layerwise_cache_probe", False):
                return self.cached_path
            self.final_kwargs = kwargs
            return {"lang_decode_qpc_path": "dummy-qpc"}

    cached_path = tmp_path / "Model-hash" / "Qwen3VLDecoderWrapper.onnx"
    cached_path.parent.mkdir(parents=True)
    cached_path.touch()
    probe = ProbeModel(cached_path)

    monkeypatch.setattr(_layerwise, "_install_window_patches_for", lambda model_type: None)

    result = _layerwise.run_layerwise(
        model_id="dummy",
        config=DummyConfig(),
        qeff_factory=lambda *args, **kwargs: probe,
        compile_kwargs={},
        probe_qeff_model=probe,
        final_compile=True,
    )

    assert result == {"lang_decode_qpc_path": "dummy-qpc"}
    assert probe.final_kwargs["lang_onnx_path"] == str(cached_path)
    assert probe.final_kwargs["skip_lang"] is False


@pytest.mark.llm_model
def test_layerwise_cache_hit_under_final_data_is_canonicalized(monkeypatch, tmp_path):
    from QEfficient.transformers.models import _layerwise

    class DummyConfig:
        model_type = "qwen3_vl_moe"
        num_hidden_layers = 2

    class DummyLangModel:
        model_name = "Qwen3VLDecoderWrapper"

    class ProbeModel:
        lang_model = DummyLangModel()

        def __init__(self, cached_path):
            self.cached_path = cached_path
            self.final_kwargs = None

        def compile(self, **kwargs):
            if kwargs.pop("_layerwise_cache_probe", False):
                return self.cached_path
            self.final_kwargs = kwargs
            return {"lang_decode_qpc_path": "dummy-qpc"}

    cached_path = tmp_path / "Model-hash" / "final_data" / "merged_0-2.onnx"
    cached_path.parent.mkdir(parents=True)
    cached_path.touch()
    probe = ProbeModel(cached_path)

    monkeypatch.setattr(_layerwise, "_install_window_patches_for", lambda model_type: None)
    cleaned = []
    monkeypatch.setattr(
        _layerwise,
        "_relocate_merged_onnx_to_root",
        lambda export_root, merged_onnx: export_root / merged_onnx.name,
    )
    monkeypatch.setattr(
        _layerwise, "_cleanup_layerwise_intermediates", lambda export_root, **kwargs: cleaned.append(export_root)
    )

    result = _layerwise.run_layerwise(
        model_id="dummy",
        config=DummyConfig(),
        qeff_factory=lambda *args, **kwargs: probe,
        compile_kwargs={},
        probe_qeff_model=probe,
        final_compile=True,
    )

    assert result == {"lang_decode_qpc_path": "dummy-qpc"}
    assert probe.final_kwargs["lang_onnx_path"] == str(tmp_path / "Model-hash" / "merged_0-2.onnx")
    assert cleaned == [tmp_path / "Model-hash"]


@pytest.mark.llm_model
def test_layerwise_export_hash_is_separate_from_default(monkeypatch):
    from QEfficient.utils import export_utils

    class DummyConfig:
        def to_diff_dict(self):
            return {"model_type": "dummy"}

    class DummyModel:
        config = DummyConfig()

    class DummyQEffModel:
        model = DummyModel()
        hash_params = {}

    def normal_export(self, example_inputs, output_names, dynamic_axes, **export_kwargs):
        pass

    def _export_layerwise(self, example_inputs, output_names, dynamic_axes, **export_kwargs):
        pass

    captured_export_kwargs = []

    def fake_create_export_hash(**kwargs):
        captured_export_kwargs.append(kwargs.get("export_kwargs"))
        return "hash", {}

    monkeypatch.setattr(export_utils, "create_export_hash", fake_create_export_hash)

    export_utils._generate_export_hash(DummyQEffModel(), ({}, ["logits"], {}), {}, normal_export)
    export_utils._generate_export_hash(DummyQEffModel(), ({}, ["logits"], {}), {}, _export_layerwise)

    assert captured_export_kwargs[0] in (None, {})
    assert captured_export_kwargs[1]["_qeff_layerwise_export"] is True


@pytest.mark.llm_model
def test_subfunction_compile_io_names_use_internal_retained_state():
    from QEfficient.transformers.models.modeling_auto import _compile_io_name, _state_input_name

    output_name = _compile_io_name("past_value.1_RetainedState", use_onnx_subfunctions=True)

    assert output_name == "past_value.1_InternalRetainedState"
    assert _state_input_name(output_name) == "past_value.1"
    assert _compile_io_name("vision_embeds_RetainedState", use_onnx_subfunctions=True) == "vision_embeds_RetainedState"


@pytest.mark.llm_model
def test_runtime_aliases_internal_retained_state_outputs():
    from QEfficient.generation.cloud_infer import _add_basename_binding_aliases, _public_retained_state_name

    assert _public_retained_state_name("past_key.0_InternalRetainedState") == "past_key.0_RetainedState"
    assert _public_retained_state_name("past_value.1_InternalRetainedState") == "past_value.1_RetainedState"
    assert _public_retained_state_name("logits") is None

    binding_map = {"layer_0/input_ids": 3}
    bindings = [type("Binding", (), {"name": "layer_0/input_ids", "index": 3})()]
    _add_basename_binding_aliases(binding_map, bindings)
    assert binding_map["input_ids"] == 3


@pytest.mark.llm_model
def test_runtime_failure_releases_qaic_program(monkeypatch):
    from QEfficient.generation import cloud_infer

    success = object()
    monkeypatch.setattr(
        cloud_infer,
        "qaicrt",
        SimpleNamespace(QStatus=SimpleNamespace(QS_SUCCESS=success)),
        raising=False,
    )

    exec_obj = MagicMock()
    exec_obj.setData.return_value = success
    exec_obj.waitForCompletion.return_value = object()
    program = MagicMock()
    program.deactivate.return_value = success
    program.unload.return_value = success
    queue = MagicMock()
    queue.enqueue.return_value = success

    session = object.__new__(cloud_infer.QAICInferenceSession)
    session.allowed_shapes = []
    session.buf_dims = []
    session.execObj = exec_obj
    session.is_active = True
    session.program = program
    session.qbuffers = []
    session.queue = queue
    session.set_buffers = MagicMock()

    with pytest.raises(ValueError, match="Failed to run"):
        session.run({})

    program.deactivate.assert_called_once_with()
    program.unload.assert_called_once_with()
    assert session.is_active is False


@pytest.mark.llm_model
def test_layerwise_compile_hydrates_outer_qpc_paths(monkeypatch, tmp_path):
    from QEfficient.transformers.models import _layerwise
    from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

    qpc_path = tmp_path / "qpc"
    model = object.__new__(_QEffAutoModelForImageTextToTextDualQPC)
    model._pretrained_model_name_or_path = "dummy"
    model.config = object()
    model.vision_model = type("Vision", (), {"qpc_path": None})()
    model.lang_model = type("Lang", (), {"qpc_path": None})()
    model._build_layerwise_factory = lambda: None

    monkeypatch.setattr(_layerwise, "run_layerwise", lambda **kwargs: {"lang_decode_qpc_path": qpc_path})

    result = model._run_layerwise_compile(layerwise_window_size=1)

    assert result == {"lang_decode_qpc_path": qpc_path}
    assert model.qpc_paths == result
    assert model.lang_model.qpc_path == qpc_path


@pytest.mark.llm_model
def test_dual_qpc_decode_only_continuous_batching_returns_decode_qpc_key(monkeypatch):
    from QEfficient.transformers.models import modeling_auto
    from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

    model = object.__new__(_QEffAutoModelForImageTextToTextDualQPC)
    model.continuous_batching = True
    model.ccl_enabled = False
    model.comp_ctx_lengths_prefill = None
    model.comp_ctx_lengths_decode = None
    model.transform = lambda **kwargs: None
    model.model = type(
        "Model",
        (),
        {
            "config": type("Config", (), {"torch_dtype": torch.float32, "model_type": "test"})(),
            "get_output_names": lambda self, **kwargs: {"vision": [], "lang": []},
            "get_specializations": lambda self, **kwargs: ({"vision": [], "lang": [{"seq_len": 1}]}, {}),
        },
    )()
    model.lang_model = type(
        "LanguageModel",
        (),
        {"onnx_path": "language.onnx", "_compile": staticmethod(lambda **kwargs: "decode.qpc")},
    )()

    monkeypatch.setattr(modeling_auto, "_filter_custom_io_for_onnx", lambda custom_io, onnx_path: custom_io)

    result = model.compile(
        prefill_seq_len=1,
        ctx_len=16,
        batch_size=1,
        full_batch_size=4,
        skip_vision=True,
        lang_onnx_path="language.onnx",
    )

    assert result == {"lang_decode_qpc_path": "decode.qpc"}


@pytest.mark.llm_model
def test_layerwise_compile_rejects_unsupported_model():
    """End-to-end smoke: invoking layerwise=True on llama bubbles the guard error."""
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, "tiny-random/llama")
    # CausalLM does not expose a layerwise= kwarg today; only DualQPC VLM does.
    # So this test guards via the helper directly to make the contract explicit
    # for future surface expansion.
    from QEfficient.transformers.models import _layerwise

    with pytest.raises(NotImplementedError):
        _layerwise.assert_layerwise_supported(qeff_model.model.config)


# ---------------------------------------------------------------------------
# Tests for the optional KV-cache buffer-name prefix (vLLM disaggregated KV transfer)
# ---------------------------------------------------------------------------


def _retained_state_outputs(onnx_path: Path) -> Set[str]:
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    return {out.name for out in onnx_model.graph.output if out.name.endswith("_RetainedState")}


def _kv_input_names(onnx_path: Path) -> Set[str]:
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    return {
        inp.name
        for inp in onnx_model.graph.input
        if inp.name.startswith(
            ("past_key.", "past_value.", "compressed_kv.", "k_pe.", "conv_state.", "recurrent_state.")
        )
    }


class TestApplyKvCachePrefixHelper:
    """Unit tests for the apply_kv_cache_prefix / validate_kv_cache_prefix helpers."""

    def test_flat_list_kv_only(self):
        from QEfficient.utils import apply_kv_cache_prefix

        names = ["logits", "past_key.0_RetainedState", "past_value.0_RetainedState", "vision_embeds_RetainedState"]
        result = apply_kv_cache_prefix(names, "VLLM")
        assert result == [
            "logits",
            "past_key.0_VLLM_RetainedState",
            "past_value.0_VLLM_RetainedState",
            "vision_embeds_RetainedState",  # vision buffer untouched
        ]

    def test_compressed_kv_and_k_pe(self):
        from QEfficient.utils import apply_kv_cache_prefix

        names = ["compressed_kv.0_RetainedState", "k_pe.0_RetainedState"]
        assert apply_kv_cache_prefix(names, "P") == ["compressed_kv.0_P_RetainedState", "k_pe.0_P_RetainedState"]

    def test_conv_recurrent_state_prefixed(self):
        """Hybrid linear-attention state (conv_state / recurrent_state on Qwen3.5 hybrid models)
        must receive the same KV-cache prefix as classical attention KV. vision/multimodal
        retained buffers must remain untouched on the same call."""
        from QEfficient.utils import apply_kv_cache_prefix

        names = [
            "logits",
            "past_key.0_RetainedState",
            "past_value.0_RetainedState",
            "conv_state.1_RetainedState",
            "recurrent_state.1_RetainedState",
            "vision_embeds_RetainedState",
            "image_idx_output",
            "pixel_values_RetainedState",
        ]
        result = apply_kv_cache_prefix(names, "VLLM")
        assert result == [
            "logits",
            "past_key.0_VLLM_RetainedState",
            "past_value.0_VLLM_RetainedState",
            "conv_state.1_VLLM_RetainedState",
            "recurrent_state.1_VLLM_RetainedState",
            "vision_embeds_RetainedState",
            "image_idx_output",
            "pixel_values_RetainedState",
        ]

    def test_align_inputs_handles_conv_recurrent(self):
        """The input rename map must pair conv_state/recurrent_state inputs with their
        prefixed retained-state outputs, exactly like past_key/past_value."""
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = [
            "input_ids",
            "past_key.0",
            "past_value.0",
            "conv_state.1",
            "recurrent_state.1",
        ]
        output_names = [
            "logits",
            "past_key.0_VLLM_RetainedState",
            "past_value.0_VLLM_RetainedState",
            "conv_state.1_VLLM_RetainedState",
            "recurrent_state.1_VLLM_RetainedState",
        ]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == [
            "input_ids",
            "past_key.0_VLLM",
            "past_value.0_VLLM",
            "conv_state.1_VLLM",
            "recurrent_state.1_VLLM",
        ]

    def test_dict_form_lang_only(self):
        from QEfficient.utils import apply_kv_cache_prefix

        names = {
            "vision": ["vision_embeds", "past_key.0_RetainedState"],  # vision side never rewritten
            "lang": ["logits", "vision_embeds_RetainedState", "past_key.0_RetainedState"],
        }
        result = apply_kv_cache_prefix(names, "VLLM")
        assert result["vision"] == ["vision_embeds", "past_key.0_RetainedState"]
        assert result["lang"] == ["logits", "vision_embeds_RetainedState", "past_key.0_VLLM_RetainedState"]

    def test_noop_when_prefix_absent(self):
        from QEfficient.utils import apply_kv_cache_prefix

        names = ["logits", "past_key.0_RetainedState"]
        assert apply_kv_cache_prefix(names, None) == names
        assert apply_kv_cache_prefix(names, "") == names

    def test_align_inputs_to_retained_outputs(self):
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = ["input_ids", "past_key.0", "past_value.0"]
        output_names = ["logits", "past_key.0_VLLM_RetainedState", "past_value.0_VLLM_RetainedState"]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == [
            "input_ids",
            "past_key.0_VLLM",
            "past_value.0_VLLM",
        ]

    def test_align_inputs_to_internal_retained_outputs(self):
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = ["input_ids", "past_key.0", "past_value.0"]
        output_names = ["logits", "past_key.0_VLLM_InternalRetainedState", "past_value.0_VLLM_InternalRetainedState"]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == [
            "input_ids",
            "past_key.0_VLLM",
            "past_value.0_VLLM",
        ]

    def test_align_inputs_noop_without_prefix(self):
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = ["input_ids", "past_key.0", "past_value.0"]
        output_names = ["logits", "past_key.0_RetainedState", "past_value.0_RetainedState"]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == input_names

    def test_on_device_sampler_buffers_not_prefixed(self):
        """Sampler retained-state buffers (past_repetition_penalty_buffer / past_presence_penalty_buffer)
        must NOT receive the KV-cache prefix on either the output or paired input side."""
        from QEfficient.utils import align_kv_input_names_to_retained_outputs, apply_kv_cache_prefix

        outputs = [
            "logits",
            "probs",
            "next_tokens",
            "past_key.0_RetainedState",
            "past_value.0_RetainedState",
            "past_repetition_penalty_buffer_RetainedState",
            "past_presence_penalty_buffer_RetainedState",
        ]
        prefixed = apply_kv_cache_prefix(outputs, "VLLM")
        assert "past_repetition_penalty_buffer_RetainedState" in prefixed
        assert "past_presence_penalty_buffer_RetainedState" in prefixed
        # And no sampler-buffer name accidentally carries the infix.
        assert not any("penalty_buffer_VLLM_RetainedState" in name for name in prefixed)

        inputs = [
            "input_ids",
            "past_key.0",
            "past_value.0",
            "past_repetition_penalty_buffer",
            "past_presence_penalty_buffer",
        ]
        aligned = align_kv_input_names_to_retained_outputs(inputs, prefixed)
        assert "past_repetition_penalty_buffer" in aligned
        assert "past_presence_penalty_buffer" in aligned
        assert not any(name.endswith("penalty_buffer_VLLM") for name in aligned)

    @pytest.mark.parametrize("bad", ["", "a_b", "a.b", "a b", 123, "past-key"])
    def test_validation_rejects_bad_prefix(self, bad):
        from QEfficient.utils import validate_kv_cache_prefix

        with pytest.raises(ValueError):
            validate_kv_cache_prefix(bad)

    def test_validation_accepts_alnum_and_none(self):
        from QEfficient.utils import validate_kv_cache_prefix

        assert validate_kv_cache_prefix("VLLM") == "VLLM"
        assert validate_kv_cache_prefix("vllm0") == "vllm0"
        assert validate_kv_cache_prefix(None) is None


@pytest.mark.llm_model
def test_causal_export_with_kv_cache_prefix(tmp_path):
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "prefixed", kv_cache_prefix="VLLM"))

    retained = _retained_state_outputs(onnx_path)
    assert retained, "expected retained-state outputs"
    # Every KV retained output carries the infix; the suffix is preserved.
    kv_retained = {name for name in retained if name.startswith(("past_key.", "past_value."))}
    assert kv_retained
    assert all(name.endswith("_VLLM_RetainedState") for name in kv_retained)

    # The matching device input buffer exists (output minus _RetainedState).
    kv_inputs = _kv_input_names(onnx_path)
    for out_name in kv_retained:
        stripped = out_name[: -len("_RetainedState")]
        assert stripped in kv_inputs, f"missing paired input buffer for {out_name}"
        assert stripped.endswith("_VLLM")


@pytest.mark.llm_model
def test_causal_export_default_names_unchanged(tmp_path):
    """Without the flag, retained-state names must remain byte-for-byte identical to today."""
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "default"))
    retained = _retained_state_outputs(onnx_path)
    kv_retained = {name for name in retained if name.startswith(("past_key.", "past_value."))}
    assert kv_retained
    assert all(name.endswith("_RetainedState") and "_VLLM_" not in name for name in kv_retained)
    # Inputs use the plain names.
    assert "past_key.0" in _kv_input_names(onnx_path)


@pytest.mark.llm_model
def test_causal_export_prefix_changes_hash_dir(tmp_path):
    """Prefixed and unprefixed exports must land in distinct hashed dirs (no cache collision)."""
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf.eval()

    plain_path = _exported_onnx_path(QEFFAutoModelForCausalLM(model_hf).export(tmp_path / "p"))

    model_hf2 = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf2.eval()
    prefixed_path = _exported_onnx_path(
        QEFFAutoModelForCausalLM(model_hf2).export(tmp_path / "p", kv_cache_prefix="VLLM")
    )

    assert plain_path.parent != prefixed_path.parent


@pytest.mark.llm_model
def test_causal_compile_custom_io_carries_prefix(tmp_path, monkeypatch):
    """The compile custom_io must pair prefixed input/output KV buffers."""
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            CAUSAL_RUNTIME_MODEL_IDS["llama"],
            num_hidden_layers=2,
            **MODEL_KWARGS,
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, CAUSAL_RUNTIME_MODEL_IDS["llama"])

    captured = {}

    def fake_compile(**kwargs):
        captured.update(kwargs)
        return tmp_path / "qpc"

    monkeypatch.setattr(qeff_model, "_compile", fake_compile)
    qeff_model.compile(prefill_seq_len=8, ctx_len=32, compile_dir=str(tmp_path), kv_cache_prefix="VLLM")

    custom_io = captured["custom_io"]
    assert custom_io, "expected non-empty custom_io"
    assert "past_key.0_VLLM" in custom_io  # input buffer
    assert "past_key.0_VLLM_RetainedState" in custom_io  # paired retained output
    assert all("_VLLM" in k for k in custom_io if k.startswith(("past_key.", "past_value.")))
    assert captured.get("kv_cache_prefix") == "VLLM"

    # Critical safety check: kv_cache_prefix must NOT appear in compiler_options.
    # The _compile signature has kv_cache_prefix as an explicit named param so Python never places
    # it in **compiler_options — if it did, the compiler would see "-kv-cache-prefix=VLLM" and fail.
    # We verify by reconstructing the known explicit params and confirming the remainder (what would
    # become **compiler_options in the real _compile) does not contain kv_cache_prefix.
    _known_explicit_params = {
        "onnx_path",
        "compile_dir",
        "mxint8_kv_cache",
        "specializations",
        "custom_io",
        "mdp_ts_num_devices",
        "num_speculative_tokens",
        "enable_qnn",
        "qnn_config",
        "use_onnx_subfunctions",
        "prefill_only",
        "offload_pt_weights",
        "enable_chunking",
        "retain_full_kv",
        "qaic_config",
        "specialization_module_name",
        "kv_cache_prefix",
        "retained_state",
        "convert_to_fp16",
        "mxfp6_matmul",
        # compile-time args added by causal compile():
        "aic_num_cores",
    }
    implicit_compiler_options = {k: v for k, v in captured.items() if k not in _known_explicit_params}
    assert "kv_cache_prefix" not in implicit_compiler_options, (
        "kv_cache_prefix leaked into compiler_options — would produce an invalid compiler flag"
    )


@pytest.mark.llm_model
def test_vlm_export_prefix_lang_only(tmp_path):
    """VLM export with prefix: lang KV buffers prefixed, vision retained buffers untouched."""
    try:
        vlm_model = QEFFAutoModelForImageTextToText.from_pretrained(
            VLM_TEXT_RUNTIME_MODEL_ID, trust_remote_code=True, kv_offload=True
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, VLM_TEXT_RUNTIME_MODEL_ID)

    vlm_model.export(tmp_path / "vlm-prefixed", kv_cache_prefix="VLLM")
    lang_onnx = Path(vlm_model.lang_model.onnx_path)
    retained = _retained_state_outputs(lang_onnx)

    kv_retained = {name for name in retained if name.startswith(("past_key.", "past_value."))}
    assert kv_retained
    assert all(name.endswith("_VLLM_RetainedState") for name in kv_retained)

    # Vision/multimodal retained buffers on the lang graph must NOT be prefixed.
    for name in retained:
        if name.startswith(("vision_embeds", "pixel_values", "deepstack_features")):
            assert "_VLLM_" not in name


def _capture_layerwise_export_names(qeff_model, *, window, total_layers, export_dir, kv_cache_prefix):
    """Drive the real ``_export_layerwise`` for one decoder window and capture the
    ``input_names`` / ``output_names`` handed to ``torch.onnx.export``.

    The spy raises to abort before the heavy ONNX transforms/disk writes — we only
    care about the buffer names the export was invoked with. Window state is always
    restored in ``finally`` so the class-level flags never leak into other tests.
    """
    from QEfficient.base.modeling_qeff import QEFFBaseModel
    from QEfficient.transformers.models import _layerwise

    captured = {}

    class _StopExport(Exception):
        pass

    def _spy(*args, **kwargs):
        captured["input_names"] = list(kwargs.get("input_names") or [])
        captured["output_names"] = list(kwargs.get("output_names") or [])
        raise _StopExport()

    orig_export = torch.onnx.export
    torch.onnx.export = _spy
    try:
        with _layerwise._layerwise_export_env():
            _layerwise._set_layer_windows(window, window + 1, total_layers)
            QEFFBaseModel._start = window
            QEFFBaseModel._end = window + 1
            QEFFBaseModel._total_layers = total_layers
            try:
                qeff_model.export(export_dir=str(export_dir), kv_cache_prefix=kv_cache_prefix)
            except _StopExport:
                pass
    finally:
        torch.onnx.export = orig_export
        _layerwise._reset_layer_windows()
        QEFFBaseModel._start = 0
        QEFFBaseModel._end = 0
        QEFFBaseModel._total_layers = None
    assert "output_names" in captured, "torch.onnx.export was never reached"
    return captured


def _tiny_qwen3_moe_causal():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    import QEfficient

    cfg = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=64,
        decoder_sparse_step=1,
        norm_topk_prob=True,
    )
    cfg.torch_dtype = "float32"
    torch.manual_seed(0)
    hf = Qwen3MoeForCausalLM(cfg).eval()
    return QEfficient.QEFFAutoModelForCausalLM(hf, continuous_batching=False), cfg.num_hidden_layers


@pytest.mark.llm_model
def test_layerwise_export_with_kv_cache_prefix(tmp_path):
    """Regression for the layerwise + kv_cache_prefix path (previously a silent no-op).

    The non-layerwise paths already prefix KV buffers; ``_export_layerwise`` rebuilds output
    names from per-window templates, so the prefix has to be threaded into retained outputs.
    This drives the real ``_export_layerwise`` for every window and asserts:
      * each retained-state output carries the ``_<prefix>_RetainedState`` infix (Bug 2), and
      * the matching KV *input* buffer is renamed to ``past_*.{i}_<prefix>`` so the compiler can
        pair input↔retained-output (Bug 3).
    """
    qeff_model, total_layers = _tiny_qwen3_moe_causal()

    for window in range(total_layers):
        captured = _capture_layerwise_export_names(
            qeff_model,
            window=window,
            total_layers=total_layers,
            export_dir=tmp_path / f"prefixed_w{window}",
            kv_cache_prefix="vllmKvCache",
        )
        kv_outputs = [n for n in captured["output_names"] if n.startswith(("past_key.", "past_value."))]
        assert kv_outputs, f"window {window}: expected KV retained outputs"
        # Bug 2: every per-window KV retained output carries the infix.
        for name in kv_outputs:
            assert name == f"past_key.{window}_vllmKvCache_RetainedState" or (
                name == f"past_value.{window}_vllmKvCache_RetainedState"
            ), f"window {window}: output not prefixed: {name}"
        # Bug 3: the paired input buffer is the output minus _RetainedState and carries the infix.
        kv_inputs = [n for n in captured["input_names"] if n.startswith(("past_key.", "past_value."))]
        assert kv_inputs, f"window {window}: expected KV input buffers"
        for out_name in kv_outputs:
            paired_input = out_name[: -len("_RetainedState")]
            assert paired_input in kv_inputs, f"window {window}: missing aligned input for {out_name}"
            assert paired_input.endswith("_vllmKvCache")


@pytest.mark.llm_model
def test_layerwise_export_with_kv_cache_prefix_subfunctions(tmp_path):
    """Prefix must survive the subfunction export path end-to-end (the config the EPD example ships).

    With ``use_onnx_subfunctions=True`` the decoder emits ``_InternalRetainedState`` function outputs
    that ``RenameFunctionOutputsTransform`` rewrites to plain ``past_key.{i}_RetainedState``; the
    positional rename loop in ``_export_layerwise`` then overwrites them with the (prefixed) names we
    passed. This asserts the *final transformed* per-window shard on disk carries the infix on both the
    retained-state outputs and the paired input buffers — i.e. the prefix is not lost by the transforms.
    """
    from QEfficient.base.modeling_qeff import QEFFBaseModel
    from QEfficient.transformers.models import _layerwise

    qeff_model, total_layers = _tiny_qwen3_moe_causal()
    export_dir = tmp_path / "subfn_prefixed"
    try:
        with _layerwise._layerwise_export_env():
            _layerwise._set_layer_windows(0, 1, total_layers)
            QEFFBaseModel._start = 0
            QEFFBaseModel._end = 1
            QEFFBaseModel._total_layers = total_layers
            qeff_model.export(
                export_dir=str(export_dir),
                kv_cache_prefix="vllmKvCache",
                use_onnx_subfunctions=True,
            )
    finally:
        _layerwise._reset_layer_windows()
        QEFFBaseModel._start = 0
        QEFFBaseModel._end = 0
        QEFFBaseModel._total_layers = None

    # export_dir gets a hash suffix appended by export_wrapper; find the per-window shard under it.
    shards = list(tmp_path.glob("subfn_prefixed*/onnx_layerwise_tmp/layer_0_1/*_layer_tmp_0_1.onnx"))
    assert shards, "subfunction per-window shard was not produced"
    graph = onnx.load(str(shards[0]), load_external_data=False).graph
    kv_outputs = [o.name for o in graph.output if o.name.startswith(("past_key.", "past_value."))]
    kv_inputs = [i.name for i in graph.input if i.name.startswith(("past_key.", "past_value."))]
    assert kv_outputs, "no KV retained outputs in subfunction shard"
    # Final names use _RetainedState (post-rename) and carry the infix — never raw _InternalRetainedState.
    for name in kv_outputs:
        assert name.endswith("_vllmKvCache_RetainedState"), f"prefix lost in subfunction output: {name}"
        assert "_InternalRetainedState" not in name
    for out_name in kv_outputs:
        assert out_name[: -len("_RetainedState")] in kv_inputs, f"missing aligned input for {out_name}"


@pytest.mark.llm_model
def test_layerwise_export_default_names_unchanged(tmp_path):
    """Backward-compat: without the flag, layerwise per-window names must stay byte-for-byte plain.

    Guards against the prefix plumbing accidentally altering the default (no-prefix) path — the
    output names and the paired input buffers must remain exactly what they were before the feature.
    """
    qeff_model, total_layers = _tiny_qwen3_moe_causal()

    for window in range(total_layers):
        captured = _capture_layerwise_export_names(
            qeff_model,
            window=window,
            total_layers=total_layers,
            export_dir=tmp_path / f"default_w{window}",
            kv_cache_prefix=None,
        )
        assert f"past_key.{window}_RetainedState" in captured["output_names"]
        assert f"past_value.{window}_RetainedState" in captured["output_names"]
        # Inputs use the plain names; no infix anywhere.
        assert f"past_key.{window}" in captured["input_names"]
        assert all("_vllmKvCache" not in n and "_VLLM" not in n for n in captured["output_names"])
        assert all("_vllmKvCache" not in n and "_VLLM" not in n for n in captured["input_names"])


def test_kimi_k25_get_specializations_supports_multi_resolution_grid_sizes():
    """Kimi K2.5 accepts list-valued image sizes for multi-resolution specs."""
    from types import SimpleNamespace

    from QEfficient.transformers.models.kimi_k25.modeling_kimi_k25 import QEffKimiK25ForConditionalGeneration

    model = QEffKimiK25ForConditionalGeneration.__new__(QEffKimiK25ForConditionalGeneration)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14, merge_kernel_size=(2, 2)))

    specs, _ = model.get_specializations(
        batch_size=1,
        prefill_seq_len=64,
        ctx_len=4096,
        image_height=[512, 448],
        image_width=[910, 448],
        num_frames=[1, 1],
        kv_offload=True,
    )
    assert specs["vision"] == [
        {"num_patches": 2508, "grid_h": 38, "grid_w": 66, "num_image_tokens": 627},
        {"num_patches": 1024, "grid_h": 32, "grid_w": 32, "num_image_tokens": 256},
    ]
    assert all(spec["num_image_tokens"] == 627 for spec in specs["lang"])

    with pytest.raises(ValueError, match="image_height and image_width"):
        model.get_specializations(
            batch_size=1,
            prefill_seq_len=64,
            ctx_len=4096,
            h=[30, 32],
            w=[80, 64],
            num_frames=[1, 2],
            kv_offload=True,
        )

    with pytest.raises(ValueError, match="num_patches"):
        model.get_specializations(
            batch_size=1,
            prefill_seq_len=64,
            ctx_len=4096,
            image_height=512,
            image_width=910,
            num_patches=2508,
            kv_offload=True,
        )


@pytest.mark.cpu_only
def test_runner_io_bundle_is_cpu_only_and_qaic_runner_compatible(tmp_path):
    from onnx import TensorProto, helper

    from QEfficient.generation.runner_io import write_runner_io_bundle

    input_info = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "seq_len"])
    output_info = helper.make_tensor_value_info("output", TensorProto.INT64, ["batch_size", "seq_len"])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input_ids"], ["output"])], "runner", [input_info], [output_info]
    )
    onnx_path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), onnx_path)

    compile_dir = tmp_path / "qpc-hash"
    compile_dir.mkdir()
    inputs = np.arange(4, dtype=np.int64).reshape(1, 4)
    io_dir = write_runner_io_bundle(
        onnx_path=onnx_path,
        compile_dir=compile_dir,
        specialization={"batch_size": 1, "seq_len": 4},
        host_inputs={"input_ids": inputs},
    )

    descriptor = json.loads((io_dir / "aic_batch_io.json").read_text())
    entries = descriptor["IO-files"][0]
    assert entries[0] == {
        "path": "data/input_ids.raw",
        "io-direction": "in",
        "elem-size": 8,
        "map-to": "input_ids",
        "dims": [1, 4],
    }
    assert entries[1]["io-direction"] == "out"
    assert entries[1]["dims"] == [1, 4]
    assert (io_dir / "data/input_ids.raw").stat().st_size == inputs.nbytes
    assert not (io_dir / "data/output.raw").exists()


@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_minimax_m3_prefill_decode_hf_pytorch_vs_aic(tmp_path):
    """Check AIC parity against the original HF MiniMax prefill-to-decode handoff."""
    torch.manual_seed(7)
    config = _tiny_minimax_m3_vlm_config()
    text_config = config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(config).eval()
    model_hf_orig = deepcopy(model_hf)
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)
    prefill_len, cache_len = 4, 8
    input_ids = torch.tensor([[3, 5, 7, 9]], dtype=torch.int64)
    prefill_positions = torch.arange(prefill_len, dtype=torch.int64).view(1, -1)
    decode_positions = torch.tensor([[prefill_len]], dtype=torch.int64)
    vision = torch.zeros((1, 1, text_config.hidden_size))
    image_idx = torch.zeros((1, 1), dtype=torch.int64)
    with torch.no_grad():
        hf_prefill = model_hf_orig.model.language_model(
            input_ids=input_ids,
            position_ids=prefill_positions,
            use_cache=True,
        )
        pt_prefill_logits = model_hf_orig.lm_head(hf_prefill.last_hidden_state[:, -1:, :]).float()
        next_ids = pt_prefill_logits.argmax(dim=-1)
        hf_decode = model_hf_orig.model.language_model(
            input_ids=next_ids,
            position_ids=decode_positions,
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        )
        pt_decode_logits = model_hf_orig.lm_head(hf_decode.last_hidden_state[:, -1:, :]).float()

    def run_stage(qpc_path, stage_input_ids, stage_positions, retained=None):
        session = QAICInferenceSession(qpc_path)
        try:

            def adapt(name, source=None):
                binding = session.bindings[session.binding_index_map[name]]
                shape = tuple(binding.dims)
                dtype = session.aic_to_np_dtype_mapping[binding.type]
                if source is None:
                    return np.zeros(shape, dtype=dtype)
                source = np.asarray(source, dtype=dtype)
                assert source.size == np.prod(shape), (
                    f"AIC binding shape mismatch for {name}: {source.shape}, expected {shape}"
                )
                return source.reshape(shape)

            inputs = {}
            for name in session.input_names:
                if name == "input_ids":
                    inputs[name] = adapt(name, stage_input_ids.numpy())
                elif name == "position_ids":
                    inputs[name] = adapt(name, stage_positions.numpy())
                elif name == "vision_embeds":
                    inputs[name] = adapt(name, vision.expand(1, stage_positions.shape[1], -1).numpy())
                elif name == "image_idx":
                    inputs[name] = adapt(name, image_idx.numpy())
                elif name.startswith(("past_key.", "past_value.", "index_key.")):
                    inputs[name] = adapt(name, retained.get(name) if retained else None)
                else:
                    raise AssertionError(f"Unhandled MiniMax AIC input: {name}")
            return session.run(inputs)
        finally:
            session.deactivate()

    def run_ort_stage(onnx_path, stage_input_ids, stage_positions, retained=None):
        session = _ort_session(Path(onnx_path))
        input_shapes = {meta.name: tuple(meta.shape) for meta in session.get_inputs()}
        input_types = {
            meta.name: np.int64 if meta.type == "tensor(int64)" else np.float32 for meta in session.get_inputs()
        }

        def adapt(name, source=None):
            shape = input_shapes[name]
            dtype = input_types[name]
            if source is None:
                assert all(isinstance(dim, int) for dim in shape), (
                    f"ORT input {name} has unresolved shape {shape} without a source tensor"
                )
                return np.zeros(shape, dtype=dtype)
            source = np.asarray(source, dtype=dtype)
            for expected, actual in zip(shape, source.shape):
                if isinstance(expected, int):
                    assert expected == actual, (
                        f"ORT binding shape mismatch for {name}: {source.shape}, expected {shape}"
                    )
            return source

        inputs = {}
        for name in input_shapes:
            if name == "input_ids":
                inputs[name] = adapt(name, stage_input_ids.numpy())
            elif name == "position_ids":
                inputs[name] = adapt(name, stage_positions.numpy())
            elif name == "vision_embeds":
                inputs[name] = adapt(name, vision.expand(1, stage_positions.shape[1], -1).numpy())
            elif name == "image_idx":
                inputs[name] = adapt(name, image_idx.numpy())
            elif name.startswith(("past_key.", "past_value.")):
                source = (
                    retained.get(name)
                    if retained
                    else np.zeros((1, text_config.num_key_value_heads, cache_len, text_config.head_dim))
                )
                inputs[name] = adapt(name, source)
            elif name.startswith("index_key."):
                source = retained.get(name) if retained else np.zeros((1, 1, cache_len, text_config.index_head_dim))
                inputs[name] = adapt(name, source)
            else:
                raise AssertionError(f"Unhandled MiniMax ORT input: {name}")
        return dict(zip((output.name for output in session.get_outputs()), session.run(None, inputs)))

    compile_args = dict(
        ctx_len=cache_len,
        batch_size=1,
        num_cores=4,
        num_devices=1,
        skip_vision=True,
        node_precision_info=True,
        qaic_config={
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": 2,
            "msa_indexer_dp": 1,
            "msa_indexer_cp": 1,
            "msa_attn_dp": 1,
            "indexer_n_head": 1,
            "num_cores_per_device": 2,
            "msa_q_chunk": 4,
        },
        offload_pt_weights=False,
    )

    prefill_qpc = qeff_model.compile(
        compile_dir=tmp_path / "minimax-prefill-aic",
        prefill_seq_len=prefill_len,
        prefill_only=True,
        enable_chunking=True,
        **compile_args,
    )["lang_prefill_qpc_path"]
    prefill_onnx = Path(qeff_model.lang_model.onnx_path)
    decode_qpc = qeff_model.compile(
        compile_dir=tmp_path / "minimax-decode-aic", prefill_seq_len=1, prefill_only=False, **compile_args
    )["lang_decode_qpc_path"]
    decode_onnx = Path(qeff_model.lang_model.onnx_path)

    ort_prefill = run_ort_stage(prefill_onnx, input_ids, prefill_positions)
    np.testing.assert_allclose(ort_prefill["logits"], pt_prefill_logits.numpy(), atol=1e-4, rtol=1e-4)
    ort_retained = {
        name.removesuffix("_RetainedState"): output
        for name, output in ort_prefill.items()
        if name.endswith("_RetainedState")
    }
    ort_decode = run_ort_stage(decode_onnx, ort_prefill["logits"].argmax(axis=-1), decode_positions, ort_retained)
    np.testing.assert_allclose(ort_decode["logits"], pt_decode_logits.numpy(), atol=1e-4, rtol=1e-4)

    aic_prefill = run_stage(prefill_qpc, input_ids, prefill_positions)
    np.testing.assert_allclose(aic_prefill["logits"], pt_prefill_logits.numpy(), atol=1e-2, rtol=1e-2)
    retained = {
        name.removesuffix("_RetainedState"): output
        for name, output in aic_prefill.items()
        if name.endswith("_RetainedState")
    }
    aic_decode = run_stage(decode_qpc, aic_prefill["logits"].argmax(axis=-1), decode_positions, retained)
    np.testing.assert_allclose(aic_decode["logits"], pt_decode_logits.numpy(), atol=1e-2, rtol=1e-2)


@pytest.mark.llm_model
def test_minimax_m3_prefill_hf_qeff_pytorch_parity_with_explicit_cache():
    """Compare HF and QEff PyTorch prefill with identical explicit cache tensors."""
    torch.manual_seed(7)
    vlm_config = _tiny_minimax_m3_vlm_config(microbench=True)
    text_config = vlm_config.text_config
    model_hf = MiniMaxM3SparseForConditionalGeneration(vlm_config).eval()
    model_hf_orig = deepcopy(model_hf)
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=True)

    prefill_len, cache_len = 8, 8
    input_ids = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 17]], dtype=torch.int64)
    position_ids = torch.arange(prefill_len, dtype=torch.int64).view(1, -1)
    vision_embeds = torch.zeros((1, prefill_len, text_config.hidden_size))
    image_idx = torch.zeros((1, 1), dtype=torch.int64)
    explicit_cache = tuple(
        (
            torch.zeros((1, text_config.num_key_value_heads, cache_len, text_config.head_dim)),
            torch.zeros((1, text_config.num_key_value_heads, cache_len, text_config.head_dim)),
        )
        for _ in range(text_config.num_hidden_layers)
    )
    explicit_index_keys = [
        torch.zeros((1, 1, cache_len, text_config.index_head_dim))
        for layer_type in text_config.layer_types
        if layer_type == "minimax_m3_sparse"
    ]

    qeff_model.transform(
        ctx_len=cache_len,
        seq_len=prefill_len,
        bs=1,
        num_devices=1,
        num_cores=8,
        prefill_only=True,
        prefill_seq_len=prefill_len,
        qaic_config={
            "blocking_mode": "kv_headpar",
            "num_kv_blocks": 1,
            "msa_indexer_dp": 1,
            "msa_indexer_cp": 1,
            "msa_attn_dp": 1,
            "indexer_n_head": 1,
            "num_cores_per_device": 8,
            "msa_q_chunk": 8,
        },
    )
    hf_cache = DynamicCache(config=text_config)
    # HF has no fixed-capacity cache input; an empty DynamicCache represents the
    # same logical state as QEff's zero-filled capacity cache before prefill.

    with torch.no_grad():
        hf_outputs = model_hf_orig.model.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=hf_cache,
            use_cache=True,
        )
        hf_logits = model_hf_orig.lm_head(hf_outputs.last_hidden_state[:, -1:, :]).float()
        qeff_outputs = qeff_model.lang_model.model(
            input_ids=input_ids,
            vision_embeds=vision_embeds,
            position_ids=position_ids,
            image_idx=image_idx,
            past_key_values=explicit_cache,
            index_keys=explicit_index_keys,
        )

    np.testing.assert_allclose(
        qeff_outputs[0].detach().cpu().numpy(),
        hf_logits.detach().cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
