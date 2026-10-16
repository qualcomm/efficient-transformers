# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Pre-merge reproducer configs for reported QEfficient regressions.

Flow
----
1. Each ``RegressionScenario`` is a self-contained bug-report reproducer entry.
   The scenario entry, not any external Markdown format, is the source of truth.
2. Developers add the exact reproducer config before fixing a bug, then run
   the same config after the fix before merge.
3. Active scenarios run the real QEfficient software stage named by ``stage``:
   import/install, model download, model export, model compile, or inference.
4. Prefer tiny-random or 2/4-layer reductions of the same architecture. If a
   reduced model cannot reproduce the bug, keep the official model card and run
   it with ``QEFF_REPRODUCER_RUN_FULL_MODELS=1``.
5. Scenarios that require another machine's paths, private artifacts, vLLM
   services, runtime generation, or full-memory measurement are explicit skips.
   They remain visible in collection and cannot be mistaken for passing verdicts.
6. To add a future report, copy the commented template in ``SCENARIOS``, paste
   exact reporter options into ``config``, choose the correct ``model_api``, and
   implement the real stage assertion in the helper for that API.
7. To run one scenario during fix validation, set
   ``QEFF_REPRODUCER_SCENARIO=<scenario-name>`` before pytest.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
import torch

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux CI has fcntl.
    fcntl = None

pytestmark = pytest.mark.reproducer

Stage = Literal["import_install", "download", "export", "compile", "inference"]
ModelAPI = Literal["causal_lm", "image_text_to_text", "wan_t2v", "wan_i2v", "replicate_kv_heads", "embedding"]


# Scenario entries are immutable so pytest parametrization cannot mutate reporter-derived options.
@dataclass(frozen=True)
class RegressionScenario:
    """Store one reported regression with its exact stage and reduced test model.

    This class exists to keep model card, tiny checkpoint, failing options,
    and explicit deferral reason together as a single auditable test record.
    """

    name: str
    stage: Stage
    source_model_card: str
    tiny_model_id: str | None
    summary: str
    model_api: ModelAPI = "causal_lm"
    official_model_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    export_kwargs: dict[str, Any] = field(default_factory=dict)
    compile_kwargs: dict[str, Any] = field(default_factory=dict)
    inference_kwargs: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    follow_up: str | None = None
    run_full_model_by_default: bool = False


TINY_GPT2 = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"
TINY_FALCON = "hf-internal-testing/tiny-random-FalconForCausalLM"
TINY_MIXTRAL = "hf-internal-testing/tiny-random-MixtralForCausalLM"
TINY_BERT = "hf-internal-testing/tiny-random-BertModel"
TINY_GPT_OSS = "tiny-random/gpt-oss-bf16"
TINY_GEMMA3 = "tiny-random/gemma-3"
TINY_QWEN2_5_VL = "optimum-intel-internal-testing/tiny-random-qwen2.5-vl"
TINY_INTERNVL = "optimum-intel-internal-testing/tiny-random-internvl2"
TINY_QWEN3_MOE = "tiny-random/qwen3-moe"
TINY_QWEN3_5_MOE = "tiny-random/qwen3.5-moe"
TINY_QWEN3_VL_MOE = "tiny-random/qwen3-vl-moe"
FULL_MODEL_ENV = "QEFF_REPRODUCER_RUN_FULL_MODELS"
OFFICIAL_MODEL_ENV = "QEFF_REPRODUCER_USE_OFFICIAL_MODELS"
REPORT_PATH_ENV = "QEFF_REPRODUCER_REPORT_MD"
SCENARIO_FILTER_ENV = "QEFF_REPRODUCER_SCENARIO"
DEFAULT_REPORT_PATH = Path("tests/reproducer_configs/reproducer_config_results.md")
EXPECTED_SCENARIO_COUNT = 52
EXTRA_QEFF_COMPILE_OPTIONS = frozenset(
    {
        "height",
        "mdp_compiler_dump_path",
        "mdp_dump_partition_config",
        "mdp_load_partition_config",
        "mdp_num_partitions",
        "mdp_strategy",
        "mdp_ts_num_devices",
        "mm_processor_kwargs",
        "node_precision_info",
        "num_frames",
        "parallel",
        # These QAIC flags are forwarded through the VLM compile API's
        # ``**compiler_options`` and therefore do not appear in its signature.
        "split_model_io",
        "user_tiled",
        "vision_size",
        "width",
    }
)
COMPILER_ERROR_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^Operator-",
        r"QAIC_ERROR",
        r"Error message:",
        r"Unable to Compile",
        r"Compilation failed",
        r"Non-constant",
        r"not supported",
    )
)


# Generated reports should not appear for pytest --collect-only runs.
def _is_pytest_collect_only() -> bool:
    """Return whether pytest is collecting tests without executing them.

    The helper prevents collection-only invocations from creating a misleading
    partial Markdown verdict report before any scenario has actually run.
    """
    return "--collect-only" in sys.argv or "--co" in sys.argv


# Lock report writes so xdist workers cannot corrupt the runtime ledger.
@contextlib.contextmanager
def _locked_report(path: Path):
    """Yield a locked report file handle for atomic read/append operations.

    The helper exists because pytest-xdist workers can record scenario verdicts
    concurrently, and the generated Markdown report should remain readable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# Tiny model IDs mirror the quickcheck suite and keep reports cheap to rerun.
# Official model IDs are retained for bugs that cannot be reproduced faithfully
# with a reduced checkpoint. They run only when FULL_MODEL_ENV is set.
SCENARIOS: tuple[RegressionScenario, ...] = (
    # Template for future bug reports; keep commented until a real report is added.
    # RegressionScenario(
    #     name="short-neutral-slug",
    #     stage="compile",
    #     source_model_card="org/reported-model-card",
    #     tiny_model_id="hf-internal-testing/tiny-random-compatible-model",
    #     official_model_id="org/reported-model-card",
    #     summary="one-line failure mode",
    #     model_api="causal_lm",
    #     compile_kwargs={"copy_exact_reporter_flag": "copy_exact_reporter_value"},
    #     skip_reason="only if exact repro requires external artifacts or runtime services",
    # ),
    RegressionScenario(
        name="hf-transfer-import-order",
        stage="import_install",
        source_model_card="large gated Hugging Face checkpoints",
        tiny_model_id=None,
        summary="hf_transfer must be enabled before transformers import",
    ),
    RegressionScenario(
        name="cloud-export-hf-token",
        stage="download",
        source_model_card="mistralai/Mistral-7B-v0.1",
        tiny_model_id=TINY_MISTRAL,
        summary="cloud export token path used an external cache directory in the report",
        config={"hf_token": "<reported_hf_token>", "full_batch_size": 2},
        skip_reason="reported config depends on an absolute cache path from another machine",
    ),
    RegressionScenario(
        name="vlm-cli-config-dump",
        stage="export",
        source_model_card="VLM CLI/config dump",
        tiny_model_id=TINY_QWEN2_5_VL,
        summary="report did not include a concrete failing command or config payload",
        skip_reason="exact reporter options are not available",
    ),
    RegressionScenario(
        name="internvl25-decoder-language-model",
        stage="export",
        source_model_card="OpenGVLab/InternVL2_5-1B",
        tiny_model_id=TINY_INTERNVL,
        summary="InternVL language decoder wrapper export must expose language_model",
        load_kwargs={"trust_remote_code": True},
        export_kwargs={"kv_offload": True},
        skip_reason="tiny InternVL export requires optional einops and timm dependencies",
    ),
    RegressionScenario(
        name="internvl-dual-qpc-custom-num-patches",
        stage="inference",
        source_model_card="OpenGVLab/InternVL2_5-1B",
        tiny_model_id=TINY_INTERNVL,
        summary="dual-QPC custom num_patches=9 failed during AIC generation",
        config={"kv_offload": True, "num_patches": 9, "generation_len": 128},
        skip_reason="verdict requires QAIC runtime generation with dual QPCs",
    ),
    RegressionScenario(
        name="internvl-large-shape-mismatch",
        stage="inference",
        source_model_card="OpenGVLab/InternVL2_5-38B or OpenGVLab/InternVL2_5-78B",
        tiny_model_id=TINY_INTERNVL,
        summary="large InternVL shape mismatch report lacks exact compile/inference options",
        skip_reason="exact reporter options are not available",
    ),
    RegressionScenario(
        name="internvl25-dual-qpc-input-shape",
        stage="inference",
        source_model_card="OpenGVLab/InternVL2_5 family",
        tiny_model_id=TINY_INTERNVL,
        summary="dual-QPC input shape mismatch requires reported image/runtime setup",
        skip_reason="verdict requires QAIC runtime generation with reported image input",
    ),
    RegressionScenario(
        name="attention-mask-negative-infinity",
        stage="export",
        source_model_card="meta-llama/Llama-3.1-8B-Instruct",
        tiny_model_id=TINY_LLAMA,
        summary="attention masking export must use negative infinity semantics",
        export_kwargs={},
    ),
    RegressionScenario(
        name="swiftkv-compile",
        stage="compile",
        source_model_card="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
        tiny_model_id=None,
        official_model_id="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
        summary="SwiftKV cloud.infer compile options must produce a QPC",
        compile_kwargs={
            "prefill_seq_len": 2048,
            "ctx_len": 3072,
            "batch_size": 1,
            "num_devices": 4,
            "num_cores": 16,
            "full_batch_size": 1,
            "mxfp6_matmul": True,
            "mxint8_kv_cache": True,
            "aic_enable_depth_first": True,
            "allow_mxint8_mdp_io": True,
        },
    ),
    RegressionScenario(
        name="kv-head-replication-num-heads",
        stage="export",
        source_model_card="meta-llama/Llama-3.1-8B-Instruct",
        tiny_model_id=TINY_LLAMA,
        summary="replicate_kv_heads script is absent from this checkout",
        skip_reason="reported script entry point is not present in this repository",
    ),
    RegressionScenario(
        name="mllama-dual-qpc-model-attr",
        stage="compile",
        source_model_card="meta-llama/Llama-3.2-11B-Vision-Instruct",
        tiny_model_id=None,
        official_model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        summary="Mllama dual-QPC VLM compile must expose language_model config",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True, "num_hidden_layers": 2},
        compile_kwargs={
            "prefill_seq_len": 32,
            "ctx_len": 512,
            "img_size": 560,
            "batch_size": 1,
            "num_devices": 1,
            "num_cores": 16,
        },
    ),
    RegressionScenario(
        name="turbolora-export-hash",
        stage="export",
        source_model_card="TurboLoRA Llama path",
        tiny_model_id=TINY_LLAMA,
        summary="TurboLoRA needs PEFT/speculator fixtures",
        skip_reason="speculator and adapter artifacts are not available",
    ),
    RegressionScenario(
        name="external-mdp-load-config-duplicate",
        stage="compile",
        source_model_card="openai/gpt-oss-20b",
        tiny_model_id=None,
        official_model_id="openai/gpt-oss-20b",
        summary="external mdp_load_partition_config appeared twice in compile command",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 2,
            "num_cores": 8,
            "mxfp6_matmul": True,
            "mdp_num_partitions": 2,
            "mdp_strategy": "onnx",
        },
    ),
    RegressionScenario(
        name="llama4-dynamic-cache-export",
        stage="compile",
        source_model_card="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        tiny_model_id=None,
        official_model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        summary="Llama4 dynamic-cache VLM compile must preserve CtxGatherND path",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True, "num_hidden_layers": 4},
        compile_kwargs={
            "prefill_seq_len": 32,
            "ctx_len": 3072,
            "img_size": 336,
            "batch_size": 1,
            "num_devices": 4,
            "num_cores": 8,
        },
    ),
    RegressionScenario(
        name="swiftkv-replicate-kv-heads",
        stage="export",
        source_model_card="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
        tiny_model_id=None,
        official_model_id="Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
        summary="SwiftKV replicate_kv_heads export must accept attention_mask",
        model_api="replicate_kv_heads",
        export_kwargs={"prompt": "Hello, world!", "repeat": 2, "full_batch_size": 1, "num_hidden_layers": 2},
    ),
    RegressionScenario(
        name="falcon-input-layernorm-export",
        stage="export",
        source_model_card="tiiuae/falcon-40b",
        tiny_model_id=TINY_FALCON,
        summary="Falcon export failed because decoder layer lacked input_layernorm",
        export_kwargs={},
    ),
    RegressionScenario(
        name="slow-downloads-hf-transfer",
        stage="import_install",
        source_model_card="large Hugging Face checkpoints",
        tiny_model_id=None,
        summary="hf_transfer enablement must happen before HF imports",
    ),
    RegressionScenario(
        name="finetune-dataset-padding",
        stage="inference",
        source_model_card="fine-tuning datasets",
        tiny_model_id=None,
        summary="fine-tune DDP dataset padding belongs in fine-tune suite",
        skip_reason="not a QEff export/compile/inference model-stage reproducer",
    ),
    RegressionScenario(
        name="qwen-awq-replicate-kv-bias-scaling",
        stage="export",
        source_model_card="Qwen/Qwen2.5-72B-Instruct-AWQ",
        tiny_model_id=None,
        summary="report has no captured reproducer details",
        skip_reason="exact reporter options are not available",
    ),
    RegressionScenario(
        name="gpt-oss-ts2-cpl128-cl4096-compile",
        stage="compile",
        source_model_card="openai/gpt-oss-20b",
        tiny_model_id=TINY_GPT_OSS,
        summary="GPT-OSS requested TS2, CPL128, CL4096, cores8, MXFP6 compile must produce QPC",
        load_kwargs={"trust_remote_code": True},
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 2,
            "num_cores": 8,
            "mxfp6_matmul": True,
        },
    ),
    RegressionScenario(
        name="gemma3-large-image-noisy-output",
        stage="inference",
        source_model_card="google/gemma-3-27b-it",
        tiny_model_id=TINY_GEMMA3,
        summary="Gemma3 noisy output report depends on attached OCR image and prompt",
        skip_reason="attached image and expected text are not available in this repository",
    ),
    RegressionScenario(
        name="wan22-n-layer-cache-hash",
        stage="compile",
        source_model_card="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        tiny_model_id=None,
        official_model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        summary="WAN2.2 n-layer cache hash must produce deterministic QPC paths",
        model_api="wan_t2v",
        compile_kwargs={"height": 192, "width": 320, "num_frames": 9, "parallel": False},
    ),
    RegressionScenario(
        name="disagg-prefill-continuous-batching",
        stage="compile",
        source_model_card="meta-llama/Llama-3.3-70B-Instruct",
        tiny_model_id=None,
        official_model_id="meta-llama/Llama-3.3-70B-Instruct",
        summary="disagg prefill compile used full_batch_size with external MDP JSON",
        compile_kwargs={
            "prefill_seq_len": 256,
            "ctx_len": 11264,
            "batch_size": 1,
            "full_batch_size": 16,
            "num_cores": 16,
            "num_devices": 16,
            "mxfp6_matmul": True,
            "mxint8_kv_cache": True,
            "prefill_only": True,
            "aic_enable_depth_first": True,
            "allow_mxint8_mdp_io": True,
            "mdp_num_partitions": 4,
            "mdp_strategy": "onnx",
        },
    ),
    RegressionScenario(
        name="ccl-decode-fallback-context",
        stage="compile",
        source_model_card="meta-llama/Llama-3.2-1B",
        tiny_model_id=TINY_LLAMA,
        summary="decode CCL [4096] must include CL8192 fallback specialization",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 8192,
            "num_cores": 16,
            "num_devices": 1,
            "mxint8_kv_cache": True,
            "mxfp6_matmul": True,
            "batch_size": 1,
            "comp_ctx_lengths_prefill": [4000],
            "comp_ctx_lengths_decode": [4096],
        },
    ),
    RegressionScenario(
        name="ccl-none-prefill-with-decode",
        stage="compile",
        source_model_card="meta-llama/Llama-3.2-1B",
        tiny_model_id=TINY_LLAMA,
        summary="None prefill CCL and configured decode CCL must compile",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 8192,
            "num_cores": 16,
            "num_devices": 1,
            "mxint8_kv_cache": True,
            "mxfp6_matmul": True,
            "batch_size": 1,
            "comp_ctx_lengths_prefill": None,
            "comp_ctx_lengths_decode": [4096],
        },
    ),
    RegressionScenario(
        name="ccl-empty-prefill-with-decode",
        stage="compile",
        source_model_card="meta-llama/Llama-3.2-1B",
        tiny_model_id=TINY_LLAMA,
        summary="empty prefill CCL and configured decode CCL must compile",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 8192,
            "num_cores": 16,
            "num_devices": 1,
            "mxint8_kv_cache": True,
            "mxfp6_matmul": True,
            "batch_size": 1,
            "comp_ctx_lengths_prefill": [],
            "comp_ctx_lengths_decode": [4096, 8192],
        },
    ),
    RegressionScenario(
        name="mdp-dump-load-json-options",
        stage="compile",
        source_model_card="openai/gpt-oss-20b",
        tiny_model_id=None,
        official_model_id="openai/gpt-oss-20b",
        summary="custom mdp dump/load config cases must work with local generated JSON artifacts",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 2,
            "num_cores": 8,
            "mxfp6_matmul": True,
            "_mdp_intersection": True,
            "mdp_num_partitions": 2,
        },
    ),
    RegressionScenario(
        name="qwen25-vl-subfunction-split-compile",
        stage="compile",
        source_model_card="Qwen/Qwen2.5-VL-32B-Instruct",
        tiny_model_id=TINY_QWEN2_5_VL,
        official_model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        summary="Qwen2.5-VL subfunction compiler failure needs QAIC graph validation",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True},
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "img_size": 1540,
            "batch_size": 1,
            "num_devices": 1,
            "num_cores": 16,
            "use_onnx_subfunctions": True,
        },
    ),
    RegressionScenario(
        name="pipeline-prefill-mdp-dump-no-load",
        stage="compile",
        source_model_card="openai/gpt-oss-20b",
        tiny_model_id=None,
        official_model_id="openai/gpt-oss-20b",
        summary="pipeline prefill with mdp dump must generate local compiler dump and use repo NPI",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 4,
            "num_cores": 16,
            "mxfp6_matmul": True,
            "mxint8_kv_cache": True,
            "prefill_only": True,
            "use_onnx_subfunctions": True,
            "node_precision_info": "examples/disagg_serving/subfunction_120b_npi.yaml",
            "_mdp_intersection": True,
            "mdp_num_partitions": 2,
        },
    ),
    RegressionScenario(
        name="mdp-ts-files-in-qpc-dir",
        stage="compile",
        source_model_card="meta-llama/Llama-3.1-8B-Instruct",
        tiny_model_id=TINY_LLAMA,
        summary="generated mdp_ts artifacts should be owned by compile/QPC path",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 1024,
            "num_cores": 16,
            "num_devices": 4,
            "mxint8_kv_cache": True,
            "mxfp6_matmul": True,
            "batch_size": 1,
            "full_batch_size": 4,
            "mos": 1,
            "aic_enable_depth_first": True,
            "comp_ctx_lengths_prefill": [4000],
            "comp_ctx_lengths_decode": [4096],
        },
    ),
    RegressionScenario(
        name="ccl-default-enabled",
        stage="compile",
        source_model_card="CCL-enabled causal LM",
        tiny_model_id=TINY_LLAMA,
        summary="default CCL-enabled causal LM compile must succeed",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 8192,
            "num_cores": 16,
            "num_devices": 1,
            "mxint8_kv_cache": True,
            "mxfp6_matmul": True,
            "batch_size": 1,
        },
    ),
    RegressionScenario(
        name="wan-skip-compile-existing-qpc",
        stage="compile",
        source_model_card="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        tiny_model_id=None,
        official_model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        summary="WAN compile must honor existing QPC path skip behavior",
        model_api="wan_t2v",
        compile_kwargs={"height": 192, "width": 320, "num_frames": 9, "parallel": False},
    ),
    RegressionScenario(
        name="qwen3-mixtral-subfunction-reducesum",
        stage="compile",
        source_model_card="Qwen/Qwen3-30B-A3B-Instruct and Mixtral-8x7B",
        tiny_model_id=None,
        official_model_id="Qwen/Qwen3-30B-A3B-Instruct",
        summary="compiler ReduceSum failures require full ONNX graph validation",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 4,
            "num_cores": 16,
            "mxfp6_matmul": True,
            "use_onnx_subfunctions": True,
        },
    ),
    RegressionScenario(
        name="disagg-ccl-decode-gpt-oss",
        stage="compile",
        source_model_card="openai/gpt-oss-20b",
        tiny_model_id=TINY_GPT_OSS,
        summary="vLLM disagg CCL decode compile options must produce decode QPC",
        compile_kwargs={
            "prefill_seq_len": 1,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 2,
            "num_cores": 8,
            "mxfp6_matmul": True,
            "comp_ctx_lengths_decode": [4096],
        },
    ),
    RegressionScenario(
        name="qwen25-vl-subfunction-internal-assert",
        stage="compile",
        source_model_card="Qwen/Qwen2.5-VL-32B-Instruct",
        tiny_model_id=TINY_QWEN2_5_VL,
        official_model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        summary="subfunction internal compiler assert requires QAIC graph validation",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True},
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "img_size": 1540,
            "batch_size": 1,
            "num_devices": 1,
            "num_cores": 16,
            "use_onnx_subfunctions": True,
        },
    ),
    RegressionScenario(
        name="qwen25-vl-dual-qpc-ods-vocab-size",
        stage="export",
        source_model_card="Qwen/Qwen2.5-VL-3B-Instruct",
        tiny_model_id=TINY_QWEN2_5_VL,
        summary="dual-QPC on-device sampler export must read vocab_size from the decoder wrapper",
        model_api="image_text_to_text",
        load_kwargs={
            "continuous_batching": True,
            "qaic_config": {
                "include_sampler": True,
                "return_pdfs": False,
                "max_top_k_ids": 512,
            },
            "trust_remote_code": False,
            "attn_implementation": "eager",
            "kv_offload": True,
        },
        export_kwargs={"prefill_seq_len": 128},
    ),
    RegressionScenario(
        name="wan22-excessive-prints",
        stage="compile",
        source_model_card="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        tiny_model_id=None,
        official_model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        summary="WAN2.2 export/compile must complete without excessive graph prints",
        model_api="wan_t2v",
        compile_kwargs={"height": 192, "width": 320, "num_frames": 9, "parallel": False},
    ),
    RegressionScenario(
        name="embedding-model-compile",
        stage="compile",
        source_model_card="BAAI/bge-reranker-v2-m3 and intfloat/multilingual-e5-large",
        tiny_model_id=TINY_BERT,
        summary="embedding compile failure requires sequence/reranker compile path",
        model_api="embedding",
        compile_kwargs={"seq_len": 128, "batch_size": 1, "num_devices": 1, "num_cores": 16},
    ),
    RegressionScenario(
        name="qwen3-vl-smart-resize-list-inputs",
        stage="compile",
        source_model_card="Qwen/Qwen3-VL-32B-Instruct",
        tiny_model_id=None,
        official_model_id="Qwen/Qwen3-VL-2B-Instruct",
        summary="Qwen3-VL list height/width smart_resize failed before compile",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True, "num_hidden_layers": 1},
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "img_size": 1540,
            "batch_size": 1,
            "num_devices": 1,
            "num_cores": 16,
        },
    ),
    RegressionScenario(
        name="gemma4-dense-2qpc-cb2",
        stage="compile",
        source_model_card="google/gemma-4-E2B-it",
        tiny_model_id="tiny-random/gemma-4-dense",
        official_model_id="google/gemma-4-E2B-it",
        summary="Gemma4 dense 2-QPC compile with decode batch size >1",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True},
        compile_kwargs={
            "prefill_seq_len": 64,
            "ctx_len": 512,
            "img_size": 384,
            "batch_size": 1,
            "full_batch_size": 2,
            "num_devices": 1,
            "num_cores": 16,
        },
    ),
    RegressionScenario(
        name="qwen36-subfunction-custom-io",
        stage="compile",
        source_model_card="Qwen/Qwen3.6-35B-A3B",
        tiny_model_id=None,
        official_model_id="Qwen/Qwen3.6-35B-A3B",
        summary="Qwen3.6 subfunction custom_io content mismatch",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 1,
            "num_cores": 16,
            "use_onnx_subfunctions": True,
        },
    ),
    RegressionScenario(
        name="qwen3vl-custom-io-subfunctions",
        stage="compile",
        source_model_card="Qwen/Qwen3-VL-30B-A3B-Instruct",
        tiny_model_id=TINY_QWEN3_VL_MOE,
        official_model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        summary="Qwen3-VL subfunction custom_io.yaml invalid entries",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True},
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "img_size": 1540,
            "batch_size": 1,
            "num_devices": 1,
            "num_cores": 16,
            "use_onnx_subfunctions": True,
        },
    ),
    RegressionScenario(
        name="qwen3vl-moe-batch-fold-subfunction",
        stage="compile",
        source_model_card="Qwen/Qwen3-VL-235B-A22B-Instruct",
        tiny_model_id=TINY_QWEN3_VL_MOE,
        official_model_id="Qwen/Qwen3-VL-235B-A22B-Instruct",
        summary="batch-folded decode range ops must remain inside Qwen3-VL-MoE decoder subfunctions",
        model_api="image_text_to_text",
        load_kwargs={
            "attn_implementation": "eager",
            "kv_offload": True,
            "continuous_batching": True,
            "dtype": torch.float16,
            "num_hidden_layers": 4,
            "layerwise": False,
        },
        compile_kwargs={
            "batch_size": 256,
            "full_batch_size": 256,
            "kv_cache_batch_size": 256,
            "prefill_seq_len": 1,
            "ctx_len": 10240,
            "height": 354,
            "width": 536,
            "num_cores": 16,
            "num_devices": 16,
            "mxfp6_matmul": True,
            "mxint8_kv_cache": True,
            "split_model_io": True,
            "user_tiled": True,
            "prefill_only": False,
            "skip_vision": True,
            "use_onnx_subfunctions": True,
            "layerwise": False,
            "offload_pt_weights": False,
            "qaic_config": {
                "blocking_mode": "kv_batch_fold",
                "num_kv_blocks": 4,
                "ctx_len": 10240,
                "moe_config": {
                    "flavour": "expert_parallel",
                    "tree_reduce": False,
                    "cores_per_expert": 2,
                    "expert_parallel_chunk_size": 128,
                },
            },
        },
    ),
    RegressionScenario(
        name="qwen35-layerwise-no-decoder-artifacts",
        stage="export",
        source_model_card="Qwen3.5-397B-A17B",
        tiny_model_id=TINY_QWEN3_5_MOE,
        summary="layerwise 3-QPC export ran for many hours without decoder artifacts",
        skip_reason="long-running layerwise export scenario requires full model artifacts",
    ),
    RegressionScenario(
        name="gptoss-mdp-num-partitions-file",
        stage="compile",
        source_model_card="openai/gpt-oss-20b",
        tiny_model_id=None,
        official_model_id="openai/gpt-oss-20b",
        summary="GPT-OSS mdp_num_partitions should generate partition file",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 2,
            "num_cores": 8,
            "mxfp6_matmul": True,
            "mdp_num_partitions": 2,
            "mdp_strategy": "onnx",
        },
    ),
    RegressionScenario(
        name="qwen35-moe-multires-support",
        stage="compile",
        source_model_card="Qwen/Qwen3.5-35B-A3B",
        tiny_model_id=None,
        official_model_id="Qwen/Qwen3.5-35B-A3B",
        summary="multi-resolution and vision_size support for Qwen3.5 MoE",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True},
        compile_kwargs={
            "prefill_seq_len": 64,
            "ctx_len": 4096,
            "img_size": 1540,
            "batch_size": 1,
            "num_devices": 1,
            "num_cores": 16,
        },
    ),
    RegressionScenario(
        name="gptoss-gemma3-prefill-reducesum",
        stage="compile",
        source_model_card="openai/gpt-oss-20b and Gemma3",
        tiny_model_id=None,
        official_model_id="openai/gpt-oss-20b",
        summary="prefill compiler ReduceSum failure requires reported full ONNX graph",
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "batch_size": 1,
            "num_devices": 4,
            "num_cores": 16,
            "mxfp6_matmul": True,
            "prefill_only": True,
        },
    ),
    RegressionScenario(
        name="qwen3vl-moe-memory-after-prefill",
        stage="export",
        source_model_card="Qwen/Qwen3-VL-235B-A22B-Instruct",
        tiny_model_id=TINY_QWEN3_VL_MOE,
        summary="decode export RAM spike after prefill in same process",
        skip_reason="full-model memory regression requires dedicated memory measurement job",
    ),
    RegressionScenario(
        name="mdp-generation-example-script",
        stage="compile",
        source_model_card="Qwen/Qwen3-VL-30B-A3B-Instruct",
        tiny_model_id=None,
        official_model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        summary="Qwen3-VL MDP example must compile prefill without duplicate device options",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True, "layerwise": False},
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "height": 354,
            "width": 536,
            "batch_size": 1,
            "num_devices": 4,
            "num_cores": 16,
            "mos": 1,
            "mxfp6_matmul": True,
            "mxint8_kv_cache": True,
            "retain_full_kv": True,
            "prefill_only": True,
            "enable_chunking": True,
            "skip_vision": True,
            "use_onnx_subfunctions": True,
            "mdp_num_partitions": 2,
            "mdp_strategy": "onnx",
        },
    ),
    RegressionScenario(
        name="qwen-numpy-fix",
        stage="export",
        source_model_card="Qwen models",
        tiny_model_id=None,
        summary="report lacks concrete command/options",
        skip_reason="exact reporter options are not available",
    ),
    RegressionScenario(
        name="qwen3vl-moe-ctxgather-repeats",
        stage="compile",
        source_model_card="Qwen/Qwen3-VL-30B-A3B-Instruct",
        tiny_model_id=None,
        official_model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        summary="multi-resolution prefill CtxGatherCB non-constant repeats compiler failure",
        model_api="image_text_to_text",
        load_kwargs={"kv_offload": True},
        compile_kwargs={
            "prefill_seq_len": 128,
            "ctx_len": 4096,
            "img_size": 1540,
            "batch_size": 1,
            "num_devices": 1,
            "num_cores": 16,
        },
    ),
    RegressionScenario(
        name="mixtral-oom-export",
        stage="export",
        source_model_card="mistralai/Mixtral-8x7B-Instruct-v0.1",
        tiny_model_id=TINY_MIXTRAL,
        summary="full Mixtral export RAM spike requires memory measurement",
        skip_reason="full-model memory regression requires dedicated memory measurement job",
    ),
)


# Parse targeted scenario names before parametrization writes reports.
def _selected_scenario_names() -> tuple[str, ...]:
    """Return scenario names requested by ``QEFF_REPRODUCER_SCENARIO``."""
    names = []
    for name in os.environ.get(SCENARIO_FILTER_ENV, "").split(","):
        stripped_name = name.strip()
        if stripped_name and stripped_name not in names:
            names.append(stripped_name)
    return tuple(names)


# Build pytest parameters once so skipped regressions remain visible in collection.
def _params_for(stage: Stage) -> tuple[Any, ...]:
    """Return pytest parameters for one reproducer stage.

    The helper exists so every test function gets identical skip handling,
    stable readable IDs, and QAIC markers for hardware-backed stages.
    """
    selected_names = set(_selected_scenario_names())
    unknown_names = selected_names - {scenario.name for scenario in SCENARIOS}
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise pytest.UsageError(f"unknown reproducer scenario(s): {names}")
    params = []
    for scenario in SCENARIOS:
        if scenario.stage != stage:
            continue
        if selected_names and scenario.name not in selected_names:
            continue
        marks = []
        skip_reason = _skip_reason_for(scenario)
        if skip_reason:
            _record_result(scenario, "SKIPPED", skip_reason)
            marks.append(pytest.mark.skip(reason=skip_reason))
        if stage in {"compile", "inference"}:
            marks.append(pytest.mark.on_qaic)
        if scenario.model_api in {"wan_t2v", "wan_i2v"}:
            marks.append(pytest.mark.wan)
        if scenario.model_api == "image_text_to_text":
            marks.append(pytest.mark.multimodal)
        params.append(pytest.param(scenario, id=scenario.name, marks=marks))
    if not params:
        params.append(
            pytest.param(None, id=f"no-{stage}-configs", marks=pytest.mark.skip(reason=f"no selected {stage} configs"))
        )
    return tuple(params)


# Full official models are gated so normal developer runs stay bounded.
def _skip_reason_for(scenario: RegressionScenario) -> str | None:
    """Return the collection-time skip reason for a scenario, if any.

    This helper exists to keep full-model reproducers present in the catalog
    while making expensive official-model execution an explicit developer choice.
    """
    if scenario.skip_reason:
        return scenario.skip_reason
    if _requires_full_model(scenario) and not _run_full_models():
        return f"set {FULL_MODEL_ENV}=1 to run official model {scenario.official_model_id}"
    return None


# Official model fallback is required when tiny reductions cannot reproduce a bug.
def _requires_full_model(scenario: RegressionScenario) -> bool:
    """Return whether a scenario must use the official model card.

    The helper treats missing tiny checkpoints and explicit official-model runs
    as full-model cases so they never run accidentally in quick local checks.
    """
    if scenario.run_full_model_by_default:
        return False
    return bool(scenario.official_model_id and (not scenario.tiny_model_id or _use_official_models()))


# Env switches keep quick local validation separate from production repro runs.
def _run_full_models() -> bool:
    """Return whether full official models should be loaded and compiled."""
    return os.environ.get(FULL_MODEL_ENV) == "1"


def _use_official_models() -> bool:
    """Return whether official model cards override tiny reproducer models."""
    return os.environ.get(OFFICIAL_MODEL_ENV) == "1"


# Select the actual model card for the current run mode.
def _scenario_model_id(scenario: RegressionScenario) -> str:
    """Resolve the tiny or official model card for a scenario.

    Tiny models are preferred for speed; official models are used when requested
    or when no faithful reduced checkpoint exists.
    """
    if scenario.official_model_id and (not scenario.tiny_model_id or _use_official_models()):
        if not _run_full_models():
            pytest.skip(f"set {FULL_MODEL_ENV}=1 to run official model {scenario.official_model_id}")
        return scenario.official_model_id
    if scenario.tiny_model_id:
        return scenario.tiny_model_id
    pytest.skip("scenario has no runnable model")


# Report writer persists producer/consumer-readable results as tests execute.
def _record_result(scenario: RegressionScenario, status: str, detail: str) -> None:
    """Append one scenario verdict to the Markdown reproducer report.

    The helper writes a real PASS/SKIP/FAIL ledger so developers can correlate
    every reproducer config with its latest observed verdict while pytest runs.
    """
    if _is_pytest_collect_only():
        return
    path = Path(os.environ.get(REPORT_PATH_ENV, DEFAULT_REPORT_PATH))

    if scenario.official_model_id and (not scenario.tiny_model_id or _use_official_models()):
        model_id = scenario.official_model_id
    else:
        model_id = scenario.tiny_model_id or scenario.official_model_id or scenario.source_model_card
    safe_detail = detail.replace("\n", "<br>").replace("|", "\\|")
    follow_up_text = scenario.follow_up
    if not follow_up_text and scenario.skip_reason:
        follow_up_text = "add a portable runnable reproducer or confirm this remains non-runnable"
    follow_up = (follow_up_text or "").replace("\n", "<br>").replace("|", "\\|")
    row = (
        f"| {scenario.name} | {scenario.stage} | {scenario.model_api} | {model_id} | "
        f"{status} | {follow_up} | {safe_detail} |\n"
    )
    hf_cache_path = Path(
        os.environ.get("QEFF_REPRODUCER_HF_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or "/home/tmp/qeff_reproducer_hf_cache"
    )
    if not os.access(hf_cache_path.parent, os.W_OK):
        hf_cache_path = Path("/tmp/qeff_reproducer_hf_cache")

    with _locked_report(path) as handle:
        handle.seek(0)
        content = handle.read()
        if not content.startswith("# QEfficient Reproducer Config Results"):
            handle.seek(0)
            handle.truncate()
            handle.write(
                "# QEfficient Reproducer Config Results\n\n"
                "This file is generated by `tests/reproducer_configs/test_reported_reproducer_configs.py`.\n"
                "Each row is a concrete verdict from collection or execution of one reproducer config.\n\n"
                f"Hugging Face cache: `{hf_cache_path}`\n\n"
                "| Scenario | Stage | API | Model Card | Verdict | Follow-up | Detail |\n"
                "|---|---|---|---|---|---|---|\n"
            )
            content = ""
        if row not in content:
            handle.write(row)


def _record_failure(scenario: RegressionScenario, exc: BaseException) -> None:
    """Record a compact failure signature for a scenario.

    The helper keeps the Markdown report readable while preserving the exception
    type, first message lines, and compiler operator/error lines that usually
    carry the actionable failure signature.
    """
    message_lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    selected_lines = message_lines[:4]
    for line in message_lines[4:]:
        if any(pattern.search(line) for pattern in COMPILER_ERROR_PATTERNS) and line not in selected_lines:
            selected_lines.append(line)
        if len(selected_lines) >= 6:
            break
    signature = f"{type(exc).__name__}: {selected_lines[0] if selected_lines else ''}"
    if len(selected_lines) > 1:
        signature = "<br>".join([signature, *selected_lines[1:]])
    _record_result(scenario, "FAILED", signature)


def _run_with_verdict(scenario: RegressionScenario, runner, *args) -> None:
    """Run one scenario helper and record FAIL before propagating exceptions."""
    try:
        runner(scenario, *args)
    except BaseException as exc:
        if not isinstance(exc, pytest.skip.Exception):
            _record_failure(scenario, exc)
        raise


# Per-test environment handling prevents cache and QEFF_HOME leakage.
@pytest.fixture(autouse=True)
def _restore_reproducer_environment():
    """Set and restore mutable reproducer environment variables for each test.

    The fixture exists because model loading and WAN compilation require process
    environment variables, but those settings must not leak into later tests.
    """
    keys = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_MODULES_CACHE",
        "QEFF_HOME",
    )
    original_values = {key: os.environ.get(key) for key in keys}
    requested_cache = Path(
        os.environ.get("QEFF_REPRODUCER_HF_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or "/home/tmp/qeff_reproducer_hf_cache"
    )
    if not os.access(requested_cache.parent, os.W_OK):
        requested_cache = Path("/tmp/qeff_reproducer_hf_cache")
    modules_cache = requested_cache / "modules"
    try:
        requested_cache.mkdir(parents=True, exist_ok=True)
        modules_cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        pytest.skip(f"cannot create Hugging Face cache {requested_cache}: {exc}")

    os.environ["HF_HOME"] = str(requested_cache)
    os.environ["HF_HUB_CACHE"] = str(requested_cache)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)
    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
        import transformers.utils.hub as transformers_hub

        dynamic_module_utils.HF_MODULES_CACHE = str(modules_cache)
        transformers_hub.HF_MODULES_CACHE = str(modules_cache)
    except (ImportError, AttributeError) as exc:
        warnings.warn(f"could not update transformers module cache paths: {exc}", stacklevel=2)
    yield
    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# Load a fresh model per stage to avoid re-exporting offloaded module state.
def _load_causal_model(scenario: RegressionScenario):
    """Load a fresh causal QEfficient model for a runnable scenario.

    Each caller receives a new model instance so export, compile, and inference
    never share mutated/offloaded weights across independent reproducer tests.
    """
    from QEfficient import QEFFAutoModelForCausalLM

    model_id = _scenario_model_id(scenario)
    load_kwargs = {"dtype": torch.float32, "trust_remote_code": True, **scenario.load_kwargs}
    load_kwargs.setdefault("cache_dir", os.environ["HF_HUB_CACHE"])
    return QEFFAutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)


# VLM official-model cases compile vision, prefill, and decode as separate QPCs.
def _load_vlm_model(scenario: RegressionScenario):
    """Load a fresh image-text-to-text QEfficient model for VLM reproducers.

    The helper exists because VLM reports must exercise the dual-QPC API rather
    than the causal-LM helper used by text-only models.
    """
    from transformers import AutoConfig

    from QEfficient import QEFFAutoModelForImageTextToText
    from QEfficient.utils.test_utils import set_num_layers_vlm

    model_id = _scenario_model_id(scenario)
    load_kwargs = {"trust_remote_code": True, **scenario.load_kwargs}
    load_kwargs.setdefault("cache_dir", os.environ["HF_HUB_CACHE"])
    # VLM constructors reject ``num_hidden_layers`` as a top-level argument.
    # Apply the reduced-layer setting to the nested text config instead, then
    # allow the tiny checkpoint's layer-shaped weights to be loaded partially.
    num_hidden_layers = load_kwargs.pop("num_hidden_layers", -1)
    if num_hidden_layers != -1:
        config = AutoConfig.from_pretrained(model_id, **load_kwargs)
        set_num_layers_vlm(config, num_hidden_layers)
        load_kwargs["config"] = config
        load_kwargs.setdefault("ignore_mismatched_sizes", True)
    return QEFFAutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)


# Embedding reports use QEFFAutoModel rather than causal or VLM wrappers.
def _load_embedding_model(scenario: RegressionScenario):
    """Load a fresh embedding QEfficient model for sequence/reranker reports.

    The helper exists so embedding compile regressions exercise the same API as
    production embedding examples instead of the causal-LM wrapper.
    """
    from QEfficient import QEFFAutoModel

    model_id = _scenario_model_id(scenario)
    load_kwargs = {"trust_remote_code": True, **scenario.load_kwargs}
    load_kwargs.setdefault("cache_dir", os.environ["HF_HUB_CACHE"])
    return QEFFAutoModel.from_pretrained(model_id, **load_kwargs)


# A compile verdict requires the QPC directory and its program binary.
def _qpc_succeeded(qpc_path: str | Path) -> None:
    """Assert that QEfficient compile returned a usable QPC directory.

    This helper converts compile completion into an artifact-level verdict by
    checking the directory and required ``programqpc.bin`` output.
    """
    qpc = Path(qpc_path)
    assert qpc.is_dir(), f"QPC directory was not created: {qpc}"
    assert (qpc / "programqpc.bin").is_file(), f"programqpc.bin missing in QPC: {qpc}"


# Import/install reports are validated through real module imports and env checks.
def _run_import_and_install_test(scenario: RegressionScenario) -> None:
    """Validate import/install regressions without model artifacts.

    The helper exists for reports where package import order or environment
    setup caused the bug before export, compile, or runtime was reached.
    """
    import transformers  # noqa: F401

    import QEfficient  # noqa: F401

    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    _record_result(scenario, "PASSED", "imported QEfficient and transformers with hf_transfer enabled")


# Download reports exercise QEfficient's real Hugging Face download wrapper.
def _run_model_download_test(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Download the scenario's checkpoint through QEfficient utilities.

    The helper verifies model-resolution regressions at the download boundary
    before any QEfficient model wrapping or export logic is involved.
    """
    from QEfficient.utils import hf_download

    model_id = _scenario_model_id(scenario)
    model_path = Path(hf_download(repo_id=model_id, cache_dir=str(tmp_path / "hf-cache")))
    assert model_path.exists()
    _record_result(scenario, "PASSED", f"downloaded {model_path}")


# Export reports must call the same QEfficient export API users call.
def _run_model_export_test(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Export the scenario's model and assert ONNX files are produced.

    The helper exists to turn reporter export options into a real PyTorch to
    ONNX verdict instead of a placeholder option-plumbing assertion.
    """
    if scenario.model_api == "replicate_kv_heads":
        from scripts.replicate_kv_head.replicate_kv_heads import replicate_kv_heads

        model_id = _scenario_model_id(scenario)
        cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            replicate_kv_heads(model_name=model_id, **scenario.export_kwargs)
        finally:
            os.chdir(cwd)
        _record_result(scenario, "PASSED", "replicate_kv_heads export completed")
        return

    qeff_model = (
        _load_vlm_model(scenario) if scenario.model_api == "image_text_to_text" else _load_causal_model(scenario)
    )
    onnx_path = qeff_model.export(export_dir=tmp_path / scenario.name, **scenario.export_kwargs)
    paths = [Path(path) for path in onnx_path] if isinstance(onnx_path, list) else [Path(onnx_path)]
    assert all(path.is_file() for path in paths), f"ONNX export did not create files: {paths}"
    _record_result(scenario, "PASSED", "exported " + ", ".join(str(path) for path in paths))


# Compiler helper signatures provide the public options that should stay in sync.
def _known_qeff_compile_options() -> set[str]:
    """Return compile options accepted by QEfficient's helper APIs.

    The helper derives standard QAIC/QNN options from the public compile helper
    signatures and keeps only MDP or model-specialization extras local here.
    """
    from QEfficient.compile import compile_helper, qnn_compiler

    options = set(EXTRA_QEFF_COMPILE_OPTIONS)
    options.update(inspect.signature(compile_helper.compile).parameters)
    options.update(inspect.signature(qnn_compiler.compile).parameters)
    return options


# Compile kwargs are checked before QEfficient receives reporter options.
def _assert_compile_kwargs_supported(
    compile_owner, scenario: RegressionScenario, compile_kwargs: dict[str, Any]
) -> None:
    """Assert that compile options match the public API or known compiler flags.

    The helper catches typos before long export/compile jobs start. ``node_precision_info``
    is allowed as a compiler option only when the referenced YAML exists in this checkout.
    """
    parameters = inspect.signature(compile_owner.compile).parameters
    has_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    private_kwargs = {key for key in compile_kwargs if key.startswith("_")}
    assert not private_kwargs, f"{scenario.name}: private harness kwargs escaped to compile {private_kwargs}"

    accepted = set(parameters)
    if has_var_kwargs:
        accepted |= _known_qeff_compile_options()
    unknown = set(compile_kwargs) - accepted
    assert not unknown, f"{scenario.name}: unknown compile kwargs {unknown}"

    npi_path = compile_kwargs.get("node_precision_info")
    if npi_path:
        path = Path(npi_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        assert path.is_file(), f"{scenario.name}: node_precision_info YAML does not exist: {path}"


# MDP intersection reproducers generate their compiler dump locally.
def _compile_with_optional_mdp(qeff_model, scenario: RegressionScenario, compile_dir: Path):
    """Compile once or run the local two-pass MDP intersection flow.

    The helper replaces reporter-machine MDP JSON paths with artifacts generated
    inside pytest's temp directory, following the Qwen3-VL MDP compile example.
    """
    compile_kwargs = dict(scenario.compile_kwargs)
    use_intersection = compile_kwargs.pop("_mdp_intersection", False)
    if not use_intersection:
        _assert_compile_kwargs_supported(qeff_model, scenario, compile_kwargs)
        return qeff_model.compile(compile_dir=compile_dir, **compile_kwargs)

    dump_path = compile_dir / "mdp_compiler_dump.json"
    (compile_dir / "mdp_dump").mkdir(parents=True, exist_ok=True)
    (compile_dir / "mdp_intersection").mkdir(parents=True, exist_ok=True)
    dump_kwargs = dict(compile_kwargs)
    dump_kwargs.pop("mdp_num_partitions", None)
    dump_kwargs.pop("mdp_strategy", None)
    _assert_compile_kwargs_supported(qeff_model, scenario, {**dump_kwargs, "mdp_dump_partition_config": str(dump_path)})
    qeff_model.compile(compile_dir=compile_dir / "mdp_dump", mdp_dump_partition_config=str(dump_path), **dump_kwargs)

    final_kwargs = dict(compile_kwargs)
    final_kwargs["mdp_strategy"] = "intersection"
    final_kwargs["mdp_compiler_dump_path"] = str(dump_path)
    _assert_compile_kwargs_supported(qeff_model, scenario, final_kwargs)
    return qeff_model.compile(compile_dir=compile_dir / "mdp_intersection", **final_kwargs)


# Compile reports use QEfficient's compile API and validate the produced QPC.
def _run_model_compile_test(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Compile the scenario's model with reported compile options.

    The helper exists so active compile scenarios provide production-style
    success/failure verdicts from the QEfficient export-plus-compile stack.
    """
    if not Path("/opt/qti-aic/exec/qaic-compile").is_file():
        pytest.skip("qaic-compile is not available on this host")
    if scenario.model_api == "image_text_to_text":
        _run_vlm_compile_test(scenario, tmp_path)
        return
    if scenario.model_api in {"wan_t2v", "wan_i2v"}:
        _run_wan_compile_test(scenario, tmp_path)
        return

    qeff_model = _load_embedding_model(scenario) if scenario.model_api == "embedding" else _load_causal_model(scenario)
    compile_dir = tmp_path / scenario.name / "compile"
    compile_dir.mkdir(parents=True, exist_ok=True)
    qpc_path = _compile_with_optional_mdp(qeff_model, scenario, compile_dir)
    _qpc_succeeded(qpc_path)
    _record_result(scenario, "PASSED", f"QPC: {qpc_path}")


# Dual-QPC VLM compile is validated as vision + language prefill + language decode.
def _run_vlm_compile_test(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Compile all VLM deployment components for one reproducer config.

    The helper exists for full VLM reports where runtime generation is too heavy
    but compilation of vision, language prefill, and language decode must be an
    absolute pre-merge verdict.
    """
    qeff_model = _load_vlm_model(scenario)
    compile_dir = tmp_path / scenario.name / "compile"
    compile_dir.mkdir(parents=True, exist_ok=True)
    export_dir = tmp_path / scenario.name / "export"
    export_kwargs = dict(scenario.export_kwargs)
    export_kwargs.setdefault("offload_pt_weights", False)
    # Export must use the same component selection as compile.  In particular,
    # skip_vision prevents the tiny language-only reproducer from tracing the
    # unsupported vision rotary-input shape.
    skip_vision = bool(scenario.compile_kwargs.get("skip_vision", False))
    skip_lang = bool(scenario.compile_kwargs.get("skip_lang", False))
    onnx_paths = qeff_model.export(
        export_dir=export_dir,
        skip_vision=skip_vision,
        skip_lang=skip_lang,
        **export_kwargs,
    )
    vision_onnx_path, lang_onnx_path = onnx_paths
    common_kwargs = dict(scenario.compile_kwargs)
    use_intersection = common_kwargs.pop("_mdp_intersection", False)
    common_kwargs.pop("skip_vision", None)
    common_kwargs.pop("skip_lang", None)
    vision_kwargs = dict(common_kwargs)
    for language_only_key in (
        "prefill_only",
        "enable_chunking",
        "retain_full_kv",
        "mdp_num_partitions",
        "mdp_strategy",
        "mdp_compiler_dump_path",
        "mdp_dump_partition_config",
    ):
        vision_kwargs.pop(language_only_key, None)
    vision_qpcs = {}
    # Do not call compile() for a component explicitly excluded by the
    # scenario: omitting its ONNX path would make QEff re-export that component.
    if not skip_vision:
        _assert_compile_kwargs_supported(qeff_model, scenario, {**vision_kwargs, "skip_lang": True})
        vision_qpcs = qeff_model.compile(
            compile_dir=compile_dir / "vision",
            vision_onnx_path=vision_onnx_path,
            lang_onnx_path=lang_onnx_path,
            skip_lang=True,
            **vision_kwargs,
        )
    prefill_kwargs = dict(common_kwargs)
    prefill_kwargs["prefill_only"] = True
    if use_intersection:
        dump_path = compile_dir / "lang_prefill_mdp_dump.json"
        (compile_dir / "lang_prefill_mdp_dump").mkdir(parents=True, exist_ok=True)
        (compile_dir / "lang_prefill").mkdir(parents=True, exist_ok=True)
        dump_kwargs = dict(prefill_kwargs)
        dump_kwargs.pop("mdp_num_partitions", None)
        dump_kwargs.pop("mdp_strategy", None)
        _assert_compile_kwargs_supported(
            qeff_model, scenario, {**dump_kwargs, "skip_vision": True, "mdp_dump_partition_config": str(dump_path)}
        )
        qeff_model.compile(
            compile_dir=compile_dir / "lang_prefill_mdp_dump",
            vision_onnx_path=vision_onnx_path,
            lang_onnx_path=lang_onnx_path,
            skip_vision=True,
            mdp_dump_partition_config=str(dump_path),
            **dump_kwargs,
        )
        prefill_kwargs["mdp_strategy"] = "intersection"
        prefill_kwargs["mdp_compiler_dump_path"] = str(dump_path)
    _assert_compile_kwargs_supported(qeff_model, scenario, {**prefill_kwargs, "skip_vision": True})
    prefill_qpcs = qeff_model.compile(
        compile_dir=compile_dir / "lang_prefill",
        vision_onnx_path=vision_onnx_path,
        lang_onnx_path=lang_onnx_path,
        skip_vision=True,
        **prefill_kwargs,
    )
    decode_kwargs = {**common_kwargs, "prefill_seq_len": 1}
    for prefill_only_key in ("prefill_only", "enable_chunking", "mdp_num_partitions", "mdp_strategy"):
        decode_kwargs.pop(prefill_only_key, None)
    _assert_compile_kwargs_supported(qeff_model, scenario, {**decode_kwargs, "skip_vision": True})
    decode_qpcs = qeff_model.compile(
        compile_dir=compile_dir / "lang_decode",
        vision_onnx_path=vision_onnx_path,
        lang_onnx_path=lang_onnx_path,
        skip_vision=True,
        **decode_kwargs,
    )
    for qpc_map in (vision_qpcs, prefill_qpcs, decode_qpcs):
        for qpc_path in qpc_map.values():
            _qpc_succeeded(qpc_path)
    _record_result(scenario, "PASSED", f"VLM QPCs: {vision_qpcs} {prefill_qpcs} {decode_qpcs}")


# WAN reports use the real diffusion pipeline compile API, not transformer helpers.
def _run_wan_compile_test(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Compile a WAN pipeline reproducer with the reported pipeline options.

    The helper exists so WAN regressions stay in the same reproducer catalog and
    validate the real QEfficient WAN export/compile stack before merge.
    """
    model_id = _scenario_model_id(scenario)
    if scenario.model_api == "wan_i2v":
        from QEfficient.diffusers.pipelines.wan.pipeline_wan_i2v import QEffWanImageToVideoPipeline as Pipeline
    else:
        from QEfficient.diffusers.pipelines.wan.pipeline_wan import QEffWanPipeline as Pipeline

    pipeline = Pipeline.from_pretrained(model_id, cache_dir=os.environ["HF_HUB_CACHE"])
    os.environ.setdefault("QEFF_HOME", str(tmp_path / scenario.name / "qeff_home"))
    _assert_compile_kwargs_supported(pipeline, scenario, dict(scenario.compile_kwargs))
    qpc_path = pipeline.compile(**scenario.compile_kwargs)
    if qpc_path:
        _record_result(scenario, "PASSED", f"WAN QPC: {qpc_path}")
    else:
        _record_result(scenario, "PASSED", "WAN compile completed")


# Inference reports compile first, then run generation through QEfficient runtime.
def _run_model_inference_test(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Compile and generate for runnable end-to-end inference scenarios.

    The helper exists for reports whose failure only appears after QPC creation
    and validates that generation returns model output IDs.
    """
    from QEfficient.utils._utils import load_hf_tokenizer

    if not Path("/opt/qti-aic/exec/qaic-compile").is_file():
        pytest.skip("QAIC runtime is unavailable because qaic-compile is not installed on this host")
    if not any(Path("/dev").glob("qaic*")):
        pytest.skip("no /dev/qaic* device is available on this host")
    try:
        from QEfficient.generation import cloud_infer
    except Exception as exc:
        pytest.skip(f"QAIC runtime import failed: {exc}")
    if not (cloud_infer.is_qaicrt_imported and cloud_infer.is_aicapi_imported):
        pytest.skip("qaicrt or QAicApi_pb2 runtime bindings are not available")
    qeff_model = _load_causal_model(scenario)
    compile_dir = tmp_path / scenario.name / "compile"
    compile_dir.mkdir(parents=True, exist_ok=True)
    _assert_compile_kwargs_supported(qeff_model, scenario, dict(scenario.compile_kwargs))
    qpc_path = qeff_model.compile(compile_dir=compile_dir, **scenario.compile_kwargs)
    _qpc_succeeded(qpc_path)
    tokenizer = load_hf_tokenizer(_scenario_model_id(scenario))
    exec_info = qeff_model.generate(tokenizer=tokenizer, prompts=["My name is"], **scenario.inference_kwargs)
    assert exec_info.generated_ids is not None
    _record_result(scenario, "PASSED", "generation returned output IDs")


# Parametrized entrypoint for import/install-stage reported regressions.
@pytest.mark.reproducer
@pytest.mark.parametrize("scenario", _params_for("import_install"))
def test_import_and_install_reproducer(scenario: RegressionScenario) -> None:
    """Run every import/install scenario through the real import helper."""
    _run_with_verdict(scenario, _run_import_and_install_test)


# Parametrized entrypoint for model-download reported regressions.
@pytest.mark.reproducer
@pytest.mark.parametrize("scenario", _params_for("download"))
def test_model_download_reproducer(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Run every download scenario through the real QEfficient download helper."""
    _run_with_verdict(scenario, _run_model_download_test, tmp_path)


# Parametrized entrypoint for model-export reported regressions.
@pytest.mark.reproducer
@pytest.mark.parametrize("scenario", _params_for("export"))
def test_model_export_reproducer(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Run every export scenario through the real QEfficient export helper."""
    _run_with_verdict(scenario, _run_model_export_test, tmp_path)


# Parametrized entrypoint for compile-stage reported regressions.
@pytest.mark.reproducer
@pytest.mark.parametrize("scenario", _params_for("compile"))
def test_model_compile_reproducer(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Run every compile scenario through the real QEfficient compile helper."""
    _run_with_verdict(scenario, _run_model_compile_test, tmp_path)


# Parametrized entrypoint for end-to-end inference reported regressions.
@pytest.mark.reproducer
@pytest.mark.parametrize("scenario", _params_for("inference"))
def test_inference_reproducer(scenario: RegressionScenario, tmp_path: Path) -> None:
    """Run every inference scenario through real compile and generation."""
    _run_with_verdict(scenario, _run_model_inference_test, tmp_path)


# Catalog guard prevents future edits from dropping reported configs silently.
def test_all_reported_configs_are_represented() -> None:
    """Ensure the reported-regression catalog remains complete and unique.

    The test exists because skipped scenarios are still deliberate coverage
    records and must not disappear while the suite evolves.
    """
    assert len(SCENARIOS) == EXPECTED_SCENARIO_COUNT
    assert len({scenario.name for scenario in SCENARIOS}) == len(SCENARIOS)


# Path guard recursively catches local reporter paths in nested option values.
def _contains_external_path(value: Any) -> bool:
    """Return whether a config value contains a non-portable local path.

    The helper exists because reproducer configs may nest paths inside lists or
    dictionaries, and active scenarios must remain runnable on production hosts.
    """
    if isinstance(value, dict):
        return any(_contains_external_path(key) or _contains_external_path(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_external_path(item) for item in value)
    text = str(value)
    return "/home/" in text or "/local/" in text


# Active scenario guard keeps runnable tests independent of reporter machines.
def test_active_scenarios_are_real_stage_verdicts() -> None:
    """Ensure runnable scenarios use models and portable config values.

    The test exists to prevent fake active tests and accidental dependencies on
    absolute paths copied from a reporter's private machine.
    """
    for scenario in SCENARIOS:
        if scenario.skip_reason:
            continue
        if scenario.stage not in {"import_install"}:
            assert scenario.tiny_model_id or scenario.official_model_id
        combined = {
            **scenario.config,
            **scenario.load_kwargs,
            **scenario.export_kwargs,
            **scenario.compile_kwargs,
            **scenario.inference_kwargs,
        }
        assert not _contains_external_path(combined)
