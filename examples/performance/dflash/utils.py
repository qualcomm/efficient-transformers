# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from datasets import Features, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import (
    QAICInferenceSession,
    is_retained_state_name,
)
from QEfficient.generation.dflash_generation import get_qwen35_target_seq_len
from QEfficient.transformers.models.dspark_draft import QEffDSparkForCausalLM
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")
# ─────────────────────────────────────────────────────────────────────────────
# model_name (TLM short)  →  (TLM HF repo, DLM HF repo)
# DLM column is the canonical DFlash repo. TLM column is the standard HF repo
# when known; otherwise None and must be supplied via --tlm_hf_path.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_MAP = {
    "gpt-oss-20b": ("openai/gpt-oss-20b", "z-lab/gpt-oss-20b-DFlash"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "z-lab/gpt-oss-120b-DFlash"),
    "gemma-4-31B-it": ("google/gemma-4-31B-it", "z-lab/gemma-4-31B-it-DFlash"),
    "Qwen3-4B": ("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"),
    "Qwen3.5-4B": ("Qwen/Qwen3.5-4B", "z-lab/Qwen3.5-4B-DFlash"),
    "Qwen3.6-27B": ("Qwen/Qwen3.6-27B", "z-lab/Qwen3.6-27B-DFlash"),
    "Qwen3.6-35B-A3B": ("Qwen/Qwen3.6-35B-A3B", "z-lab/Qwen3.6-35B-A3B-DFlash"),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"),
    "Llama-3.1-8B-Instruct": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
    ),
    "Qwen3-VL-32B-Instruct": (
        "Qwen/Qwen3-VL-32B-Instruct",
        "path_qwen3_vl_model",
    ),
    "Qwen3-4B-DSpark": ("Qwen/Qwen3-4B", "deepseek-ai/dspark_qwen3_4b_block7"),
    "Qwen3-8B-DSpark": ("Qwen/Qwen3-8B", "deepseek-ai/dspark_qwen3_8b_block7"),
    "Qwen3-14B-DSpark": ("Qwen/Qwen3-14B", "deepseek-ai/dspark_qwen3_14b_block7"),
    "Gemma4-12B-DSpark": (
        "google/gemma-4-12B-it",
        "deepseek-ai/dspark_gemma4_12b_block7",
    ),
    "Qwen3.8-27B-DSpark": ("Qwen/Qwen3.8-27B", "RadixArk/Qwen3.8-27B-DSpark"),
}


def _build_aliases(model_map):
    aliases = {}
    for short, (tlm_repo, _) in model_map.items():
        aliases[short.lower()] = short
        if tlm_repo:
            aliases[tlm_repo.lower()] = short
            aliases[tlm_repo.split("/", 1)[-1].lower()] = short
    return aliases


MODEL_ALIASES = _build_aliases(MODEL_MAP)


def load_spd_session(qpc: str, devices: list[int]):
    session = QAICInferenceSession(qpc, devices)
    session.skip_buffers(
        [
            name
            for name in session.input_names + session.output_names
            if is_retained_state_name(name) or name.endswith("_RetainedState")
        ]
    )
    return session


def load_spd_sessions(tlm_qpc: str, dlm_qpc: str, tlm_devices: list[int], dlm_devices: list[int]):
    """Create DFlash sessions and enable retained-state cache handling."""
    return load_spd_session(dlm_qpc, dlm_devices), load_spd_session(tlm_qpc, tlm_devices)


def get_spd_prompt_chunk_size(tlm_session: QAICInferenceSession) -> int:
    """Return the largest compiled input sequence length used for TLM prefill."""
    input_index = tlm_session.binding_index_map["input_ids"]
    return max(
        [shape[input_index][1][1] for shape in tlm_session.allowed_shapes] + [tlm_session.bindings[input_index].dims[1]]
    )


def validate_spd_decode_specialization(tlm_session: QAICInferenceSession, block_size: int) -> None:
    """Require a TLM specialization that verifies one complete DFlash block."""
    if "cache_commit" in getattr(tlm_session, "input_names", ()):
        seq_len = get_qwen35_target_seq_len(tlm_session, block_size)
        logger.info(
            f"TLM uses concatenated replay + verification: {seq_len} input positions, {block_size} predictions."
        )
        return
    input_index = tlm_session.binding_index_map["input_ids"]
    sequence_lengths = sorted({shape[input_index][1][1] for shape in tlm_session.allowed_shapes})
    if not sequence_lengths:
        # Single-specialization QPCs report their fixed shape only in bindings.
        sequence_lengths = [tlm_session.bindings[input_index].dims[1]]
    logger.info(f"TLM input_ids seq_lens: {sequence_lengths}")
    if block_size not in sequence_lengths:
        raise ValueError(
            f"TLM QPC has no decode specialization with seq_len={block_size} "
            f"(found {sequence_lengths}). Recompile the TLM with dflash_block_size={block_size}."
        )


def resolve_model_name(name: str) -> str:
    """Map a user-supplied model name (short, full HF path, or basename) to
    the canonical short name used as a key in MODEL_MAP."""
    canonical = MODEL_ALIASES.get(name.lower())
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"unknown model_name '{name}'. Supported: " + ", ".join(sorted(MODEL_MAP.keys()))
        )
    return canonical


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [round(start + (i * span) / (num_draft_layers - 1)) for i in range(num_draft_layers)]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int] | None,
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def load_dflash_checkpoint(dflash_model_path: str) -> tuple[dict, dict]:
    """Load the DFlash safetensors checkpoint and config, from a local directory or
    a Hugging Face repo id.

    Returns
    -------
    state_dict : dict[str, Tensor]  — all tensors in fp32
    cfg        : dict               — parsed config.json
    """
    if os.path.isdir(dflash_model_path):
        bin_path = os.path.join(dflash_model_path, "model.safetensors")
        config_path = os.path.join(dflash_model_path, "config.json")
    else:
        bin_path = hf_hub_download(repo_id=dflash_model_path, filename="model.safetensors")
        config_path = hf_hub_download(repo_id=dflash_model_path, filename="config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    state_dict = {}
    with safe_open(bin_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        for key in keys:
            state_dict[key] = f.get_tensor(key).to(torch.float32)

    return state_dict, cfg


def get_dflash_block_size(config):
    config = config if isinstance(config, dict) else config.to_dict()
    block_size = config.get("block_size") or config.get("dflash_config", {}).get("block_size")
    if not isinstance(block_size, int) or block_size < 2:
        raise ValueError("DFlash config must provide block_size >= 2 (top-level or under dflash_config).")
    return block_size


def read_dlm_meta(dlm_repo: str, hf_token: str | None = None):
    """Load a DFlash checkpoint and return (state_dict, target_layer_ids, block_size)."""
    state_dict, cfg = load_dflash_checkpoint(dlm_repo)
    target_layer_ids = cfg.get("dflash_config", {}).get("target_layer_ids", [])
    block_size = get_dflash_block_size(cfg)
    return state_dict, target_layer_ids, block_size


def get_dlm_target_seq_len(block_size: int, prefill_seq_len: int, target_seq_len: int | None = None) -> int:
    target_seq_len = prefill_seq_len if target_seq_len is None else target_seq_len
    if (
        type(prefill_seq_len) is not int
        or prefill_seq_len < block_size
        or type(target_seq_len) is not int
        or target_seq_len < block_size
        or target_seq_len % block_size
        or prefill_seq_len % target_seq_len
    ):
        raise ValueError("DLM target length must be a positive multiple of block_size and divide prefill_seq_len.")
    return target_seq_len


def validate_dlm_target_seq_len(dlm_session, block_size, prefill_seq_len, requested=None):
    index = dlm_session.binding_index_map["target_hidden"]
    width = int(dlm_session.bindings[index].dims[1])
    get_dlm_target_seq_len(block_size, prefill_seq_len, width)
    if requested is not None and requested != width:
        raise ValueError(f"DLM QPC target length is {width}, but {requested} was requested. Recompile the DLM.")
    logger.info(f"DLM target length={width}, noise length={block_size}, prefill calls/chunk={prefill_seq_len // width}")


def _compile_tlm_phases(model, compile_kwargs, disagg_tlm):
    if not disagg_tlm:
        return model.compile(**compile_kwargs)
    compile_kwargs = compile_kwargs.copy()
    specializations = compile_kwargs.pop("specializations")
    qpcs = {}
    for phase, spec in zip(("prefill", "decode"), specializations):
        qpc = model.compile(
            **compile_kwargs,
            specializations=[spec],
            split_retained_state_io=True,
            offload_pt_weights=False,
        )
        qpcs[phase] = dict(qpc) if isinstance(qpc, dict) else str(qpc)
    return qpcs


def compile_tlm_qpc(
    tlm_repo: str,
    dlm_repo: str,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    hf_token: str | None = None,
    disagg_tlm: bool = False,
) -> str | dict[str, str]:
    """Build the TLM and compile it to a QPC. fc/hidden_norm are injected by
    QEfficient's DFlashTLMTransform (weights from dflash_dlm_repo)."""
    from QEfficient import QEFFAutoModelForCausalLM

    _, target_layer_ids, block_size = read_dlm_meta(dlm_repo, hf_token)
    tlm_target_ids = [i + 1 for i in target_layer_ids]

    logger.info(f"[compile_tlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    config = AutoConfig.from_pretrained(tlm_repo, token=hf_token)
    if disagg_tlm and config.model_type != "gpt_oss":
        raise ValueError("Split text DFlash TLM compilation currently supports GPT-OSS only.")
    qaic_config = {"target_layer_ids": tlm_target_ids, "dflash_dlm_repo": dlm_repo}
    compile_kwargs = {}
    if config.model_type == "gpt_oss":
        compile_kwargs = {
            "qaic_config": qaic_config,
            "prefill_only": True,
            "enable_chunking": True,
            "use_onnx_subfunctions": False,
            "specializations": [
                {
                    "batch_size": 1,
                    "seq_len": length,
                    "ctx_len": ctx_len,
                    "sliding_window": config.sliding_window,
                    "num_logits_to_keep": num_logits,
                }
                for length, num_logits in (
                    (prefill_seq_len, 1),
                    (block_size, block_size),
                )
            ],
        }
    tlm_qeff = QEFFAutoModelForCausalLM.from_pretrained(
        tlm_repo,
        config=config,
        torch_dtype=torch.float32,
        token=hf_token,
        qaic_config=qaic_config,
    )
    compile_kwargs.update(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        dflash_block_size=block_size,
    )
    qpc = _compile_tlm_phases(tlm_qeff, compile_kwargs, disagg_tlm)
    if not disagg_tlm:
        qpc = str(qpc)
    logger.info(f"[compile_tlm] qpc={qpc}")
    return qpc


def compile_dlm_qpc(
    tlm_repo: str,
    dlm_repo: str,
    *,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    prefill_seq_len: int = 128,
    target_seq_len: int | None = None,
    hf_token: str | None = None,
) -> str:
    """Build the DLM and compile it to a QPC. lm_head/embed_tokens are injected
    by QEfficient's DFlashDLMTransform (weights from dflash_tlm_repo)."""
    from QEfficient import QEFFAutoModelForCausalLM

    _, _, block_size = read_dlm_meta(dlm_repo, hf_token)

    target_seq_len = get_dlm_target_seq_len(block_size, prefill_seq_len, target_seq_len)
    qaic_config = {"dflash_dlm": True, "dflash_tlm_repo": tlm_repo}
    if target_seq_len != block_size:
        qaic_config["dflash_target_seq_len"] = target_seq_len

    logger.info(f"[compile_dlm] dlm={dlm_repo}  block_size={block_size}")
    dlm_qeff = QEFFAutoModelForCausalLM.from_pretrained(
        dlm_repo,
        torch_dtype=torch.float32,
        token=hf_token,
        qaic_config=qaic_config,
    )
    qpc = dlm_qeff.compile(
        prefill_seq_len=block_size,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        prefill_only=True,
    )
    qpc = str(qpc)
    logger.info(f"[compile_dlm] qpc={qpc}")
    return qpc


# fc-output range used to scale the injected DFlash fc so activations stay in fp16 range
# (RMSNorm(x/s) == RMSNorm(x), so the scaling is zero-accuracy-cost).
_VLM_FC_TARGET_ABSMAX = 128.0


def compile_gemma_vlm_qpcs(
    tlm_repo: str,
    dlm_repo: str,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    hf_token: str | None = None,
) -> tuple[str, str]:
    """Build a vision-language model (VLM) TLM — vision encoder + language decoder — and
    compile it for SPD. Returns ``(lang_qpc, vision_qpc)``.

    Built through the VLM class ``QEFFAutoModelForImageTextToText`` (``kv_offload=True``) so
    the language QPC accepts image features (``vision_embeds``) and a matching
    vision-encoder QPC is produced. The DFlash ``fc``/``hidden_norm`` are copied from the
    DFlash checkpoint into the inner text model and fc is scaled into fp16 range (same
    scaling as the text ``compile_tlm_qpc`` path); the DFlashTLMTransform attaches the
    modules but leaves them random until this injection runs.

    The vision encoder compiles on the same ``num_cores``/``num_devices`` budget as the
    language decoder. ``node_precision_info`` stays on: the VLM language decoder needs
    FP32 node-precision-info on its precision-sensitive ops or decode collapses.

    Currently exercised with the gemma4 VLM family; other VLMs can be added to MODEL_MAP.
    """

    state_dict, target_layer_ids, block_size = read_dlm_meta(dlm_repo, hf_token)
    tlm_target_ids = [i + 1 for i in target_layer_ids]

    logger.info(f"[compile_vlm_tlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    config = AutoConfig.from_pretrained(tlm_repo, trust_remote_code=True, token=hf_token)
    tlm_qeff = QEFFAutoModelForImageTextToText.from_pretrained(
        tlm_repo,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,
        ignore_mismatched_sizes=True,
        token=hf_token,
        qaic_config={"target_layer_ids": tlm_target_ids, "dflash_dlm_repo": dlm_repo},
    )

    # DFlashTLMTransform attached fc/hidden_norm but left them random; copy in the real
    # DFlash weights and scale fc into fp16 range.
    text_model = tlm_qeff.lang_model.model.language_model
    text_model.fc.weight.data.copy_(state_dict["fc.weight"].to(torch.float32))
    with torch.no_grad():
        in_feat = text_model.fc.in_features
        max_row_norm = text_model.fc.weight.data.norm(dim=1).max().item()
        s = max((in_feat**0.5) * max_row_norm / _VLM_FC_TARGET_ABSMAX, 1.0)
        text_model.fc.weight.data.div_(s)
    text_model.hidden_norm.weight.data.copy_(state_dict["hidden_norm.weight"].to(torch.float32))
    logger.info(f"[compile_vlm_tlm] fc/hidden_norm injected (fc scale s={s:.6f})")

    qpc = tlm_qeff.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
        use_onnx_subfunctions=config.model_type == "gemma4",
        split_model_io=True,
        batch_size=1,
        dflash_block_size=block_size,
        node_precision_info=True,
    )
    lang_qpc = str(qpc["lang_qpc_path"])
    vision_qpc = str(qpc["vision_qpc_path"])
    logger.info(f"[compile_vlm_tlm] lang_qpc={lang_qpc}  vision_qpc={vision_qpc}")
    return lang_qpc, vision_qpc


def compile_qwen3vl_vlm_qpcs(
    tlm_repo: str,
    dlm_repo: str,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    height: int | None = None,
    width: int | None = None,
    hf_token: str | None = None,
    disagg_tlm: bool = False,
) -> tuple[str | dict[str, str], str]:
    """Build the Qwen3-VL or Qwen3.5 TLM vision encoder + language decoder for
    SPD. Returns ``(lang_qpc, vision_qpc)``.

    Same DFlashTLMTransform-attaches-random / inject-real-weights pattern as
    ``compile_gemma_vlm_qpcs``, kept as a separate function (rather than generalizing
    that one) since qwen3-vl's compile knobs differ: it needs ``height``/``width``
    (vision input resolution) and does not use gemma's
    ``aic_enable_depth_first``/``node_precision_info``/``batch_size`` options.
    ``split_model_io`` IS required here too -- without it qaic-compile drops most of
    the graph's I/O from the compiled QPC's binding set (confirmed: only 6 of the
    expected 266 bindings survive, including no ``vision_embeds`` and no KV cache,
    even though the ONNX graph itself is correct). When ``height``/``width`` are
    ``None``, they are omitted so ``get_specializations`` applies its own
    ``constants.QWEN3_VL_HEIGHT``/``WIDTH`` fallback.
    """

    state_dict, target_layer_ids, block_size = read_dlm_meta(dlm_repo, hf_token)
    tlm_target_ids = [i + 1 for i in target_layer_ids]

    logger.info(f"[compile_qwen3vl_tlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    config = AutoConfig.from_pretrained(tlm_repo, trust_remote_code=True, token=hf_token)
    is_qwen35 = config.model_type == "qwen3_5"
    is_qwen35_moe = config.model_type == "qwen3_5_moe"
    if disagg_tlm and not is_qwen35_moe:
        raise ValueError("Split VLM DFlash TLM compilation currently supports Qwen3.5-MoE only.")
    qaic_config = {"target_layer_ids": tlm_target_ids, "dflash_dlm_repo": dlm_repo}
    if is_qwen35 or is_qwen35_moe:
        if block_size < 2 or block_size & (block_size - 1):
            raise ValueError("Qwen3.5 DFlash requires a power-of-two draft block_size >= 2.")
        if prefill_seq_len < 2 * block_size or prefill_seq_len % block_size:
            raise ValueError(
                f"Qwen3.5 DFlash requires prefill_seq_len to be a multiple of {block_size} and at least {2 * block_size}."
            )
        logger.info("Qwen3.5 DFlash combines accepted-prefix replay and verification in one TLM call.")
    tlm_qeff = QEFFAutoModelForImageTextToText.from_pretrained(
        tlm_repo,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,
        ignore_mismatched_sizes=True,
        token=hf_token,
        qaic_config=qaic_config,
    )

    # Apply the DLM projection weights with the same fp16-range scaling as the transform.
    text_model = tlm_qeff.lang_model.model.language_model
    text_model.fc.weight.data.copy_(state_dict["fc.weight"].to(torch.float32))
    with torch.no_grad():
        in_feat = text_model.fc.in_features
        max_row_norm = text_model.fc.weight.data.norm(dim=1).max().item()
        s = max((in_feat**0.5) * max_row_norm / _VLM_FC_TARGET_ABSMAX, 1.0)
        text_model.fc.weight.data.div_(s)
    text_model.hidden_norm.weight.data.copy_(state_dict["hidden_norm.weight"].to(torch.float32))
    logger.info(f"[compile_qwen3vl_tlm] fc/hidden_norm injected (fc scale s={s:.6f})")
    compile_kwargs = {
        "prefill_seq_len": prefill_seq_len,
        "ctx_len": ctx_len,
        "num_cores": num_cores,
        "num_devices": num_devices,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "mos": 1,
        "use_onnx_subfunctions": False,
        "split_model_io": True,
        "dflash_block_size": block_size,
    }
    if is_qwen35 or is_qwen35_moe:
        # Concatenated replay and verification use chunk attention.
        compile_kwargs["prefill_only"] = True
    if is_qwen35_moe:
        compile_kwargs["qaic_config"] = qaic_config
        compile_kwargs["enable_chunking"] = True
    if height is not None:
        compile_kwargs["height"] = height
    if width is not None:
        compile_kwargs["width"] = width

    if disagg_tlm:
        text_model.dflash_chunk_size = block_size
        specializations, _ = tlm_qeff.model.get_specializations(
            batch_size=1,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            height=height,
            width=width,
            kv_offload=True,
        )
        compile_kwargs["specializations"] = specializations["lang"]

    qpc = _compile_tlm_phases(tlm_qeff, compile_kwargs, disagg_tlm)
    if disagg_tlm:
        lang_qpc = {phase: str(paths["lang_prefill_qpc_path"]) for phase, paths in qpc.items()}
        vision_qpc = str(qpc["prefill"]["vision_qpc_path"])
    else:
        lang_qpc = str(qpc["lang_prefill_qpc_path"] if is_qwen35 or is_qwen35_moe else qpc["lang_qpc_path"])
        vision_qpc = str(qpc["vision_qpc_path"])
    logger.info(f"[compile_qwen3vl_tlm] lang_qpc={lang_qpc}  vision_qpc={vision_qpc}")
    return lang_qpc, vision_qpc


def _get_text_inner(base_model: AutoModelForCausalLM) -> nn.Module:
    """Return the submodule that owns the text decoder (``.layers`` / ``.embed_tokens``).

    Plain causal LMs (Llama, Qwen3): ``base_model.model``. VLMs whose text decoder is
    nested (e.g. gemma4 ``Gemma4ForConditionalGeneration``): ``base_model.model.language_model``.
    """
    inner = base_model.model
    if not hasattr(inner, "layers") and hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner


def extract_lm_head(
    model: AutoModelForCausalLM,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (lm_head_weight, lm_head_bias) as fp32. Falls back to the module when the
    state_dict has no ``lm_head.weight`` (tied embeddings deduplicate it)."""
    sd = model.state_dict()
    if "lm_head.weight" in sd:
        weight = sd["lm_head.weight"].to(torch.float32)
    else:
        weight = model.lm_head.weight.detach().to(torch.float32)
    bias = sd.get("lm_head.bias")
    if bias is None:
        b = getattr(getattr(model, "lm_head", None), "bias", None)
        if b is not None:
            bias = b.detach()
    if bias is not None:
        bias = bias.to(torch.float32)
    return weight, bias


def extract_embed(model: AutoModelForCausalLM) -> torch.Tensor:
    """Return embed_tokens.weight (fp32). Handles flat layouts (Llama/Qwen3) and nested
    VLM wrappers (gemma4 ``model.language_model.embed_tokens``)."""
    sd = model.state_dict()
    for key in (
        "model.embed_tokens.weight",  # Llama, Qwen3, gemma2/3 text-only
        "model.language_model.embed_tokens.weight",  # gemma4 VLM
        "model.text_model.embed_tokens.weight",  # other wrapped layouts
        "language_model.model.embed_tokens.weight",
    ):
        if key in sd:
            return sd[key].to(torch.float32)
    inner = _get_text_inner(model)
    if hasattr(inner, "embed_tokens"):
        return inner.embed_tokens.weight.detach().to(torch.float32)
    raise KeyError(f"Could not locate embed_tokens.weight in {type(model).__name__}.")


def build_dlm_model(
    dflash_model_path: str,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None = None,
    embed_weight: torch.Tensor | None = None,
) -> AutoModelForCausalLM:
    """Load the DFlash draft and inject lm_head / embed_tokens from the base TLM.

    The draft checkpoint ships WITHOUT lm_head / embed_tokens (huge, identical to the
    base), so ``from_pretrained`` newly-initializes them (a "MISSING" load report — that
    is EXPECTED). We overwrite them with the real base weights. For a VLM base the head /
    embedding are nested and tied; ``extract_lm_head`` / ``extract_embed`` return those.
    Also strips fc / hidden_norm if the checkpoint carries them.
    """
    dlm_model = AutoModelForCausalLM.from_pretrained(dflash_model_path, torch_dtype=torch.float32)
    dlm_inner = _get_text_inner(dlm_model)

    lm_w = dlm_model.lm_head.weight
    if tuple(lm_head_weight.shape) != tuple(lm_w.shape):
        raise ValueError(f"lm_head shape mismatch: base {tuple(lm_head_weight.shape)} vs DLM {tuple(lm_w.shape)}.")
    if embed_weight is not None and tuple(embed_weight.shape) != tuple(dlm_inner.embed_tokens.weight.shape):
        raise ValueError(
            f"embed_tokens shape mismatch: base {tuple(embed_weight.shape)} "
            f"vs DLM {tuple(dlm_inner.embed_tokens.weight.shape)}."
        )

    with torch.no_grad():
        dlm_model.lm_head.weight.copy_(lm_head_weight)
        if embed_weight is not None:
            dlm_inner.embed_tokens.weight.copy_(embed_weight)

    if lm_head_bias is not None:
        if dlm_model.lm_head.bias is None:
            dlm_model.lm_head.bias = nn.Parameter(lm_head_bias)
        else:
            with torch.no_grad():
                dlm_model.lm_head.bias.copy_(lm_head_bias)

    # The draft config has tie_word_embeddings=True; make the tie explicit so a later
    # tie_weights() cannot silently undo one copy. For a VLM base lm_head == embed.
    if getattr(dlm_model.config, "tie_word_embeddings", False) and embed_weight is not None:
        with torch.no_grad():
            dlm_model.lm_head.weight.copy_(embed_weight)
        if not torch.equal(lm_head_weight, embed_weight):
            logger.warning("[DLM] tie_word_embeddings=True but base lm_head != embed_tokens; using embed_tokens.")

    for attr in ("fc", "hidden_norm"):
        if hasattr(dlm_model, attr):
            delattr(dlm_model, attr)
            logger.info(f"[DLM] Removed dlm_model.{attr}")

    # print (not logger.info) — proof the injected weights landed (logger defaults to WARNING).
    with torch.no_grad():
        lm_absmax = dlm_model.lm_head.weight.abs().max().item()
        emb_absmax = dlm_inner.embed_tokens.weight.abs().max().item()
    print(
        f"[DLM] lm_head injected     shape={tuple(dlm_model.lm_head.weight.shape)} absmax={lm_absmax:.4f}"
        + ("  ⚠ ZERO — injection may have failed!" if lm_absmax < 1e-6 else "  ✓")
    )
    print(
        f"[DLM] embed_tokens injected shape={tuple(dlm_inner.embed_tokens.weight.shape)} absmax={emb_absmax:.4f}"
        + ("  ⚠ ZERO — injection may have failed!" if emb_absmax < 1e-6 else "  ✓")
    )
    return dlm_model


def compile_gemma_vlm_dlm_qpc(
    tlm_repo: str,
    dlm_repo: str,
    *,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    prefill_seq_len: int = 128,
    target_seq_len: int | None = None,
    hf_token: str | None = None,
) -> str:
    """Build + compile the DFlash DLM (draft) for a vision-language model base.

    Unlike the text ``compile_dlm_qpc`` (which lets DFlashDLMTransform load lm_head/embed
    from ``dflash_tlm_repo`` via flat state-dict keys), a VLM base like gemma4 keeps
    lm_head/embed on a nested ``language_model`` submodule that the flat loader cannot
    find. So here we load the base VLM, extract those weights VLM-aware, and inject them
    into the draft explicitly.
    """
    from QEfficient import QEFFAutoModelForCausalLM

    _, _, block_size = read_dlm_meta(dlm_repo, hf_token)

    target_seq_len = get_dlm_target_seq_len(block_size, prefill_seq_len, target_seq_len)
    qaic_config = {"dflash_dlm": True}
    if target_seq_len != block_size:
        qaic_config["dflash_target_seq_len"] = target_seq_len

    logger.info(f"[compile_vlm_dlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    base_model = AutoModelForCausalLM.from_pretrained(tlm_repo, torch_dtype=torch.float32, token=hf_token)
    lm_head_w, lm_head_b = extract_lm_head(base_model)
    embed_w = extract_embed(base_model)
    del base_model

    dlm_model = build_dlm_model(dlm_repo, lm_head_w, lm_head_b, embed_w)
    dlm_qeff = QEFFAutoModelForCausalLM(dlm_model, qaic_config=qaic_config)
    qpc = dlm_qeff.compile(
        prefill_seq_len=block_size,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        prefill_only=True,
    )
    qpc = str(qpc)
    logger.info(f"[compile_vlm_dlm] qpc={qpc}")
    return qpc


def compile_qwen3vl_vlm_dlm_qpc(
    tlm_repo: str,
    dlm_repo: str,
    *,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    prefill_seq_len: int = 128,
    target_seq_len: int | None = None,
    hf_token: str | None = None,
) -> str:
    """Build + compile the DFlash DLM for a Qwen3-VL or Qwen3.5 TLM.

    Same VLM-aware lm_head/embed injection as ``compile_gemma_vlm_dlm_qpc``, kept as a
    separate function since qwen3-vl's config is not registered under
    ``AutoModelForCausalLM`` (it is a conditional-generation VLM config); loading the
    base TLM here goes through ``AutoModelForImageTextToText`` instead.
    """
    from QEfficient import QEFFAutoModelForCausalLM

    _, _, block_size = read_dlm_meta(dlm_repo, hf_token)

    target_seq_len = get_dlm_target_seq_len(block_size, prefill_seq_len, target_seq_len)
    qaic_config = {"dflash_dlm": True}
    if target_seq_len != block_size:
        qaic_config["dflash_target_seq_len"] = target_seq_len

    logger.info(f"[compile_qwen3vl_dlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    base_model = AutoModelForImageTextToText.from_pretrained(tlm_repo, torch_dtype=torch.float32, token=hf_token)
    lm_head_w, lm_head_b = extract_lm_head(base_model)
    embed_w = extract_embed(base_model)
    del base_model

    dlm_model = build_dlm_model(dlm_repo, lm_head_w, lm_head_b, embed_w)
    dlm_qeff = QEFFAutoModelForCausalLM(dlm_model, qaic_config=qaic_config)
    qpc = dlm_qeff.compile(
        prefill_seq_len=block_size,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        prefill_only=True,
    )
    qpc = str(qpc)
    logger.info(f"[compile_qwen3vl_dlm] qpc={qpc}")
    return qpc


def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(
            lambda x: {
                "formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x["input"] else x["instruction"])
            }
        )
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = [
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features,
        )

    return dataset


_DEFAULT_FMT = "{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
_CODING_FMT = (
    "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
)

_CATEGORY_FMT = {
    "math": _DEFAULT_FMT,
    "math_reasoning": _DEFAULT_FMT,
    "coding": _CODING_FMT,
    "reasoning": _DEFAULT_FMT,
    "stem": _DEFAULT_FMT,
    "qa": _DEFAULT_FMT,
    "rag": _DEFAULT_FMT,
    "extraction": _DEFAULT_FMT,
    "humanities": _DEFAULT_FMT,
    "writing": _DEFAULT_FMT,
    "summarization": _DEFAULT_FMT,
    "translation": _DEFAULT_FMT,
    "roleplay": _DEFAULT_FMT,
}


def format_prompt(prompt: str, category: str = "") -> str:
    fmt = _CATEGORY_FMT.get(category, _DEFAULT_FMT)
    return fmt.format(prompt=prompt)


def reformat_jsonl_by_category(questions: list) -> list:
    """Apply instruction prefix to JSONL questions based on their category.

    Mirrors the prompt_fmt logic in load_and_process_dataset: for categories
    where a specific instruction is obvious (math, coding) a tailored prefix is
    used; for all others the same step-by-step reasoning instruction is applied.
    """
    for q in questions:
        category = q.get("category", "")
        q["turns"][0] = format_prompt(q["turns"][0], category)
    return questions


def compile_dspark_qpcs(
    target_model,
    draft_model,
    *,
    output_dir,
    prefill_seq_len,
    ctx_len,
    target_cores,
    draft_cores,
    target_devices,
    draft_devices,
    draft_prefill_target_seq_len=None,
    vision=False,
    height=448,
    width=448,
    hf_token=None,
):
    require_model_cache()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QEFF_HOME", str(output_dir / "qeff"))
    draft_config = AutoConfig.from_pretrained(draft_model, token=hf_token)
    target_config = AutoConfig.from_pretrained(target_model, token=hf_token)
    target_text_config = getattr(target_config, "text_config", target_config)
    if draft_config.hidden_size != target_text_config.hidden_size:
        raise ValueError("Target and DSpark draft hidden dimensions differ.")
    target_layer_ids = [layer_id + 1 for layer_id in draft_config.target_layer_ids]
    block_size = int(draft_config.block_size)
    verify_width = block_size + 1
    if verify_width & (verify_width - 1):
        raise ValueError("DSpark requires block_size + 1 to be a power of two.")
    if prefill_seq_len < 2 * verify_width or prefill_seq_len % verify_width:
        raise ValueError("prefill_seq_len must be a multiple of block_size + 1 and at least twice that width.")

    target_qaic_config = {
        "dspark_tlm": True,
        "dspark_dlm_repo": draft_model,
        "target_layer_ids": target_layer_ids,
        "dspark_block_size": block_size,
    }
    target_model_type = target_config.model_type
    target_compile_kwargs = {
        "prefill_seq_len": prefill_seq_len,
        "ctx_len": ctx_len,
        "num_cores": target_cores,
        "num_devices": len(target_devices),
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "mos": 1,
        "dflash_block_size": verify_width,
    }
    if target_model_type == "qwen3":
        target = QEFFAutoModelForCausalLM.from_pretrained(
            target_model,
            config=target_config,
            torch_dtype=torch.float16,
            token=hf_token,
            qaic_config=target_qaic_config,
        )
        lang_qpc_path = str(target.compile(**target_compile_kwargs))
        vision_qpc_path = None
    else:
        target = QEFFAutoModelForImageTextToText.from_pretrained(
            target_model,
            config=target_config,
            torch_dtype=torch.float16,
            token=hf_token,
            kv_offload=True,
            qaic_config=target_qaic_config,
        )
        target_paths = target.compile(
            skip_vision=not vision,
            height=height,
            width=width,
            split_model_io=True,
            use_onnx_subfunctions="gemma4" in target_model_type,
            **target_compile_kwargs,
        )
        lang_qpc_path = str(target_paths.get("lang_qpc_path") or target_paths["lang_prefill_qpc_path"])
        vision_qpc_path = str(target_paths["vision_qpc_path"]) if vision else None

    draft_target_seq_len = draft_prefill_target_seq_len or prefill_seq_len
    draft_target_seq_lens = sorted({verify_width, draft_target_seq_len})
    draft = QEFFAutoModelForCausalLM.from_pretrained(
        draft_model,
        config=draft_config,
        torch_dtype=torch.float16,
        token=hf_token,
        qaic_config={
            "dspark_dlm": True,
            "dspark_tlm_repo": target_model,
            "dflash_target_seq_len": verify_width,
            "dspark_return_token_ids": True,
        },
    )
    draft_qpc_path = str(
        draft.compile(
            prefill_seq_len=block_size,
            ctx_len=ctx_len,
            num_cores=draft_cores,
            num_devices=len(draft_devices),
            mxfp6_matmul=True,
            mxint8_kv_cache=True,
            mos=1,
            prefill_only=True,
            dflash_block_size=block_size,
            specializations=[
                {
                    "batch_size": 1,
                    "seq_len": block_size,
                    "ctx_len": ctx_len,
                    "target_seq_len": target_seq_len,
                }
                for target_seq_len in draft_target_seq_lens
            ],
        )
    )
    manifest = {
        "target_model": target_model,
        "draft_model": draft_model,
        "block_size": block_size,
        "mask_token_id": int(draft_config.mask_token_id),
        "prefill_seq_len": prefill_seq_len,
        "ctx_len": ctx_len,
        "lang_qpc_path": lang_qpc_path,
        "vision_qpc_path": vision_qpc_path,
        "draft_qpc_path": draft_qpc_path,
        "target_devices": target_devices,
        "draft_devices": draft_devices,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


DEFAULT_DRAFT = "RadixArk/Qwen3.8-27B-DSpark"
DEFAULT_TARGET = "Qwen/Qwen3.8-27B"


def require_model_cache():
    if not os.environ.get("HF_HUB_CACHE"):
        raise ValueError("Set HF_HUB_CACHE to your model cache before loading checkpoints; --smoke-test is offline.")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def load_tokenizer(model_id):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.vocab_size > 1:
        return tokenizer
    model_path = Path(model_id)
    if not model_path.is_dir():
        repo_cache = Path(os.environ["HF_HUB_CACHE"]) / f"models--{model_id.replace('/', '--')}"
        ref_path = repo_cache / "refs" / "main"
        revision = (
            ref_path.read_text().strip() if ref_path.exists() else next((repo_cache / "snapshots").iterdir()).name
        )
        model_path = repo_cache / "snapshots" / revision
        if not (model_path / "tokenizer.json").exists():
            model_path = next(
                path for path in (repo_cache / "snapshots").iterdir() if (path / "tokenizer.json").exists()
            )
    tokenizer_config = json.loads((model_path / "tokenizer_config.json").read_text())
    return PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        bos_token=tokenizer_config.get("bos_token"),
        eos_token=tokenizer_config.get("eos_token"),
        pad_token=tokenizer_config.get("pad_token"),
        additional_special_tokens=tokenizer_config.get("additional_special_tokens"),
        chat_template=tokenizer_config.get("chat_template"),
        model_max_length=tokenizer_config.get("model_max_length", 262144),
    )


def format_chat_text(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def parse_devices(value):
    devices = [int(device.strip()) for device in value.split(",")]
    if not devices or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("Device IDs must be nonnegative integers.")
    return devices


def inference_parser(vision=False):
    parser = argparse.ArgumentParser(description="DSpark greedy speculation with hybrid-state verification/replay.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Check a random tiny model on CPU, without downloads.",
    )
    parser.add_argument("--manifest", type=Path, help="Compilation manifest produced by compile.py.")
    parser.add_argument("--metrics-output", type=Path, help="Save the reported metrics as a JSON file.")
    parser.add_argument("--prompt", default="Explain why speculative decoding preserves greedy output.")
    parser.add_argument("--generation-len", type=int, default=64)
    parser.add_argument("--target-devices", type=parse_devices, default=[0])
    parser.add_argument("--draft-devices", type=parse_devices, default=[0])
    if vision:
        parser.add_argument(
            "--image",
            type=Path,
            help="Local image; required unless --smoke-test is used.",
        )
        parser.add_argument("--vision-devices", type=parse_devices, default=[0])
    return parser


def report_metrics(metrics, output_path=None):
    serialized = json.dumps(metrics, indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n")
    print(serialized)


def make_hybrid_cache(model, config, ctx_len):
    cache = []
    for layer_index, layer_type in enumerate(config.layer_types):
        if layer_type == "full_attention":
            shape = (1, config.num_key_value_heads, ctx_len, config.head_dim)
            cache.append((torch.zeros(shape), torch.zeros(shape)))
        else:
            layer = model.layers[layer_index].linear_attn
            conv_shape = (
                1,
                layer.num_k_heads,
                layer.conv_dim // layer.num_k_heads,
                layer.conv_kernel_size,
            )
            recurrent_shape = (1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
            cache.append((torch.zeros(conv_shape), torch.zeros(recurrent_shape)))
    return tuple(cache)


class TorchTarget:
    def __init__(self, wrapper, ctx_len, vision_embeds=None):
        self.wrapper = wrapper
        self.vision_embeds = vision_embeds
        self.image_index = torch.zeros((1, 1), dtype=torch.long)
        if vision_embeds is None:
            self.model = wrapper.model
            inner = self.model.model
            config = self.model.config
        else:
            self.model = wrapper.lang_model.model
            inner = wrapper.model.model.language_model
            config = wrapper.model.config.text_config
        self.cache = make_hybrid_cache(inner, config, ctx_len)
        self.fused_replay = config.model_type == "qwen3_5_text"

    @torch.no_grad()
    def __call__(self, input_ids, position_ids, commit, num_logits):
        if self.vision_embeds is None:
            position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
        inputs = {
            "input_ids": torch.from_numpy(input_ids),
            "position_ids": torch.from_numpy(position_ids),
            "past_key_values": self.cache,
            "cache_commit": torch.tensor([int(commit)]),
            "num_logits_to_keep": torch.zeros((num_logits, 1), dtype=torch.long),
        }
        if self.vision_embeds is not None:
            inputs.update(vision_embeds=self.vision_embeds, image_idx=self.image_index)
            predictions, _, image_index, features, cache = self.model(**inputs)
            if commit:
                self.image_index = image_index
        else:
            outputs = self.model(**inputs)
            predictions, features, cache = (
                outputs.logits,
                outputs.hidden_states,
                outputs.past_key_values,
            )
        self.cache = cache
        if predictions.ndim == 3:
            predictions = predictions.argmax(dim=-1)
        return predictions.numpy().copy(), features.numpy().copy()


class TorchDraft:
    def __init__(self, model, ctx_len):
        self.model = model
        config = model.config
        self.target_seq_len = model.model.block_size + 1
        self.target_seq_lens = [self.target_seq_len]
        shape = (1, config.num_key_value_heads, ctx_len, config.head_dim)
        self.cache = tuple((torch.zeros(shape), torch.zeros(shape)) for _ in range(config.num_hidden_layers))

    @torch.no_grad()
    def __call__(self, input_ids, features, feature_positions, noise_positions, seed):
        outputs = self.model(
            input_ids=torch.from_numpy(input_ids),
            target_hidden=torch.from_numpy(features),
            position_ids_target=torch.from_numpy(feature_positions),
            position_ids=torch.from_numpy(noise_positions),
            prev_token_ids=torch.tensor([[seed]]),
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        predictions = outputs.logits
        if predictions.ndim == 3:
            predictions = predictions.argmax(dim=-1)
        return predictions.numpy().copy()


class DeviceTarget:
    def __init__(self, session, vision_embeds):
        self.session = session
        self.vision_embeds = vision_embeds
        self.image_index = np.zeros((1, 1), dtype=np.int64)
        required = {"num_logits_to_keep", "position_ids", "input_ids"}
        if not required.issubset(session.input_names) or "hidden_states" not in session.output_names:
            raise ValueError("The target QPC lacks the DSpark verification contract; recompile it.")
        self.fused_replay = "cache_commit" in session.input_names
        position_binding = session.bindings[session.binding_index_map["position_ids"]]
        self.position_rank = len(position_binding.dims)
        retained = [name for name in session.input_names + session.output_names if is_cache_name(name)]
        if "vision_embeds_RetainedState" in session.input_names + session.output_names:
            retained.append("vision_embeds_RetainedState")
        session.skip_buffers(retained)

    def decode_positions(self, start, width, rope_delta=0):
        flat = np.arange(start, start + width, dtype=np.int64)[None, :]
        if self.position_rank == 2:
            return flat
        return np.concatenate((flat[None], np.repeat((flat + rope_delta)[None], 3, axis=0)), axis=0)

    def __call__(self, input_ids, position_ids, commit, num_logits):
        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "num_logits_to_keep": np.zeros((num_logits, 1), dtype=np.int64),
        }
        if self.fused_replay:
            inputs["cache_commit"] = np.array([int(commit)], dtype=np.int64)
        if "image_idx" in self.session.input_names:
            inputs["image_idx"] = self.image_index
        if self.vision_embeds is not None:
            inputs["vision_embeds"] = self.vision_embeds
        set_dynamic_output_buffers(self.session, inputs)
        outputs = run_session(self.session, inputs)
        if self.vision_embeds is not None:
            self.session.skip_buffers(["vision_embeds"])
            self.vision_embeds = None
        if commit and "image_idx_output" in outputs:
            self.image_index = outputs["image_idx_output"].copy()
        predictions = outputs["logits"]
        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)
        return predictions.copy(), outputs["hidden_states"].copy()


class DeviceDraft:
    def __init__(self, session):
        self.session = session
        if "prev_token_ids" not in session.input_names:
            raise ValueError("The draft QPC lacks DSpark's Markov seed input; recompile it.")
        if "draft_tokens" in session.output_names:
            self.output_name = "draft_tokens"
        elif "logits" in session.output_names:
            self.output_name = "logits"
        else:
            raise ValueError("The draft QPC lacks DSpark token or logits output; recompile it.")
        target_binding_index = session.binding_index_map["target_hidden"]
        target_shapes = [shape[target_binding_index][1] for shape in session.allowed_shapes]
        target_shape = target_shapes[0] if target_shapes else session.bindings[target_binding_index].dims
        self.target_seq_lens = sorted({int(shape[1]) for shape in target_shapes}) or [int(target_shape[1])]
        self.target_seq_len = self.target_seq_lens[0]
        session.skip_buffers([name for name in session.input_names + session.output_names if is_cache_name(name)])

    def __call__(self, input_ids, features, feature_positions, noise_positions, seed):
        outputs = run_session(
            self.session,
            {
                "input_ids": input_ids,
                "target_hidden": features,
                "position_ids_target": feature_positions,
                "position_ids": noise_positions,
                "prev_token_ids": np.array([[seed]], dtype=np.int64),
            },
        )
        predictions = outputs[self.output_name]
        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)
        return predictions.copy()


def is_cache_name(name):
    return name.startswith(("past_key.", "past_value.", "conv_state.", "recurrent_state."))


def run_session(session, inputs):
    converted = {}
    for name, value in inputs.items():
        if name in session.input_names:
            binding = session.bindings[session.binding_index_map[name]]
            dtype = session.aic_to_np_dtype_mapping[binding.type]
            converted[name] = np.ascontiguousarray(value, dtype=dtype)
    return session.run(converted)


def set_dynamic_output_buffers(session, inputs):
    if not session.allowed_shapes:
        return
    matching_shapes = []
    for allowed_shape in session.allowed_shapes:
        if all(
            tuple(allowed_shape[session.binding_index_map[name]][1]) == np.asarray(value).shape
            for name, value in inputs.items()
            if name in session.binding_index_map
        ):
            matching_shapes.append(allowed_shape)
    if len(matching_shapes) != 1:
        raise ValueError(f"Expected one QAIC specialization for input shapes, found {len(matching_shapes)}.")
    shape_key = tuple(
        sorted(
            (name, tuple(np.asarray(value).shape))
            for name, value in inputs.items()
            if name in session.binding_index_map
        )
    )
    if getattr(session, "_dspark_output_shape_key", None) == shape_key:
        return
    output_buffer_cache = getattr(session, "_dspark_output_buffer_cache", {})
    output_buffers = output_buffer_cache.get(shape_key)
    if output_buffers is not None:
        session.set_buffers(output_buffers)
        session._dspark_output_shape_key = shape_key
        return
    output_buffers = {}
    for output_name in session.output_names:
        basename = output_name.rsplit("/", 1)[-1]
        if basename.endswith("RetainedState"):
            continue
        binding_index = session.binding_index_map[output_name]
        binding = session.bindings[binding_index]
        output_shape = matching_shapes[0][binding_index][1]
        output_dtype = session.aic_to_np_dtype_mapping[binding.type]
        output_buffers[output_name] = np.empty(output_shape, dtype=output_dtype)
    output_buffer_cache[shape_key] = output_buffers
    session._dspark_output_buffer_cache = output_buffer_cache
    session._dspark_output_shape_key = shape_key
    session.set_buffers(output_buffers)


def prepare_vision_embeds(session, vision_embeds):
    binding_index = session.binding_index_map.get("vision_embeds")
    if binding_index is None:
        if vision_embeds is not None:
            raise ValueError("The target QPC has no vision_embeds input; recompile it with --vision.")
        return None
    shapes = [shape[binding_index][1] for shape in session.allowed_shapes]
    vision_shape = tuple(shapes[0] if shapes else session.bindings[binding_index].dims)
    padded_vision = np.zeros(vision_shape, dtype=np.float32)
    if vision_embeds is not None:
        if vision_embeds.shape[1] > vision_shape[1]:
            raise ValueError("Image token count exceeds the compiled vision size; recompile for this resolution.")
        padded_vision[:, : vision_embeds.shape[1]] = vision_embeds
    return padded_vision


def decode_positions(start, width, rope_delta=0):
    flat = np.arange(start, start + width, dtype=np.int64)[None, :]
    return np.concatenate((flat[None], np.repeat((flat + rope_delta)[None], 3, axis=0)), axis=0)


def run_dspark(
    target,
    draft,
    input_ids,
    position_ids,
    *,
    block_size,
    mask_token_id,
    prefill_seq_len,
    ctx_len,
    generation_len,
    rope_delta=0,
    eos_token_ids=(),
):
    verify_width = block_size + 1
    if verify_width & (verify_width - 1) or prefill_seq_len < 2 * verify_width or prefill_seq_len % verify_width:
        raise ValueError(
            "Verification width must be a power of two; prefill width must be a multiple at least twice as large."
        )
    prompt_len = input_ids.shape[1]
    if input_ids.shape[0] != 1 or prompt_len < 1 or generation_len < 1:
        raise ValueError("DSpark requires a nonempty single prompt and positive generation length.")
    if prompt_len + generation_len + verify_width > ctx_len:
        raise ValueError("ctx_len must reserve the prompt, requested output, and a complete verification block.")
    start_time = time.perf_counter()
    prompt_features = []
    for offset in range(0, prompt_len, prefill_seq_len):
        count = min(prefill_seq_len, prompt_len - offset)
        tokens = np.zeros((1, prefill_seq_len), dtype=np.int64)
        positions = np.full((*position_ids.shape[:-1], prefill_seq_len), -1, dtype=np.int64)
        tokens[:, :count] = input_ids[:, offset : offset + count]
        positions[..., :count] = position_ids[..., offset : offset + count]
        predictions, features = target(tokens, positions, True, 1)
        prompt_features.append(features[:, :count].copy())
    seed = int(predictions[0, -1])
    features = np.concatenate(prompt_features, axis=1)
    feature_start = 0
    committed_len = prompt_len
    generated = []
    accepted_lengths = []
    matched_lengths = []
    target_calls = len(prompt_features)
    draft_calls = 0
    draft_seconds = 0.0
    verify_seconds = 0.0
    fused_replay = getattr(target, "fused_replay", True)
    replay_tokens = np.zeros((1, verify_width), dtype=np.int64)
    replay_positions = np.full((*position_ids.shape[:-1], verify_width), -1, dtype=np.int64)
    fallback_target_seq_len = getattr(draft, "target_seq_len", block_size)
    draft_target_seq_lens = sorted(set(getattr(draft, "target_seq_lens", [fallback_target_seq_len])))
    if not draft_target_seq_lens or draft_target_seq_lens[0] < 1:
        raise ValueError("Draft target feature widths must be positive.")
    prefill_time = time.perf_counter() - start_time
    while len(generated) < generation_len:
        noise = np.full((1, block_size), mask_token_id, dtype=np.int64)
        noise[:, 0] = seed
        noise_positions = np.arange(committed_len, committed_len + block_size, dtype=np.int64)[None, :]
        draft_target_seq_len = next(
            (width for width in draft_target_seq_lens if width >= features.shape[1]),
            draft_target_seq_lens[-1],
        )
        for offset in range(0, features.shape[1], draft_target_seq_len):
            count = min(draft_target_seq_len, features.shape[1] - offset)
            update = np.zeros((1, draft_target_seq_len, features.shape[-1]), dtype=features.dtype)
            update[:, :count] = features[:, offset : offset + count]
            update_positions = np.full((1, draft_target_seq_len), -1, dtype=np.int64)
            update_positions[:, :count] = np.arange(feature_start + offset, feature_start + offset + count)
            call_start = time.perf_counter()
            proposals = draft(noise, update, update_positions, noise_positions, seed)
            draft_seconds += time.perf_counter() - call_start
            draft_calls += 1
        candidates = np.concatenate((np.array([[seed]], dtype=np.int64), proposals), axis=1)
        positions = (
            target.decode_positions(committed_len, verify_width, rope_delta)
            if hasattr(target, "decode_positions")
            else decode_positions(committed_len, verify_width, rope_delta)
        )
        call_start = time.perf_counter()
        target_tokens = np.concatenate((replay_tokens, candidates), axis=1) if fused_replay else candidates
        target_positions = np.concatenate((replay_positions, positions), axis=-1) if fused_replay else positions
        predictions, verified_features = target(target_tokens, target_positions, False, verify_width)
        verify_seconds += time.perf_counter() - call_start
        target_calls += 1
        matches = proposals[0] == predictions[0, :block_size]
        mismatches = np.flatnonzero(~matches)
        accepted = int(mismatches[0]) if mismatches.size else block_size
        count = min(accepted + 1, generation_len - len(generated))
        emitted = candidates[0, :count].tolist()
        eos_offsets = [index for index, token in enumerate(emitted) if token in eos_token_ids]
        if eos_offsets:
            count = eos_offsets[0] + 1
            emitted = emitted[:count]
        current_features = verified_features[:, -verify_width:] if fused_replay else verified_features
        features = current_features[:, :count].copy()
        next_seed = int(predictions[0, accepted])
        if fused_replay:
            replay_tokens = candidates.copy()
            replay_positions = positions.copy()
            replay_positions[..., count:] = -1
        generated.extend(emitted)
        accepted_lengths.append(count - 1)
        matched_lengths.append(accepted)
        feature_start = committed_len
        committed_len += count
        seed = next_seed
        if eos_offsets:
            break
    elapsed = time.perf_counter() - start_time
    rounds = len(accepted_lengths)
    target_seconds = verify_seconds
    target_round_rate = rounds / max(target_seconds, 1e-9)
    decode_speed = len(generated) / max(elapsed - prefill_time, 1e-9)
    return {
        "token_ids": generated,
        "prefill_seconds": prefill_time,
        "elapsed_seconds": elapsed,
        "decode_tokens_per_second": decode_speed,
        "spd_tokens_per_second": decode_speed,
        "acceptance_rate": sum(accepted_lengths) / rounds,
        "draft_acceptance_fraction": sum(matched_lengths) / (block_size * rounds),
        "accepted_draft_tokens": accepted_lengths,
        "matched_draft_tokens": matched_lengths,
        "generated_tokens": len(generated),
        "speculation_rounds": rounds,
        "tlm_decode_seconds": target_seconds,
        "tlm_verify_seconds": verify_seconds,
        "tlm_replay_seconds": 0.0,
        "dlm_decode_seconds": draft_seconds,
        "tlm_tokens_per_second": target_round_rate,
        "tlm_rounds_per_second": target_round_rate,
        "tlm_verify_calls_per_second": rounds / max(verify_seconds, 1e-9),
        "tlm_fused_calls_per_second": rounds / max(verify_seconds, 1e-9),
        "tlm_replay_calls_per_second": 0.0,
        "tlm_verified_positions_per_second": verify_width * rounds / max(verify_seconds, 1e-9),
        "dlm_tokens_per_second": (block_size * rounds) / max(draft_seconds, 1e-9),
        "target_calls": target_calls,
        "draft_calls": draft_calls,
    }


def make_dspark_smoke_models(vision=False):
    torch.manual_seed(42)
    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        layer_types=["full_attention", "linear_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 1, 1],
        },
        dtype="float32",
    )
    qaic_config = {
        "dspark_tlm": True,
        "target_layer_ids": [1, 2],
        "dspark_block_size": 3,
    }
    if vision:
        vision_config = Qwen3_5VisionConfig(
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
        config = Qwen3_5Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=3,
            video_token_id=4,
            vision_start_token_id=5,
            vision_end_token_id=6,
        )
        hf_model = Qwen3_5ForConditionalGeneration(config).eval()
        reference = deepcopy(hf_model)
        target = QEFFAutoModelForImageTextToText(hf_model, kv_offload=True, qaic_config=qaic_config)
        embeddings = hf_model.model.get_input_embeddings()
    else:
        hf_model = Qwen3_5ForCausalLM(text_config).eval()
        reference = deepcopy(hf_model)
        target = QEFFAutoModelForCausalLM(hf_model, qaic_config=qaic_config)
        embeddings = hf_model.model.embed_tokens
    draft_config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        block_size=3,
        target_layer_ids=[0, 1],
        target_hidden_size=16,
        mask_token_id=63,
        markov_rank=8,
        markov_head_type="vanilla",
        projector_type="dspark",
        dtype="float32",
    )
    draft = QEffDSparkForCausalLM(draft_config, embed_tokens=embeddings, lm_head=hf_model.lm_head).eval()
    return (
        target,
        QEFFAutoModelForCausalLM(
            draft,
            qaic_config={
                "dspark_dlm": True,
                "dflash_target_seq_len": draft_config.block_size + 1,
                "dspark_return_token_ids": True,
            },
        ),
        reference,
    )


@torch.no_grad()
def dspark_smoke_test(vision=False, generation_len=12, draft_target_seq_lens=None):
    target, draft, reference = make_dspark_smoke_models(vision)
    prompt = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15, 16]])
    vision_embeds = None
    if vision:
        pixels = torch.randn(4, 48)
        grid = torch.zeros((1, 1, 2, 2), dtype=torch.long)
        vision_embeds = target.vision_model.model(pixels, grid)
        expected = reference.model.visual(pixels, grid_thw=torch.tensor([[1, 2, 2]]))
        expected = expected[0] if isinstance(expected, tuple) else expected.pooler_output
        torch.testing.assert_close(
            vision_embeds.reshape(-1, 16),
            expected.reshape(-1, 16),
            atol=1e-5,
            rtol=1e-5,
        )
        prompt[:, 1:5] = 3
    torch_draft = TorchDraft(draft.model, 128)
    if draft_target_seq_lens is not None:
        torch_draft.target_seq_lens = sorted(set(draft_target_seq_lens))
        torch_draft.target_seq_len = torch_draft.target_seq_lens[0]
    metrics = run_dspark(
        TorchTarget(target, 128, vision_embeds),
        torch_draft,
        prompt.numpy(),
        decode_positions(0, prompt.shape[1]),
        block_size=3,
        mask_token_id=63,
        prefill_seq_len=8,
        ctx_len=128,
        generation_len=generation_len,
    )
    reference_ids = prompt.clone()
    expected_tokens = []
    for _ in range(generation_len):
        if vision:
            embeds = reference.model.get_input_embeddings()(reference_ids)
            embeds[:, 1:5] = vision_embeds
            outputs = reference.model.language_model(inputs_embeds=embeds, use_cache=False)
            logits = reference.lm_head(outputs.last_hidden_state)
        else:
            logits = reference(input_ids=reference_ids, use_cache=False).logits
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        expected_tokens.append(int(next_token))
        reference_ids = torch.cat((reference_ids, next_token), dim=1)
    if metrics["token_ids"] != expected_tokens:
        raise AssertionError(f"DSpark differs from HF greedy decoding: {metrics['token_ids']} != {expected_tokens}")
    metrics.update(mode="tiny_cpu_vision" if vision else "tiny_cpu_text", hf_greedy_parity=True)
    return metrics


HUMANEVAL_PROMPT = "Write a solution to the following problem and make sure that it passes the tests:\n{prompt}"


def aggregate_dspark_metrics(rows, block_size):
    rounds = sum(row["speculation_rounds"] for row in rows)
    generated = sum(row["generated_tokens"] for row in rows)
    accepted = sum(sum(row["accepted_draft_tokens"]) for row in rows)
    matched = sum(sum(row["matched_draft_tokens"]) for row in rows)
    target_seconds = sum(row["tlm_decode_seconds"] for row in rows)
    verify_seconds = sum(row["tlm_verify_seconds"] for row in rows)
    draft_seconds = sum(row["dlm_decode_seconds"] for row in rows)
    decode_seconds = sum(row["elapsed_seconds"] - row["prefill_seconds"] for row in rows)
    return {
        "dataset": "humaneval",
        "samples": len(rows),
        "generated_tokens": generated,
        "speculation_rounds": rounds,
        "acceptance_length": generated / rounds,
        "acceptance_rate": accepted / rounds,
        "draft_acceptance_fraction": matched / (block_size * rounds),
        "tlm_tokens_per_second": rounds / target_seconds,
        "tlm_rounds_per_second": rounds / target_seconds,
        "tlm_verify_calls_per_second": rounds / verify_seconds,
        "tlm_verified_positions_per_second": (block_size + 1) * rounds / verify_seconds,
        "dlm_tokens_per_second": block_size * rounds / draft_seconds,
        "spd_tokens_per_second": generated / decode_seconds,
        "decode_seconds": decode_seconds,
    }


def run_dspark_humaneval(
    manifest_path,
    output_dir,
    *,
    target_devices,
    draft_devices,
    max_new_tokens=2048,
    limit=164,
    dataset_file=None,
):
    require_model_cache()
    manifest = json.loads(Path(manifest_path).read_text())
    if dataset_file is None:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompts = [HUMANEVAL_PROMPT.format(prompt=record["prompt"]) for record in dataset]
    else:
        records = [json.loads(line) for line in Path(dataset_file).read_text().splitlines() if line.strip()]
        prompts = [record["turns"][0] for record in records]
    prompts = prompts[:limit]
    if len(prompts) != limit:
        raise ValueError(f"Requested {limit} HumanEval prompts, found {len(prompts)}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "samples.jsonl"
    completed = {}
    if rows_path.exists():
        completed = {row["sample_index"]: row for row in map(json.loads, rows_path.read_text().splitlines())}

    tokenizer = load_tokenizer(manifest["target_model"])
    config = AutoConfig.from_pretrained(manifest["target_model"])
    text_config = getattr(config, "text_config", config)
    eos_ids = text_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    target_session = QAICInferenceSession(manifest["lang_qpc_path"], target_devices)
    draft_session = QAICInferenceSession(manifest["draft_qpc_path"], draft_devices)
    target = DeviceTarget(target_session, None)
    draft = DeviceDraft(draft_session)
    started = time.perf_counter()
    try:
        for sample_index, prompt in enumerate(prompts):
            if sample_index in completed:
                continue
            chat_text = format_chat_text(tokenizer, prompt)
            input_ids = tokenizer(chat_text, return_tensors="np")["input_ids"]
            if input_ids.shape[1] + max_new_tokens + manifest["block_size"] + 1 > manifest["ctx_len"]:
                raise ValueError(f"HumanEval sample {sample_index} exceeds the compiled context length.")
            metrics = run_dspark(
                target,
                draft,
                input_ids,
                target.decode_positions(0, input_ids.shape[1]),
                block_size=manifest["block_size"],
                mask_token_id=manifest["mask_token_id"],
                prefill_seq_len=manifest["prefill_seq_len"],
                ctx_len=manifest["ctx_len"],
                generation_len=max_new_tokens,
                eos_token_ids=eos_ids,
            )
            metrics.update(
                sample_index=sample_index,
                prompt_tokens=int(input_ids.shape[1]),
                acceptance_length=metrics["generated_tokens"] / metrics["speculation_rounds"],
            )
            with rows_path.open("a") as output:
                output.write(json.dumps(metrics) + "\n")
            completed[sample_index] = metrics
            print(
                f"[{sample_index + 1}/{len(prompts)}] "
                f"acceptance_length={metrics['acceptance_length']:.4f} "
                f"TLM={metrics['tlm_rounds_per_second']:.4f} "
                f"DLM={metrics['dlm_tokens_per_second']:.4f} "
                f"SPD={metrics['spd_tokens_per_second']:.4f}",
                flush=True,
            )
    finally:
        draft_session.deactivate()
        target_session.deactivate()

    rows = [completed[index] for index in range(len(prompts))]
    summary = aggregate_dspark_metrics(rows, manifest["block_size"])
    summary.update(max_new_tokens=max_new_tokens, wall_time_seconds=time.perf_counter() - started)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def run_dspark_device_inference(args, vision=False):
    from transformers import AutoConfig, AutoProcessor

    from QEfficient.generation.cloud_infer import QAICInferenceSession

    require_model_cache()
    if args.manifest is None:
        raise ValueError("--manifest is required for QAIC inference.")
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("lang_qpc_path") or not manifest.get("draft_qpc_path"):
        raise ValueError("The manifest has no compiled target/draft QPCs; rerun compile.py without --export-only.")
    tokenizer = load_tokenizer(manifest["target_model"])
    config = AutoConfig.from_pretrained(manifest["target_model"])
    vision_embeds = None
    rope_delta = 0
    if vision:
        from PIL import Image

        if args.image is None or not manifest.get("vision_qpc_path"):
            raise ValueError("Vision inference requires --image and a manifest compiled with --vision.")
        processor = AutoProcessor.from_pretrained(manifest["target_model"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(args.image)},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(args.image) as image:
            inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
        if inputs["image_grid_thw"].shape[0] != 1:
            raise ValueError("Only one image per request is currently supported.")
        with torch.device("meta"):
            rope_model = Qwen3_5ForConditionalGeneration(config)
        rotary, deltas = rope_model.model.get_rope_index(
            inputs["input_ids"],
            inputs["mm_token_type_ids"],
            inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
        input_ids = inputs["input_ids"].numpy()
        position_ids = np.concatenate((np.arange(input_ids.shape[1])[None, None, :], rotary.numpy()), axis=0)
        rope_delta = int(deltas.reshape(-1)[0])
        vision_session = QAICInferenceSession(manifest["vision_qpc_path"], args.vision_devices)
        grid_time, grid_height, grid_width = inputs["image_grid_thw"][0].tolist()
        grid = np.zeros((1, grid_time, grid_height, grid_width), dtype=np.int64)
        vision_embeds = run_session(
            vision_session,
            {"pixel_values": inputs["pixel_values"].numpy(), "image_grid_thw": grid},
        )["vision_embeds"].copy()
        vision_session.deactivate()
    else:
        text = format_chat_text(tokenizer, args.prompt)
        input_ids = tokenizer(text, return_tensors="np")["input_ids"]
        position_ids = (
            decode_positions(0, input_ids.shape[1])
            if getattr(config, "model_type", "") == "qwen3_5"
            else np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
        )
    target_session = QAICInferenceSession(manifest["lang_qpc_path"], args.target_devices)
    draft_session = QAICInferenceSession(manifest["draft_qpc_path"], args.draft_devices)
    padded_vision = prepare_vision_embeds(target_session, vision_embeds)
    text_config = getattr(config, "text_config", config)
    eos_ids = text_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    metrics = run_dspark(
        DeviceTarget(target_session, padded_vision),
        DeviceDraft(draft_session),
        input_ids,
        position_ids,
        block_size=manifest["block_size"],
        mask_token_id=manifest["mask_token_id"],
        prefill_seq_len=manifest["prefill_seq_len"],
        ctx_len=manifest["ctx_len"],
        generation_len=args.generation_len,
        rope_delta=rope_delta,
        eos_token_ids=eos_ids,
    )
    metrics["text"] = tokenizer.decode(metrics["token_ids"], skip_special_tokens=True)
    metrics["mode"] = "qaic_vision" if vision else "qaic_text"
    return metrics


def run_dspark_cli(args, *, vision=False, default_prompt=None):
    if args.smoke_test:
        report_metrics(
            dspark_smoke_test(vision=vision, generation_len=args.generation_len),
            args.metrics_output,
        )
        return

    target_model, draft_model = MODEL_MAP[args.model_name]
    if not args.model_name.endswith("-DSpark"):
        raise ValueError("Select a *-DSpark model_name when --technique dspark is used.")
    if vision:
        if args.image_path is None:
            raise ValueError("DSpark vision inference requires --image-path.")
        if not target_model.startswith("Qwen/Qwen3.8"):
            raise NotImplementedError("The DSpark vision runner currently supports the Qwen3.8 multimodal target.")

    output_dir = (
        args.dspark_output_dir
        or Path(os.environ.get("QEFF_HOME", "dspark-artifacts")) / "dflash-dspark" / args.model_name
    )
    if args.manifest is None:
        args.manifest = compile_dspark_qpcs(
            target_model,
            draft_model,
            output_dir=output_dir,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            target_cores=args.tlm_cores,
            draft_cores=args.dlm_cores,
            target_devices=args.tlm_devices,
            draft_devices=args.dlm_devices,
            draft_prefill_target_seq_len=args.draft_prefill_target_seq_len,
            vision=vision,
            height=getattr(args, "height", None) or 448,
            width=getattr(args, "width", None) or 448,
            hf_token=args.hf_token,
        )

    args.target_devices = args.tlm_devices
    args.draft_devices = args.dlm_devices
    if getattr(args, "dataset", "single") == "humaneval":
        benchmark_dir = args.benchmark_output_dir or output_dir / "humaneval"
        run_dspark_humaneval(
            args.manifest,
            benchmark_dir,
            target_devices=args.target_devices,
            draft_devices=args.draft_devices,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            dataset_file=args.dataset_file,
        )
        return

    if vision:
        args.prompt = args.image_prompt or args.prompt or default_prompt
        args.image = args.image_path
    elif not args.prompt:
        raise ValueError("--prompt is required for single-prompt inference.")
    report_metrics(run_dspark_device_inference(args, vision=vision), args.metrics_output)
