# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import json
import os

import torch
from datasets import Features, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import (
    QAICInferenceSession,
    is_retained_state_name,
)
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")
# ─────────────────────────────────────────────────────────────────────────────
# model_name (TLM short)  →  (TLM HF repo, DLM HF repo)
# DLM column is the canonical DFlash repo. TLM column is the standard HF repo
# when known; otherwise None and must be supplied via --tlm_hf_path.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_MAP = {
    "gemma-4-31B-it": ("google/gemma-4-31B-it", "z-lab/gemma-4-31B-it-DFlash"),
    "Qwen3-4B": ("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"),
    "Llama-3.1-8B-Instruct": ("meta-llama/Llama-3.1-8B-Instruct", "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"),
    "Qwen3-VL-32B-Instruct": (
        "Qwen/Qwen3-VL-32B-Instruct",
        "path_qwen3_vl_model",
    ),
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


def load_spd_sessions(tlm_qpc: str, dlm_qpc: str, tlm_devices: list[int], dlm_devices: list[int]):
    """Create DFlash sessions and enable retained-state cache handling."""
    dlm_session = QAICInferenceSession(dlm_qpc, dlm_devices)
    tlm_session = QAICInferenceSession(tlm_qpc, tlm_devices)
    for session in (dlm_session, tlm_session):
        session.skip_buffers(
            [
                name
                for name in session.input_names + session.output_names
                if is_retained_state_name(name) or name.endswith("_RetainedState")
            ]
        )
    return dlm_session, tlm_session


def get_spd_prompt_chunk_size(tlm_session: QAICInferenceSession) -> int:
    """Return the largest compiled input sequence length used for TLM prefill."""
    input_index = tlm_session.binding_index_map["input_ids"]
    return max(
        [shape[input_index][1][1] for shape in tlm_session.allowed_shapes] + [tlm_session.bindings[input_index].dims[1]]
    )


def validate_spd_decode_specialization(tlm_session: QAICInferenceSession, block_size: int) -> None:
    """Require a TLM specialization that verifies one complete DFlash block."""
    input_index = tlm_session.binding_index_map["input_ids"]
    sequence_lengths = sorted({shape[input_index][1][1] for shape in tlm_session.allowed_shapes})
    logger.info(f"TLM input_ids seq_lens (allowed shapes): {sequence_lengths}")
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


def read_dlm_meta(dlm_repo: str, hf_token: str | None = None):
    """Load a DFlash checkpoint and return (state_dict, target_layer_ids, block_size)."""
    state_dict, cfg = load_dflash_checkpoint(dlm_repo)
    target_layer_ids = cfg.get("dflash_config", {}).get("target_layer_ids", [])
    block_size = cfg.get("block_size", None)
    return state_dict, target_layer_ids, block_size


def compile_tlm_qpc(
    tlm_repo: str,
    dlm_repo: str,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    hf_token: str | None = None,
) -> str:
    """Build the TLM and compile it to a QPC. fc/hidden_norm are injected by
    QEfficient's DFlashTLMTransform (weights from dflash_dlm_repo)."""
    from QEfficient import QEFFAutoModelForCausalLM

    _, target_layer_ids, block_size = read_dlm_meta(dlm_repo, hf_token)
    tlm_target_ids = [i + 1 for i in target_layer_ids]

    logger.info(f"[compile_tlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    tlm_qeff = QEFFAutoModelForCausalLM.from_pretrained(
        tlm_repo,
        torch_dtype=torch.float32,
        token=hf_token,
        qaic_config={"target_layer_ids": tlm_target_ids, "dflash_dlm_repo": dlm_repo},
    )
    qpc = tlm_qeff.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        dflash_block_size=block_size,
    )
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
    hf_token: str | None = None,
) -> str:
    """Build the DLM and compile it to a QPC. lm_head/embed_tokens are injected
    by QEfficient's DFlashDLMTransform (weights from dflash_tlm_repo)."""
    from QEfficient import QEFFAutoModelForCausalLM

    _, _, block_size = read_dlm_meta(dlm_repo, hf_token)

    logger.info(f"[compile_dlm] dlm={dlm_repo}  block_size={block_size}")
    dlm_qeff = QEFFAutoModelForCausalLM.from_pretrained(
        dlm_repo,
        torch_dtype=torch.float32,
        token=hf_token,
        qaic_config={"dflash_dlm": True, "dflash_tlm_repo": tlm_repo},
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
        qaic_config={"target_layer_ids": tlm_target_ids},
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
        use_onnx_subfunctions=False,
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
) -> tuple[str, str]:
    """Build the qwen3-vl TLM — vision encoder + language decoder — and compile it for
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
    tlm_qeff = QEFFAutoModelForImageTextToText.from_pretrained(
        tlm_repo,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,
        ignore_mismatched_sizes=True,
        token=hf_token,
        qaic_config={"target_layer_ids": tlm_target_ids},
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
    if height is not None:
        compile_kwargs["height"] = height
    if width is not None:
        compile_kwargs["width"] = width

    qpc = tlm_qeff.compile(**compile_kwargs)
    lang_qpc = str(qpc["lang_qpc_path"])
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


def extract_lm_head(model: AutoModelForCausalLM) -> tuple[torch.Tensor, torch.Tensor | None]:
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

    logger.info(f"[compile_vlm_dlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    base_model = AutoModelForCausalLM.from_pretrained(tlm_repo, torch_dtype=torch.float32, token=hf_token)
    lm_head_w, lm_head_b = extract_lm_head(base_model)
    embed_w = extract_embed(base_model)
    del base_model

    dlm_model = build_dlm_model(dlm_repo, lm_head_w, lm_head_b, embed_w)
    dlm_qeff = QEFFAutoModelForCausalLM(dlm_model, qaic_config={"dflash_dlm": True})
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
    hf_token: str | None = None,
) -> str:
    """Build + compile the DFlash DLM (draft) for the qwen3-vl TLM.

    Same VLM-aware lm_head/embed injection as ``compile_gemma_vlm_dlm_qpc``, kept as a
    separate function since qwen3-vl's config is not registered under
    ``AutoModelForCausalLM`` (it is a conditional-generation VLM config); loading the
    base TLM here goes through ``AutoModelForImageTextToText`` instead.
    """
    from QEfficient import QEFFAutoModelForCausalLM

    _, _, block_size = read_dlm_meta(dlm_repo, hf_token)

    logger.info(f"[compile_qwen3vl_dlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    base_model = AutoModelForImageTextToText.from_pretrained(tlm_repo, torch_dtype=torch.float32, token=hf_token)
    lm_head_w, lm_head_b = extract_lm_head(base_model)
    embed_w = extract_embed(base_model)
    del base_model

    dlm_model = build_dlm_model(dlm_repo, lm_head_w, lm_head_b, embed_w)
    dlm_qeff = QEFFAutoModelForCausalLM(dlm_model, qaic_config={"dflash_dlm": True})
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
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
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
            lambda x: {"turns": [format_lcb(x)]}, remove_columns=dataset.column_names, features=target_features
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
