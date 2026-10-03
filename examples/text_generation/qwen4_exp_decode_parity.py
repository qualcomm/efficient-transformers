# -----------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
"""Token-by-token HF/QEff PyTorch parity runner for Qwen3.8-Flash-Next.

This runner intentionally does not call ``generate``.  Its PLE n-gram table is
kept in safetensors on the host and only the selected `[B, 1, ple_embed_dim]`
activation crosses into either model.  It loads a configurable leading layer
window, never the 48-layer model or the full PLE embedding table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen4ExpForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

from QEfficient.transformers.models.qwen4_exp.host_ngram import HostNGramHistory, ShardedNGramLookup
from QEfficient.transformers.models.qwen4_exp.modeling_qwen4_exp import QEffQwen4ExpForCausalLM


class _HostProvidedNGramEmbedding(torch.nn.Module):
    """Reference-model PLE adapter; the caller sets the current host activation."""

    def __init__(self):
        super().__init__()
        self.embeddings = None

    def forward(self, input_ids, past_key_values):
        del input_ids, past_key_values
        if self.embeddings is None:
            raise RuntimeError("Set host PLE embeddings before invoking the reference model")
        return self.embeddings


def _window_config(snapshot: Path, layers: int, context: int) -> Qwen4ExpTextConfig:
    raw = json.loads((snapshot / "config.json").read_text())["text_config"]
    raw["num_hidden_layers"] = layers
    raw["layer_types"] = raw["layer_types"][:layers]
    raw["ple_layer_ids"] = [layer_id for layer_id in raw["ple_layer_ids"] if layer_id <= layers]
    raw["max_position_embeddings"] = context
    return Qwen4ExpTextConfig(**raw)


def _checkpoint_state(snapshot: Path, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Read only tensors owned by the leading text-model window, excluding PLE tables."""
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    expected = set(model.state_dict())
    selected = {}
    for source_key, filename in weight_map.items():
        target_key = source_key.removeprefix("model.language_model.")
        if target_key.startswith("model."):
            target_key = target_key.removeprefix("model.")
        elif source_key.startswith("model.language_model."):
            target_key = f"model.{target_key}"
        if ".ple_embedding.ngram_embedding." in target_key or target_key not in expected:
            continue
        with safe_open(str(snapshot / filename), framework="pt", device="cpu") as handle:
            selected[target_key] = handle.get_tensor(source_key)
    missing = [name for name, tensor in model.state_dict().items() if name not in selected and tensor.is_meta]
    if missing:
        raise RuntimeError(f"Checkpoint window is missing {len(missing)} required tensors, e.g. {missing[:3]}")
    return selected


def _make_models(config: Qwen4ExpTextConfig, snapshot: Path):
    """Build reference/QEff models on meta first, then share the selected tensors."""
    with torch.device("meta"):
        reference = Qwen4ExpForCausalLM(config)
        qeff = Qwen4ExpForCausalLM(config)
    host_ple = _HostProvidedNGramEmbedding()
    for layer_id in config.ple_layer_ids:
        reference.model.layers[layer_id - 1].ple.ple_embedding = host_ple
    # The QEff conversion removes this meta table after loading; it is never read.
    state = _checkpoint_state(snapshot, reference)
    reference.load_state_dict(state, strict=False, assign=True)
    qeff.load_state_dict(state, strict=False, assign=True)
    # The meta-model construction omits nonpersistent HF rotary buffers from the
    # checkpoint state. Recreate them before using the reference model.
    reference.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)
    qeff.__class__ = QEffQwen4ExpForCausalLM
    qeff.__qeff_init__()
    return reference.eval(), qeff.eval(), host_ple


def _qeff_cache(model, dtype: torch.dtype):
    cache = model.get_dummy_inputs(batch_size=1)["past_key_values"]
    return tuple(
        tuple(
            state.to(
                dtype=torch.float32
                if name.startswith(("gdn_recurrent_state", "qsa_partial_state"))
                else dtype
            )
            for name, state in zip(model.get_onnx_past_key_value_names(layer_idx), layer)
        )
        for layer_idx, layer in enumerate(cache)
    )


def run(snapshot: Path, token_ids: list[int], steps: int, layers: int, context: int, atol: float, rtol: float) -> None:
    config = _window_config(snapshot, layers, context)
    reference, qeff, host_ple = _make_models(config, snapshot)
    lookup = ShardedNGramLookup(config, snapshot)
    history = HostNGramHistory(config.ngram_size, config.eos_token_id)
    qeff_cache = _qeff_cache(qeff, qeff.lm_head.weight.dtype)
    hf_cache = DynamicCache(config=config)
    token = torch.tensor([[token_ids[0]]], dtype=torch.long)
    position = 0
    for step in range(steps):
        ngram_embeddings = lookup.lookup(token, history)
        host_ple.embeddings = ngram_embeddings
        position_ids = torch.full((4, 1, 1), position, dtype=torch.long)
        with torch.no_grad():
            hf_output = reference(
                input_ids=token, position_ids=position_ids, past_key_values=hf_cache, use_cache=True
            )
            qeff_output = qeff(
                input_ids=token,
                ngram_embeddings=ngram_embeddings,
                position_ids=position_ids,
                past_key_values=qeff_cache,
            )
        torch.testing.assert_close(qeff_output.logits, hf_output.logits, atol=atol, rtol=rtol)
        max_error = (qeff_output.logits.float() - hf_output.logits.float()).abs().max().item()
        print(f"step={step} max_abs_logit_error={max_error:.6g}")
        qeff_cache = qeff_output.past_key_values
        hf_cache = hf_output.past_key_values
        token = hf_output.logits[:, -1].argmax(dim=-1, keepdim=True)
        position += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Completed local Qwen3.8-Flash-Next snapshot")
    parser.add_argument("--token-ids", type=int, nargs="+", default=[1], help="Initial token; later tokens are greedy HF")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4, help="Leading layer window; 4 exercises GDN, PLE, and QSA")
    parser.add_argument("--context", type=int, default=32, help="Allocated QSA cache context")
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    args = parser.parse_args()
    run(args.snapshot, args.token_ids, args.steps, args.layers, args.context, args.atol, args.rtol)


if __name__ == "__main__":
    main()
