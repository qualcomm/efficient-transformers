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

import numpy as np
import torch
from safetensors import safe_open
from transformers import Qwen4ExpForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.qwen4_exp.host_ngram import HostNGramHistory, ShardedNGramLookup
from QEfficient.transformers.models.qwen4_exp.modeling_qwen4_exp import QEffQwen4ExpForCausalLM

QWEN4_RETAINED_STATE_NAMES = (
    "gdn_conv_state.0",
    "gdn_recurrent_state.0",
    "gdn_conv_state.1",
    "gdn_recurrent_state.1",
    "ple_conv_state.1",
    "gdn_conv_state.2",
    "gdn_recurrent_state.2",
    "qsa_key_state.3",
    "qsa_value_state.3",
    "qsa_index_state.3",
    "qsa_partial_state.3",
)
QWEN4_RETAINED_OUTPUT_NAMES = tuple(f"{name}_RetainedState" for name in QWEN4_RETAINED_STATE_NAMES)
QWEN4_QAIC_HOST_INPUT_NAMES = ("input_ids", "ngram_embeddings", "position_ids")
QWEN4_QAIC_INPUT_SHAPES = {
    "input_ids": (1, 1),
    "ngram_embeddings": (1, 1, 2560),
    "position_ids": (4, 1, 1),
}
QWEN4_QAIC_STATE_SHAPES = {
    "gdn_conv_state.0": (1, 10240, 4),
    "gdn_recurrent_state.0": (1, 48, 128, 128),
    "gdn_conv_state.1": (1, 10240, 4),
    "gdn_recurrent_state.1": (1, 48, 128, 128),
    "ple_conv_state.1": (1, 10240, 9),
    "gdn_conv_state.2": (1, 10240, 4),
    "gdn_recurrent_state.2": (1, 48, 128, 128),
    "qsa_key_state.3": (1, 2, 512, 256),
    "qsa_value_state.3": (1, 2, 512, 256),
    "qsa_index_state.3": (1, 128, 128),
    "qsa_partial_state.3": (1, 1, 128),
}
QWEN4_QAIC_LOGITS_SHAPE = (1, 1, 248320)


def _binding_basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _session_binding_shape(session, name: str) -> tuple[int, ...]:
    binding = session.bindings[session.binding_index_map[name]]
    return tuple(binding.dims)


class Qwen4ExpQAICDecodeRunner:
    """Single-token Qwen4-Exp decode runner with device-owned retained state."""

    def __init__(self, qpc_path: Path, device_ids: list[int] | None = None):
        self.session = QAICInferenceSession(qpc_path, device_ids=device_ids)
        known_retained_names = set(QWEN4_RETAINED_STATE_NAMES + QWEN4_RETAINED_OUTPUT_NAMES)
        skipped_names = [
            name
            for name in (*self.session.input_names, *self.session.output_names)
            if _binding_basename(name) in known_retained_names
        ]
        # Retained states may be internal to a tensor-sliced QPC. Skip only
        # bindings exposed by this session, leaving all exposed state on device.
        self.session.skip_buffers(skipped_names)
        session_inputs = {_binding_basename(name): name for name in self.session.input_names}
        self.host_input_names = {
            name: session_inputs[name]
            for name in QWEN4_QAIC_HOST_INPUT_NAMES
            if name in session_inputs
        }
        missing_host_inputs = set(QWEN4_QAIC_HOST_INPUT_NAMES) - set(self.host_input_names)
        if missing_host_inputs:
            raise ValueError(f"QPC is missing Qwen host inputs: {sorted(missing_host_inputs)}")
        self.host_input_shapes = {
            name: _session_binding_shape(self.session, binding_name)
            for name, binding_name in self.host_input_names.items()
        }
        for name, expected_shape in QWEN4_QAIC_INPUT_SHAPES.items():
            if self.host_input_shapes[name] != expected_shape:
                raise ValueError(f"Unexpected QPC shape for {name}: {self.host_input_shapes[name]}")

        logits_name = next((name for name in self.session.output_names if _binding_basename(name) == "logits"), None)
        if logits_name is None:
            raise ValueError(f"QPC is missing logits output: {self.session.output_names}")
        self.logits_name = logits_name
        self.logits_shape = _session_binding_shape(self.session, logits_name)
        self.session.set_buffers({self.logits_name: np.zeros(self.logits_shape, dtype=np.float32)})

    def run_step(self, input_ids: np.ndarray, ngram_embeddings: torch.Tensor, position_ids: np.ndarray) -> np.ndarray:
        inputs = {
            self.host_input_names["input_ids"]: np.asarray(input_ids, dtype=np.int64),
            self.host_input_names["ngram_embeddings"]: ngram_embeddings.detach().cpu().numpy().astype(np.float32, copy=False),
            self.host_input_names["position_ids"]: np.asarray(position_ids, dtype=np.int64),
        }
        for name, binding_name in self.host_input_names.items():
            if tuple(inputs[binding_name].shape) != self.host_input_shapes[name]:
                raise ValueError(f"{name} must be {self.host_input_shapes[name]}")
        outputs = self.session.run(inputs)
        logits_output = next((output for name, output in outputs.items() if _binding_basename(name) == "logits"), None)
        if logits_output is None:
            raise ValueError(f"QAIC run did not return logits: {sorted(outputs)}")
        logits = np.asarray(logits_output, dtype=np.float32)
        if tuple(logits.shape) != self.logits_shape:
            raise ValueError(f"Unexpected QAIC logits shape: {tuple(logits.shape)}")
        return logits


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
    # Export the PyTorch/ONNX graph in FP32. QAIC compilation performs the
    # hardware-side FP16 conversion; retained-state custom I/O is configured
    # separately as FP16 for that compiler boundary.
    raw["dtype"] = "float32"
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
    # The QEff QSA implementation is explicit eager attention.  The HF side
    # must use the same attention path for an accepted parity comparison.
    config._attn_implementation = "eager"
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
    reference = reference.float()
    qeff = qeff.float()
    # The meta-model construction omits nonpersistent HF rotary buffers from the
    # checkpoint state. Recreate them before using the reference model.
    reference.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)
    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

    qeff, transformed = KVCacheTransform.apply(qeff)
    if not transformed or not isinstance(qeff, QEffQwen4ExpForCausalLM):
        raise RuntimeError("KVCacheTransform did not produce the Qwen4-Exp QEfficient model")
    return reference.eval(), qeff.eval(), host_ple


def _qeff_cache(model, dtype: torch.dtype):
    cache = model.get_dummy_inputs(batch_size=1)["past_key_values"]
    return tuple(
        tuple(
            state.to(dtype=torch.float32 if name.startswith(("gdn_recurrent_state", "qsa_partial_state")) else dtype)
            for name, state in zip(model.get_onnx_past_key_value_names(layer_idx), layer)
        )
        for layer_idx, layer in enumerate(cache)
    )


def _run_prompt_and_generation_loop(prompt_token_ids, generation_steps, run_token, compare_logits):
    """Consume a known prompt, then greedily decode from its final logits.

    ``run_token`` consumes exactly one input token and returns HF, QEff, and
    QAIC logits.  Prompt logits before the final prompt token are deliberately
    discarded: their successor is supplied by the known prompt instead.
    """
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must contain at least one token")
    if generation_steps < 1:
        raise ValueError("generation_steps must be at least one")

    events = []
    for position, token_id in enumerate(prompt_token_ids):
        hf_logits, qeff_logits, qaic_logits = run_token(token_id, position)
        is_final_prompt_token = position == len(prompt_token_ids) - 1
        phase = "prompt-final" if is_final_prompt_token else "prompt"
        if not is_final_prompt_token:
            next_token_id = prompt_token_ids[position + 1]
            print(
                f"phase={phase} position={position} input_token={token_id} next_token={next_token_id} logits=discarded"
            )
            events.append({"phase": phase, "input_token": token_id, "next_token": next_token_id, "compared": False})
            continue

        next_token_id = compare_logits(phase, position, token_id, hf_logits, qeff_logits, qaic_logits)
        events.append({"phase": phase, "input_token": token_id, "next_token": next_token_id, "compared": True})
        break

    generated_token_ids = [next_token_id]
    for generation_index in range(1, generation_steps):
        position = len(prompt_token_ids) + generation_index - 1
        hf_logits, qeff_logits, qaic_logits = run_token(next_token_id, position)
        next_token_id = compare_logits("generation", position, next_token_id, hf_logits, qeff_logits, qaic_logits)
        generated_token_ids.append(next_token_id)
        events.append(
            {
                "phase": "generation",
                "input_token": events[-1]["next_token"],
                "next_token": next_token_id,
                "compared": True,
            }
        )
    return generated_token_ids, events


def _compare_hf_qeff_qaic_logits(phase, position, input_token, hf_logits, qeff_logits, qaic_logits, atol, rtol):
    hf_logits = np.asarray(hf_logits, dtype=np.float32)
    qeff_logits = np.asarray(qeff_logits, dtype=np.float32)
    qaic_logits = np.asarray(qaic_logits, dtype=np.float32)
    qeff_max_error = np.max(np.abs(qeff_logits - hf_logits)).item()
    qaic_error = np.abs(qaic_logits - hf_logits)
    qaic_max_error = qaic_error.max().item()
    qaic_mad = qaic_error.mean().item()
    hf_next_token = int(hf_logits.argmax(axis=-1).reshape(-1)[0])
    qeff_next_token = int(qeff_logits.argmax(axis=-1).reshape(-1)[0])
    qaic_next_token = int(qaic_logits.argmax(axis=-1).reshape(-1)[0])
    print(
        f"phase={phase} position={position} input_token={input_token} "
        f"hf_next_token={hf_next_token} qeff_next_token={qeff_next_token} qaic_next_token={qaic_next_token} "
        f"qeff_max_abs={qeff_max_error:.6g} qaic_max_abs={qaic_max_error:.6g} qaic_mad={qaic_mad:.6g}"
    )
    if qeff_next_token != hf_next_token:
        raise AssertionError(
            f"HF/QEff greedy-token mismatch at position {position}: {hf_next_token} != {qeff_next_token}"
        )
    if qaic_next_token != hf_next_token:
        raise AssertionError(
            f"HF/QAIC greedy-token mismatch at position {position}: {hf_next_token} != {qaic_next_token}"
        )
    np.testing.assert_allclose(qeff_logits, hf_logits, atol=atol, rtol=rtol)
    np.testing.assert_allclose(qaic_logits, hf_logits, atol=atol, rtol=rtol)
    return hf_next_token


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
        ngram_embeddings = lookup.lookup(token, history).to(reference.lm_head.weight.dtype)
        host_ple.embeddings = ngram_embeddings
        position_ids = torch.full((4, 1, 1), position, dtype=torch.long)
        with torch.no_grad():
            hf_output = reference(input_ids=token, position_ids=position_ids, past_key_values=hf_cache, use_cache=True)
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


def run_hf_qeff_qaic(
    snapshot: Path,
    qpc_path: Path,
    prompt_token_ids: list[int],
    generation_steps: int,
    layers: int,
    context: int,
    atol: float,
    rtol: float,
    device_ids: list[int] | None = None,
) -> None:
    """Run a fresh QAIC session against matching HF/QEff one-token decode calls."""
    if layers < 1 or context != 512:
        raise ValueError("Qwen4-Exp QAIC requires --layers >= 1 and --context 512")
    config = _window_config(snapshot, layers, context)
    reference, qeff, host_ple = _make_models(config, snapshot)
    lookup = ShardedNGramLookup(config, snapshot)
    history = HostNGramHistory(config.ngram_size, config.eos_token_id)
    qeff_cache = _qeff_cache(qeff, qeff.lm_head.weight.dtype)
    hf_cache = DynamicCache(config=config)
    qaic_runner = Qwen4ExpQAICDecodeRunner(qpc_path, device_ids=device_ids)

    def run_token(token_id: int, position: int):
        nonlocal hf_cache, qeff_cache
        token = torch.tensor([[token_id]], dtype=torch.long)
        ngram_embeddings = lookup.lookup(token, history).to(reference.lm_head.weight.dtype)
        host_ple.embeddings = ngram_embeddings
        position_ids = torch.full((4, 1, 1), position, dtype=torch.long)
        with torch.no_grad():
            hf_output = reference(input_ids=token, position_ids=position_ids, past_key_values=hf_cache, use_cache=True)
            qeff_output = qeff(
                input_ids=token,
                ngram_embeddings=ngram_embeddings,
                position_ids=position_ids,
                past_key_values=qeff_cache,
            )
        qaic_logits = qaic_runner.run_step(
            token.detach().cpu().numpy(), ngram_embeddings, position_ids.detach().cpu().numpy()
        )
        hf_cache = hf_output.past_key_values
        qeff_cache = qeff_output.past_key_values
        return (
            hf_output.logits.detach().float().cpu().numpy(),
            qeff_output.logits.detach().float().cpu().numpy(),
            qaic_logits,
        )

    generated_token_ids, _ = _run_prompt_and_generation_loop(
        prompt_token_ids,
        generation_steps,
        run_token,
        lambda phase, position, input_token, hf_logits, qeff_logits, qaic_logits: _compare_hf_qeff_qaic_logits(
            phase, position, input_token, hf_logits, qeff_logits, qaic_logits, atol, rtol
        ),
    )
    print(f"generated_token_ids={generated_token_ids}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Completed local Qwen3.8-Flash-Next snapshot")
    parser.add_argument(
        "--token-ids", type=int, nargs="+", default=[1], help="Initial token; later tokens are greedy HF"
    )
    parser.add_argument(
        "--prompt-token-ids",
        type=int,
        nargs="+",
        help="Known prompt tokens for QAIC mode; defaults to --token-ids",
    )
    parser.add_argument("--qpc-path", type=Path, help="Existing 4-layer Qwen4-Exp decode QPC directory")
    parser.add_argument("--device-id", type=int, nargs="+", help="Optional QAIC device IDs")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4, help="Leading layer window; 4 exercises GDN, PLE, and QSA")
    parser.add_argument("--context", type=int, default=32, help="Allocated QSA cache context")
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    args = parser.parse_args()
    if args.qpc_path is None:
        run(args.snapshot, args.token_ids, args.steps, args.layers, args.context, args.atol, args.rtol)
    else:
        run_hf_qeff_qaic(
            args.snapshot,
            args.qpc_path,
            args.prompt_token_ids or args.token_ids,
            args.steps,
            args.layers,
            args.context,
            args.atol,
            args.rtol,
            args.device_id,
        )


if __name__ == "__main__":
    main()
