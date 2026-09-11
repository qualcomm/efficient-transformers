# Qwen3.8-2.4T-A95B Decode Enablement Notes

## Target
- Model: `Qwen/Qwen3.8-2.4T-A95B`
- HF architecture: `Qwen3_5MoeForCausalLM`
- Model type: `qwen3_5_moe_text`
- Scope: decode-only path, using `prefill_seq_len=1`; no prefill graph enablement in this phase.
- First validation path: dynamo export on a tiny/config-derived Qwen3.5-MoE text model,
  then weight-free export for the target.

## Model Facts
- 92 layers with repeating `linear_attention, linear_attention, linear_attention, full_attention`.
- `full_attention_interval=4`, `max_position_embeddings=262144`, `dtype=bfloat16`.
- Full attention uses `num_attention_heads=64`, `num_key_value_heads=4`, `head_dim=256`.
- Linear attention uses `linear_conv_kernel_dim=4`, `linear_num_key_heads=16`,
  `linear_num_value_heads=128`, `linear_key_head_dim=128`, `linear_value_head_dim=128`.
- MoE uses `num_experts=512`, `num_experts_per_tok=10`, `moe_intermediate_size=2048`,
  and `shared_expert_intermediate_size=2048`.
- Full-attention retained state uses `past_key.N` and `past_value.N`.
- Linear-attention retained state uses `conv_state.N` and `recurrent_state.N`.
- MoE checkpoint keys use fused expert tensors such as `model.layers.N.mlp.experts.gate_up_proj` and `down_proj`.

## Implementation Log
- Reuse existing Qwen3.5-MoE wrappers and transforms; do not add a new model family.
- Use CausalLM decode compile/export semantics (`prefill_seq_len=1`) instead of
  `export(decode_only=True)`, which is intentionally unsupported for CausalLM.
- Keep full target runs weight-free. Non-weight-free dynamo export should be limited to
  synthetic/tiny configs because loading the full checkpoint would materialize terabytes of weights.
- Added a decode-oriented dynamo example at `examples/dynamo/causal_lm/qwen3_8_2_4t_a95b_decode_dynamo.py`.
- Added Qwen3.5/Qwen3.5-MoE retained-state naming hooks so full-attention layers export
  `past_key.N`/`past_value.N` and linear-attention layers export `conv_state.N`/`recurrent_state.N`.
- Updated CausalLM export/compile glue to use model-provided retained-state specs/names,
  including hybrid custom IO for compile.
- Fixed the first synthetic dynamo blocker in the Qwen3.5-MoE text path by avoiding
  data-dependent `int(position_ids.max().item())` target-length calculation outside layerwise export.
- Fixed the second synthetic dynamo blocker by routing Qwen3.5/Qwen3.5-MoE gated DeltaNet RMSNorm
  through the same `select_interface(..., torch.ops.qefficient.rms_norm)` custom-op path used by
  other RMSNorm replacements during dynamo export.
- Made the decode dynamo example default to plain dynamo export and require
  `--use-onnx-subfunctions` for the subfunction path. Synthetic export currently reaches a
  subfunction-specific generated-graph argument mismatch after the earlier dynamo blockers are fixed.
- Plain synthetic tiny dynamo export passes:
  `.qeff/qwen3_8_decode/synthetic_tiny_dynamo_plain-101ca3565a45f320/Qwen3_5MoeForCausalLM.onnx`.
- Exported ONNX retained state inputs/outputs match the expected target pattern:
  linear layers `0-2` use `conv_state`/`recurrent_state`, full-attention layer `3` uses
  `past_key`/`past_value`.
- First target weight-free export reached `torch.export` and failed on RoPE cache indexing because
  `cos_cached`/`sin_cached` are `meta` tensors while position indices were CPU tensors. Fixed the
  shared Qwen3.5/Qwen3.5-MoE M-RoPE helper so flattened indices follow the RoPE cache device.
- Second target weight-free export failed in the fallback linear-attention chunk rule because
  precomputed CPU masks were multiplied with `meta` tensors. Fixed Qwen3.5/Qwen3.5-MoE chunk rules
  so causal/strict masks follow the active tensor device.
- Third target weight-free export progressed to full-attention RoPE application and failed because
  RoPE tensors could still be CPU at the multiply boundary. Fixed Qwen3.5/Qwen3.5-MoE RoPE apply
  helpers so `cos`/`sin` follow the query tensor device before multiplication.
- The decode example now calls `qeff_model.model.eval()` after loading the target model to avoid
  exporting the weight-free model in training mode.
- Tiny synthetic weight-free export then failed in the checkpoint resolver because HF-saved
  Qwen3.5-MoE text checkpoints use `model.language_model.*` keys while the QEff text graph exports
  `model.*` parameters. Added a narrow resolver fallback for this prefix layout.
- Tiny synthetic weight-free export next exposed QEff-generated linear-attention helper buffers
  (`_mask_causal`, `_mask_strict`, `_ones_lower`, `_eye`) as ONNX initializers. These buffers are
  not checkpoint weights, so they are now classified as computed initializers and remain embedded.
- Tiny synthetic weight-free export also exposed alternate MoE parameter aliases such as
  `model.layers.N.mlp.moe_weights.gate` while the live QEff module owns
  `model.layers.N.mlp.experts.moe_weights.gate`. The promotion candidate set now includes known
  MoE aliases so these meta initializers are promoted to external weight inputs instead of being
  serialized.
- Added quickcheck regression coverage for hybrid ONNX state names, dynamo dynamic-shape nesting,
  and compile custom IO with `use_onnx_subfunctions=True`.
- Added weight-free resolver regression coverage for Qwen3.5-MoE text checkpoint prefixes,
  generated linear-attention helper buffers, and MoE alias initializer promotion.
- The example script sets `HF_HUB_ENABLE_HF_TRANSFER=1` before any Hugging Face Hub access.
- Fetched only the public `config.json` metadata with `curl`; no checkpoint or tokenizer files were
  downloaded during this pass.
- Tiny synthetic weight-free dynamo export now passes:
  `.qeff/qwen3_8_decode/synthetic_tiny_weight_free_dynamo_plain-4250396c7419d698/Qwen3_5MoeForCausalLM.onnx`.
- The tiny weight-free artifact has 76 external weight inputs in `weight_spec.json`, keeps 15
  computed initializers embedded, and exports retained states as `conv_state.0/1/2`,
  `recurrent_state.0/1/2`, `past_key.3`, and `past_value.3`.
- First tiny synthetic weight-free compile reached QAIC compiler and failed on `node_slice_7`
  because the initial tiny config used `head_dim=8` with Qwen3.5's default M-RoPE section logic,
  producing an invalid empty slice after compiler clamping. This was a synthetic-config issue,
  not a target model issue: the target uses `head_dim=256`.
- Updated the synthetic tiny config to `hidden_size=128`, `head_dim=32`, preserving
  `partial_rotary_factor=0.25`, so the tiny rotary dimension is 8 and M-RoPE slices remain valid.
  The helper now regenerates the synthetic checkpoint when its saved config differs.
- Tiny synthetic weight-free compile now passes:
  `.qeff/qwen3_8_decode/synthetic_tiny_weight_free_compile/qpc-e7aab9d790475fa2/qpc`.
- The compiled custom IO includes all hybrid decode states:
  `conv_state.0/1/2`, `recurrent_state.0/1/2`, `past_key.3`, and `past_value.3`,
  plus matching `_RetainedState` outputs.

## Open Checks
- KV blocking first pass in progress. Current scope is `blocking_mode="kv"` only; `kv_headpar`
  will be enabled after plain KV blocking is validated.
- Fixed blocked attention forwarding in Qwen3.5 and Qwen3.5-MoE from `comp_ctx_length`
  to `comp_ctx_lengths`, so compute-context-length handling reaches the generic blocked
  attention interface.
- Added `--blocking-mode`, `--num-kv-blocks`, and `--headpar-split` arguments to the
  Qwen3.8 decode dynamo example. Only `kv` is being tested in the current pass.
- Updated the synthetic tiny config with `rope_parameters["mrope_section"] = [2, 1, 1]`
  to keep M-RoPE slices valid for `head_dim=32`.
- KV blocking dynamo export succeeded but compile initially failed in QAIC's fp16 conversion
  path (`channelwiseQuantizeFloatBias`) when the synthetic tiny model used float32. The same
  blocked dynamo ONNX compiles when `-convert-to-fp16` is not used, and a target-like bfloat16
  one-layer tiny model compiles with `blocking_mode="kv"`.
- Updated the synthetic tiny example to use bfloat16 config and bfloat16 local synthetic weights,
  matching the real target config dtype and avoiding the float32-to-fp16 compiler path.
- The 4-layer hybrid bfloat16 KV compile then exposed a retained-state dtype mismatch:
  linear-attention `recurrent_state.N` inputs were bfloat16 but `_RetainedState` outputs could
  remain float32 on the non-continuous-batching path. Cast those outputs back to the original
  retained-state dtype in Qwen3.5 and Qwen3.5-MoE.
- Tiny weight-free KV export failed before ONNX save on used meta `lifted_tensor_*` initializers
  created by scalar `torch.tensor(...)` calls inside the plain KV blocked attention path. Replaced
  those KV-path scalar tensors with Python scalar/block-boundary equivalents so they do not become
  checkpoint-like meta initializers.
- Synthetic weight-free loading reset the tiny model config dtype to float32 unless dtype was passed
  explicitly. The example now passes/restores bfloat16 for synthetic weight-free runs so compile
  follows the target-like bfloat16 path instead of `-convert-to-fp16`.
- KV blocking validation passed for the 4-layer hybrid synthetic tiny weight-free decode graph:
  `.qeff/qwen3_8_decode_kv_pass3/synthetic_tiny_weight_free_compile_kv_bf16/qpc-0eaa33ea9ce84962/qpc`.
- The passing weight-free KV export produced
  `.qeff/qwen3_8_decode_kv_pass3/Qwen3_5MoeForCausalLM/Qwen3_5MoeForCausalLM-7adb8e13876cb07f/Qwen3_5MoeForCausalLM.onnx`
  with a `weight_spec.json` containing 76 external weight inputs.
- Validation run:
  `python examples/dynamo/causal_lm/qwen3_8_2_4t_a95b_decode_dynamo.py --synthetic-tiny --weight-free --ctx-len 128 --num-cores 4 --num-devices 1 --blocking-mode kv --num-kv-blocks 2 --compile-dir .qeff/qwen3_8_decode_kv_pass3/synthetic_tiny_weight_free_compile_kv_bf16`.
- Additional checks passed: `git diff --check`, `python -m ruff check ...`, `python -m pytest -q tests/weight_free/test_transforms.py`,
  and `python -m pytest -q tests/unit_test/models/test_model_quickcheck.py -k "qwen3_5_moe"`.
- Confirm target weight-free export resolves fused MoE checkpoint keys to QEff `moe_weights` keys
  on the real `Qwen/Qwen3.8-2.4T-A95B` config/checkpoint metadata.
- Revisit `use_onnx_subfunctions=True` for Qwen3.5-MoE CausalLM after plain dynamo export is green.
