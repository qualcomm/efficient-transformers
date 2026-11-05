---
name: qeff-glm-moe-dsa-export
description: Enable or debug GLM-MoE-DSA models, especially zai-org/GLM-5.3, through QEfficient Dynamo ONNX export and FP8 weight-free checkpoint preparation. Use for nested DSA cache export failures, GLM initializer-name mismatches, FP8 expert conversion, reduced/full-layer exports, or HF/QEff/QAIC decode parity.
---

# GLM-MoE-DSA Dynamo And Weight-Free Export

## Goal

Preserve GLM model semantics while making its DSA caches, MoE weights, and FP8
checkpoint representable across QEff PyTorch, Dynamo ONNX, weight-free external
weights, and QAIC decode execution.

Treat these as separate correctness boundaries:

1. HF PyTorch versus QEff PyTorch.
2. Dynamo graph capture and ONNX translation.
3. ONNX initializer names versus prepared checkpoint keys.
4. Compile and QAIC execution.
5. Decode-only token parity against an HF decode-only baseline.

Do not infer numerical parity from successful export or compilation.

## Architecture Inventory

Before editing, confirm the checkpoint/config rather than assuming a generic MoE
layout. For `zai-org/GLM-5.3`:

- `model_type` is `glm_moe_dsa`.
- Attention uses compressed KV plus RoPE-side cache tensors.
- DSA indexer cache exists only for full-indexer layers; shared-indexer layers
  reuse prior top-k indices.
- Query projection starts as fused `q_b_proj.weight`, while QEff execution uses
  derived `q_up` and `q_rope` parameters.
- Dense MLP layers use ordinary `gate_proj`, `up_proj`, and `down_proj` keys.
- Sparse MLP layers store separate per-expert `experts.<idx>.*_proj.weight`
  tensors with matching `weight_scale_inv` tensors.
- The real checkpoint uses blockwise FP8, including partial final blocks.

Inspect the current config fields and checkpoint index before changing code:

```bash
rg -n 'model_type|num_hidden_layers|indexer_types|mlp_layer_types|weight_block_size' <checkpoint>/config.json
```

## Dynamo Export Requirements

### Preserve Nested Cache Structure

The flattened ONNX names must be reconstructed into the exact Python pytree used
by `forward`.

- Convert `compressed_kv.<layer>` and `k_pe.<layer>` dynamic axes into nested
  `compressed_kvs` entries.
- Match the concrete container type used by example inputs. Dynamo treats lists
  and tuples as different pytree specifications.
- Convert `indexer_key.<layer>` entries into the ordered
  `indexer_key_cache` list.
- Generate indexer cache entries only for full-indexer layers, using
  `get_indexer_cache_layers(config)`.

Implement this in `QEfficient/utils/export_utils.py` and keep export input/output
name construction aligned in
`QEfficient/transformers/models/modeling_auto.py`.

### Make Fake Custom-Op Shapes Faithful

`ctx_gather_3d` gathers the context dimension but preserves all trailing feature
dimensions. Its fake implementation must return:

```python
(data.shape[0], ctx_indices.shape[1], *data.shape[2:])
```

Returning only `[batch, sequence]` loses the feature dimension and causes later
attention shape failures during fake-tensor propagation.

### Keep KV Shapes Symbolically Provable

In `_expand_glm_moe_dsa_kv`:

- Remove the singleton cache-head dimension explicitly before `kv_b_proj`.
- Reshape with `module.num_heads`; do not rely on `-1` for the head dimension.
- Restore the singleton dimension on `k_rot` explicitly.
- Expand to explicit `(batch_size, num_heads, seq_length, qk_rope_head_dim)`.

These operations let Dynamo prove the cache-head and feature dimensions instead
of introducing ambiguous symbolic products.

### Avoid Batch-Specialized Query Projection

Use `torch.matmul` for `q_resid @ q_up` and `q_resid @ q_rope`. The derived
weights have a leading singleton dimension, and `bmm` incorrectly constrains or
specializes that dimension against the symbolic batch.

### Separate Eager-Only Context Trimming

For CPU/HF-style incremental parity, trim retained caches and masks to the live
context. Do not execute data-dependent trimming during ONNX export or tracing:

```python
if position_ids is None or torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
    return attention_mask, states
```

Dynamo must see fixed cache capacity and symbolic positions, not Python values
obtained through `.item()`.

### Stabilize Export Outputs

During export, return logits and cache state as a deterministic tuple. Set a
temporary model export flag around `_export`, then remove it in `finally`, so
normal eager behavior remains unchanged.

The exported state contract includes:

- `compressed_kv.<layer>_RetainedState`
- `k_pe.<layer>_RetainedState`
- `indexer_key.<cache-index>_RetainedState`

Keep compile custom-I/O names synchronized with these outputs.

### Preserve Numerical Behavior

Exportability changes must not hide parity regressions. GLM-5.3 required:

- FP32 accumulation in RMSNorm, followed by conversion back to input dtype.
- RoPE caches built in FP32 and converted to the current activation dtype on use.
- Router logits and routing decisions computed in FP32.
- MoE output converted back to the residual activation dtype.

Validate these independently from graph-export success.

## Weight-Free FP8 Requirements

### Construct A Meta Model Directly

For weight-free export, call:

```python
QEFFAutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    weight_free=True,
    qaic_config={"mla_absorption": {"cache_compressed": True}},
)
```

Do not load the full HF model first. Set both `config.dtype` and
`config.torch_dtype` explicitly before construction.

The GLM FP8 path in `_run_quantizer_for_wf` must bypass the ordinary QEff FP8
module quantizer. The checkpoint transform emits dequantized floating-point
weights matching the meta model, so applying another quantizer produces the
wrong modules and initializer names.

### Select The GLM Transform Before Generic MoE Transforms

Register the GLM transform ahead of generic per-expert stacking. Applicability
must verify:

- `model_type == "glm_moe_dsa"`
- FP8 quantization
- fused `q_b_proj.weight` plus its scale
- real dense MLP keys
- real per-expert sparse MLP keys

Always forward `model_config` into both `is_applicable` and `apply`.

### Rewrite The Real Checkpoint Layout

For every configured active layer:

- Exclude layers outside `num_hidden_layers`.
- Exclude `model.mtp` and consumed `*_scale_inv` entries.
- Load each FP8 tensor with its scale.
- Dequantize with the configured two-dimensional `weight_block_size`.
- Pad only when the final block is partial, then slice back to the original
  weight shape.
- Convert the result to the export target dtype.

Preserve ordinary dense and attention keys. For every `q_b_proj.weight`, use the
same `split_glm_moe_dsa_q_b_proj` helper as runtime initialization to materialize:

- `self_attn.q_up`
- `self_attn.q_rope`

For every sparse layer, replace individual expert tensors with:

- `model.layers.<layer>.mlp.moe_weights.gate` as `[E,H,I]`
- `model.layers.<layer>.mlp.moe_weights.up` as `[E,H,I]`
- `model.layers.<layer>.mlp.moe_weights.down` as `[E,I,H]`

The checkpoint transform and `QEffGlmMoeDsaMoE.transform_weights` must produce
identical names and layouts. Weight-free resolution is name-based; a numerically
correct tensor under a different name is still an export failure.

### Bound Preparation Memory

The current GLM transform is intentionally sequential at application level:

- one base shard at a time
- one sparse layer at a time
- one gate/up/down projection at a time

Each full-model BF16 expert projection is 6 GiB, so one layer writes 18 GiB.
Parallelizing layers multiplies live stacked tensors and page-cache pressure.
Before adding workers, measure storage throughput and peak RSS; this workload can
be limited by `balance_dirty_pages`, in which case more workers increase memory
without reducing wall time.

### Keep Prepared-Checkpoint Caches Distinct

Prepared directory names and manifests must include at least:

- model type
- configured layer count
- target dtype
- source checkpoint fingerprint
- transform list

This prevents a reduced model, full model, or different dtype from reusing an
incompatible prepared directory. Use `QEFF_WF_HOME` to select the prepared
checkpoint root.

### Promote Initializers And Save The Weight Spec

After Dynamo export and checkpoint preparation:

1. Resolve ONNX initializer names against the prepared checkpoint.
2. Promote weight initializers to graph inputs.
3. Prune unused fake initializers.
4. Save the ONNX graph.
5. Save `weight_spec.json` beside it.
6. Embed the weight spec as ONNX metadata when the normal QEff transform flow
   completes.

Export-only acceptance requires both the ONNX and `weight_spec.json` to exist.

## HF Decode Parity Caveat

The partial-edge FP8 dequantization patch in
`scripts/glm53_four_layer_decode_compile_generate.py` is only for loading the
reduced HF baseline with `AutoModelForCausalLM`. It must use the checkpoint's
configured block size, normally `(128, 128)`, rather than inferring block size by
dividing tensor dimensions by scale-grid dimensions.

That bug affected the original HF comparison model and caused first-token
mismatch. It was not a defect in the weight-free checkpoint transform. Do not
install this HF compatibility patch in export-only weight-free mode.

For parity, compare decode-only against decode-only: feed the prompt one token at
a time into HF, retain its cache, then greedily decode. Comparing QAIC decode-only
execution against HF prefill generation tests a different numerical path.

## Validation

Run narrow checks first:

```bash
ruff format --check \
  QEfficient/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py \
  QEfficient/customop/dynamo_ops.py \
  QEfficient/utils/export_utils.py \
  QEfficient/exporter/weight_free/export.py \
  QEfficient/exporter/weight_free/checkpoint_transforms.py \
  tests/weight_free/test_transforms.py

pytest -q tests/weight_free/test_transforms.py -k glm
pytest -q tests/unit_test/models/test_model_quickcheck.py -k glm_moe_dsa
```

Tests should cover:

- `ctx_gather_3d` fake output retains trailing feature dimensions.
- Dynamic shapes preserve nested compressed KV and indexer cache structures.
- QEff PyTorch decode matches HF and shared-indexer cache mapping is correct.
- GLM transform selection precedes generic MoE stacking.
- FP8 partial-block dequantization uses configured block dimensions.
- Dense weights, `q_up`, and `q_rope` are present.
- Every active sparse layer has canonical gate/up/down tensors with correct
  shapes.
- Scale tensors, inactive layers, and MTP tensors are absent.
- Prepared-checkpoint cache identity changes with layer count and dtype.
- Weight-free and ordinary Dynamo exports have distinct export hashes.

For a real export, set:

```bash
export HF_HUB_CACHE=/path/to/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_WF_HOME=/path/to/fresh/prepared-checkpoints
```

Use export-only first. Compile and QAIC parity are later gates and must not be
triggered implicitly by an export validation command.

## Debugging Order

When export fails, classify the boundary before editing:

1. Fake-tensor shape error: inspect custom-op fake implementations.
2. Dynamic-shape/tree mismatch: compare `forward`, example-input containers,
   flattened ONNX names, and reconstructed `dynamic_shapes` pytrees.
3. Symbolic reshape/expand failure: replace inferred dimensions with explicit
   config constants and preserve singleton cache-head axes deliberately.
4. Missing weight-spec key: compare ONNX initializer names against transformed
   checkpoint keys and layouts.
5. First-token parity failure: compare HF and QEff loaded tensors before changing
   model math; verify the reduced HF FP8 loader uses the configured block size.
6. Later-token divergence: inspect retained-state names, cache update indices,
   live-context masking, RoPE dtype, and router precision.

Record exact tensor names, shapes, dtypes, first mismatch position, artifact
paths, and commands used. Do not describe export or compile success as parity.
