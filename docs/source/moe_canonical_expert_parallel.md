# Canonical MoE weights for expert-parallel prefill

## Summary

MoE prefill and decode use different execution strategies:

- **Decode BMM** gathers the selected experts and performs batched matrix
  multiplication.
- **Expert-parallel prefill** distributes groups of experts across pipeline
  stages and processes the groups in a blocked loop.

Both strategies can operate on the same canonical expert tensors. The prepared
checkpoint therefore stores one layout, and the prefill graph cleaderives the
expert-parallel view with constant tensor operations. The compiler can fold
these operations into the constant handling performed during compilation.

This design avoids producing a second expert-parallel checkpoint and avoids
adding layout operations to the weight-spec format.

## The two layouts

The canonical layout used by the prepared checkpoint is:

```text
gate: [E, H, I]
up:   [E, H, I]
down: [E, I, H]
```

Here `E` is the total number of experts, `H` is the hidden size, and `I` is the
intermediate size. Optional biases use `[E, I]` for gate/up and `[E, H]` for
down.

Expert-parallel execution conceptually uses:

```text
gate: [E/P, P, H, I]
up:   [E/P, P, H, I]
down: [E/P, P, I, H]
```

`P` is the number of pipeline stages and `E/P` is the number of experts handled
by each stage.

The expert order is intentionally chosen so that each stage owns a contiguous
range in the canonical tensor. For example, with `E=8` and `P=2`:

```text
canonical expert order: 0 1 2 3 4 5 6 7
stage 0:                0 1 2 3
stage 1:                4 5 6 7
```

## Graph-side layout conversion

For canonical weights, `moe_expert_parallel` applies this transformation to
each expert tensor:

```text
[E, ...]
  -> reshape [P, E/P, ...]
  -> transpose the first two axes
  -> [E/P, P, ...]
```

For `E=8`, `P=2`, this is:

```text
[8, H, I] -> [2, 4, H, I] -> [4, 2, H, I]
```

The loop then selects one pipeline-stage column, producing `[E/P, ...]` for
the local expert computation. The operations are applied to ONNX constants
after export, so the compiler can fold them without requiring the checkpoint
loader to understand a new layout-transform language.

The implementation is in
[`QEfficient/transformers/moe/flavours.py`](../../QEfficient/transformers/moe/flavours.py),
in `_expert_parallel_layout()` and `moe_expert_parallel()`.

Legacy packed tensors with rank 4 are still accepted. They already have the
`[E/P, P, ...]` layout and bypass the graph reshape/transpose. The PyTorch
export transform restores such weights to canonical form when possible.

## Checkpoint preparation

The weight-free checkpoint pipeline converts model-specific source layouts into
canonical tensors:

1. Separate expert tensors are stacked by expert index.
2. Fused tensors such as `gate_up` are split into separate canonical `gate` and
   `up` tensors.
3. Model-specific transposes and dtype conversion are applied.
4. The final prepared files contain canonical `moe_weights.gate`,
   `moe_weights.up`, and `moe_weights.down` tensors.

Expert-parallel packing is no longer a checkpoint stage. Consequently, the
prepared checkpoint hash depends on the actual checkpoint plan, model reference,
target dtype, and transform group, but not on MoE flavour, `P`, or `E/P`.

Decode and expert-parallel prefill therefore resolve to the same directory, for
example:

```text
.../qwen3-moe-qeff-prepared-<hash>/
```

The relevant implementation is in
[`QEfficient/exporter/weight_free/checkpoint_transforms.py`](../../QEfficient/exporter/weight_free/checkpoint_transforms.py)
and
[`QEfficient/exporter/weight_free/export.py`](../../QEfficient/exporter/weight_free/export.py).

## Weight-spec contract

The weight spec remains version 5. An entry maps an ONNX graph input directly to
one tensor in a prepared safetensors file:

```json
{
  "name": "model.layers.0.mlp.moe_weights.gate",
  "location": {
    "file": 0,
    "key": "model.layers.0.mlp.moe_weights.gate"
  }
}
```

There is no `shape` or `transform` field. The graph input shape must match the
stored tensor shape. The exporter reads safetensors headers and rejects a
mismatch before compilation, which catches accidental packed/canonical layout
confusion early.

The shape guard is implemented in
[`QEfficient/exporter/weight_free/checkpoint_key_resolver.py`](../../QEfficient/exporter/weight_free/checkpoint_key_resolver.py).

## Runtime flavour behavior

The default selection remains:

```text
prefill -> expert_parallel, when supported
decode  -> decode_bmm, when supported
```

Decode BMM uses the canonical tensors directly and gathers expert indices from
the router output. Expert-parallel prefill converts the canonical constants in
the graph and processes each pipeline slot. Both paths use the same underlying
bytes from the prepared checkpoint.

The model-side implementation is in
[`QEfficient/transformers/models/pytorch_transforms.py`](../../QEfficient/transformers/models/pytorch_transforms.py),
[`QEfficient/transformers/moe/block.py`](../../QEfficient/transformers/moe/block.py),
and
[`QEfficient/transformers/moe/flavours.py`](../../QEfficient/transformers/moe/flavours.py).

## Why this is preferable to weight-spec transforms

An earlier design extended the weight spec with version 6 `reshape` and
`transpose` operations. That would have allowed the loader to convert a
canonical checkpoint tensor into the prefill layout at load time.

The compiler already supports reshape and transpose of constants. Keeping the
operations in the exported PyTorch/ONNX graph has several advantages:

- the checkpoint format remains unchanged;
- the compiler owns constant folding and layout materialization;
- prefill and decode share the normal version 5 loader path;
- there is no second prepared checkpoint or loader-side transform contract.

## Compatibility and safety

- Existing canonical checkpoints continue to work.
- Legacy packed MoE weights are recognized and accepted by the model-side
  execution path.
- The exporter validates every promoted weight input against the stored tensor
  header shape.
- A mismatch is rejected rather than silently binding a tensor with a
  different layout.
- The canonical layout requires `E == P * (E/P)` for expert-parallel prefill.

This design changes the prepared-checkpoint contents and hash for expert-
parallel exports compared with older packing behavior. Old prepared directories
remain separate cache entries; they are not overwritten.

## Validation

The focused validation performed for this change includes:

```bash
pytest -n auto tests/unit_test/transforms/test_transform_accuracy.py \
  -k "expert_parallel or ExpertParallel or facade"
pytest -n auto tests/weight_free/test_transforms.py
pytest -n auto tests/unit_test/models/test_model_quickcheck.py -k moe
```

The canonical-versus-packed parity tests passed for multiple `(P, E/P)` layouts
and for MoE models with and without biases. The real
`tiny-random/qwen3-moe` flow was also validated on QAIC:

- expert-parallel prefill compiled and executed;
- decode BMM compiled and executed;
- both used the same prepared checkpoint directory;
- the `full_batch_size=2` continuous-batching QPCs executed direct prefill and
  decode smoke tests.

The default decode path is decode BMM. Explicitly forcing decode expert-parallel
is a separate compiler path and currently encounters the QAIC importer error
`limit and start value should be different` for the tested tiny Qwen3 graph.
