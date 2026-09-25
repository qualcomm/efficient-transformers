# PyTorch Dynamo ONNX Semantic Node Names

## Problem

PyTorch Dynamo ONNX export preserves the originating module path in each
node's `pkg.torch.onnx.name_scopes` metadata, but emits generic
`NodeProto.name` values such as `node_linear` and
`node_invoke_subgraph_17__2`. Compiler partitioning and trace tools commonly
consume `NodeProto.name`, not exporter metadata, so they cannot associate
operations with their PyTorch module paths.

The issue is in the generic Dynamo ONNX exporter, not in Qwen, weight-free
export, or QAIC compilation.

## Upstream Design

The upstream change is being developed in the PyTorch exporter at
`torch/onnx/_internal/exporter/_core.py`.

- It is default behavior for `dynamo=True`; no public exporter option is added.
- It derives a slash-separated semantic path from the existing
  `nn_module_stack` / `name_scopes` values.
- It appends the existing generated name as `__<generated-name>` so that a
  single FX node lowered to multiple ONNX nodes remains uniquely named.
- Nodes without a real module stack retain their existing names.
- Tensor names, edges, operators, weights, function identities, and metadata
  are not modified.
- The implementation is generic and applies to the main graph and local ONNX
  functions. It does not include QEff-specific function deduplication,
  `invoke_subgraph` type substitution, or transformer-layer index removal.

Example output:

```text
/blocks/0/proj/linear__node_linear
/blocks/1/proj/linear_1__node_linear_1
```

## PyTorch Worktree

- Worktree: `/tmp/pytorch-upstream-naming`
- Branch: `amitraj/dynamo-semantic-onnx-node-names`
- Base: PyTorch `main` at `adb3a7b465e393ed08ea621408849abcaf2eabf4`
- Local patch commit: `cd76cc98f7f7da7bcce19ca9df737193de50ee57`
- Reference: PyTorch PR #197540, which is the implementation and testing
  precedent for a narrow generic Dynamo export fix.
- Intended signed-off commit: `bug(onnx.dynamo): preserve semantic node names`

The signed-off commit must use only the configured human author identity. Do
not include an AI attribution in the sign-off line.

## Current Validation

The local QEff environment contains PyTorch `2.13.0+cpu`, so it cannot execute
the current PyTorch-main suite directly. The candidate `_core.py` was loaded
alongside that binary and used to hot-patch the exporter for a two-block linear
model. The export produced the expected semantic names, all names were unique,
and the original metadata remained present.

Completed checks:

```bash
/local/mnt/workspace/amitraj/Pmodels/env/qeff_qwen_3/bin/python -m py_compile \
  torch/onnx/_internal/exporter/_core.py \
  test/onnx/exporter/test_api.py \
  test/onnx/exporter/test_core.py
```

The reduced Dynamo export also passed with the candidate `_core.py` loaded
alongside the installed binary. It produced
`/blocks/0/proj/linear__node_linear` and
`/blocks/1/proj/linear_1__node_linear_1`; all node names were unique. Saving
the ONNX program and reloading it with `onnx.load` preserved the same names in
the serialized `ModelProto.graph.node` entries.

The full upstream validation after installing or building the PyTorch worktree
must run the new focused `test_api.py` and `test_core.py` tests, then
`spin quicklint`.

On 2026-09-25, a full sparse-checkout expansion was started to prepare that
build but was stopped before dependency initialization because it was fetching
the complete filtered PyTorch source tree too slowly for this validation
session. The worktree was restored to its original sparse paths and remains
clean at `cd76cc9`. This does not replace the required current-main test run.

## Local Function Follow-up

Current PyTorch `main` still does not preserve single-use nested regions as
local ONNX functions in the public exporter test. PR #197540 addresses the
non-strict export behavior needed for that path. The naming formatter is
already applied at the shared IR-node creation point, so it will cover local
functions once they are preserved. After #197540 merges, add or enable an
end-to-end nested `invoke_subgraph` assertion that checks names on both the
main-graph callsite and local-function body.

## QEff Follow-up

Do not remove `DynamoSemanticNodeNameTransform` when this upstream patch lands.
It remains a compatibility fallback for older PyTorch releases and performs
QEff-specific normalization after subfunction deduplication. Once QEff upgrades
to a PyTorch release containing the upstream fix, rerun the original four-layer
Qwen weight-free KV-head-parallel QAIC validation and compare ONNX naming,
CPU/QAIC tokens, and compiler trace usability before reducing or removing it.
