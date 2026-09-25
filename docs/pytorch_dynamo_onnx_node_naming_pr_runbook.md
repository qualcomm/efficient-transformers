# PyTorch Dynamo ONNX Naming PR Runbook

This runbook prepares and raises the upstream PyTorch PR for semantic Dynamo
ONNX node names.

## Prepared Patch

```text
Worktree: /tmp/pytorch-upstream-naming
Branch: amitraj/dynamo-semantic-onnx-node-names
Commit: cd76cc98f7f7da7bcce19ca9df737193de50ee57
Title: bug(onnx.dynamo): preserve semantic node names
```

The commit is signed off by `Amit Raj <amitraj@qti.qualcomm.com>`.

## Prepare a Full PyTorch Checkout

The prepared worktree was cloned sparsely for investigation. Expand it before
building or running upstream tests.

```bash
cd /tmp/pytorch-upstream-naming
git sparse-checkout disable
git submodule update --init --recursive
```

## Build Current PyTorch Main

Use a dedicated environment. The QEff environment contains PyTorch
`2.13.0+cpu` and cannot validate the current PyTorch-main test suite.

```bash
cd /tmp/pytorch-upstream-naming
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --group dev
python -m pip install --no-build-isolation -v -e .
```

## Validate the Patch

Run the focused regression tests and lint after the build completes.

```bash
cd /tmp/pytorch-upstream-naming
source .venv/bin/activate
python test/onnx/exporter/test_core.py SemanticNodeNameTest
python test/onnx/exporter/test_api.py TestExportAPIDynamo.test_semantic_node_names_include_module_scope
spin quicklint
git status --short
git show --check HEAD
```

The serialized ONNX regression verifies that save/load preserves these names:

```text
/blocks/0/proj/linear__node_linear
/blocks/1/proj/linear_1__node_linear_1
```

## Push and Open the PR

Fork `pytorch/pytorch` in GitHub, then add the fork as a separate remote and
push only the prepared branch.

```bash
cd /tmp/pytorch-upstream-naming
git remote add fork git@github.com:<your-github-user>/pytorch.git
git push -u fork amitraj/dynamo-semantic-onnx-node-names
```

Open a PR from:

```text
<your-github-user>:amitraj/dynamo-semantic-onnx-node-names
```

into `pytorch/pytorch:main` with the prepared commit title.

The PR description must state that the change promotes existing
`pkg.torch.onnx.name_scopes` metadata into `NodeProto.name`, preserves the
generated node name as a uniqueness suffix, and does not change tensors,
edges, operators, weights, or function identities.

## Known Follow-up

PyTorch PR #197540 is required to preserve nested regions as local ONNX
functions during non-strict export. Once it lands, add an end-to-end local
`invoke_subgraph` naming regression. Do not claim that coverage in this PR
until that test has run.

QEff's post-export naming transform remains required as a compatibility
fallback and for QEff-specific subfunction normalization until an upgraded
PyTorch release has been validated with the four-layer Qwen QAIC configuration.
