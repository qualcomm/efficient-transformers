Preserve semantic Dynamo ONNX node names.

Dynamo ONNX export records the originating PyTorch module path in
`pkg.torch.onnx.name_scopes`, but emits generic node names such as
`node_linear`. Downstream trace and partitioning tools consume `NodeProto.name`
and therefore cannot map an ONNX operation back to its module path.

Promote existing module-scope metadata into `NodeProto.name`. Retain the
generated node name as a suffix so that nodes remain unique when one FX node
lowers to multiple ONNX nodes. Nodes without a module stack retain their
existing names.

This changes node labels only. Tensor names, graph edges, operators, weights,
function identities, and metadata remain unchanged.

Tests:
- Python syntax compilation of the changed exporter and tests.
- Reduced Dynamo export with repeated linear blocks.
- ONNX save/load validation confirming serialized `ModelProto.graph.node`
  names are semantic and unique.

The focused current-main PyTorch test suite still needs to run on a full
PyTorch-main build before this draft is marked ready for review.

AI assistance was used in preparing this patch; the author reviewed the diff
and validation results.
