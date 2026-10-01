# QRANIUMSW-64771: Missing Nodes in Qwen3.8 MDP Generation

Ticket: <https://jira-dc.qualcomm.com/jira/browse/QRANIUMSW-64771>

## Ticket Metadata

| Field | Value |
| --- | --- |
| Summary | Missing nodes in MDP generation of Qwen3.8 2.4T |
| Status | Open |
| Priority | P1 |
| Component | `MODEL_ONBOARDING` |
| Label | `onboarding` |
| Reporter | Chulhee Lee |
| Assignee | Anuj Gupta |
| Created | 2026-09-30 12:22:34 -0700 |
| Updated | 2026-09-30 15:45:40 -0700 |

The ticket was fetched through the authenticated tools endpoint on 2026-10-01.
There were no ticket comments at that time.

## Reported Problem

The report concerns a Qwen3.8-2.4T weight-free Dynamo ONNX export with ONNX
subfunctions, intended for a TS16 x PP6 deployment. When QEfficient generates
an MDP configuration from the compiler dump, nodes in that dump cannot be
found in the QEff MDP representation:

```text
WARNING - QEfficient.compile.mdp_generator - 138 compiler-dump nodes not found
in QEff MDP (compiler may have renamed them). Examples:
['model.layers.32.linear_attn._ones_lower', 'recurrent_state.42',
 'model.layers.46.linear_attn._ones_lower', 'recurrent_state.78',
 'model.layers.10.linear_attn._ones_lower']
```

The reporter states that the generated MDP file omits many nodes present in the
compiler-generated original MDP dump and that using the result then causes a
compile error.

## Reporter Reproduction

The reporter first generates a compiler MDP dump with `qaic-compile` against a
weight-free ONNX model, using retained state and ONNX subfunctions:

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=../Qwen3_5MoeForCausalLM/Qwen3_5MoeForCausalLM-77ec2736ade84c5b/Qwen3_5MoeForCausalLM.onnx \
  -aic-hw -convert-to-fp16 -aic-num-cores=4 \
  -aic-binary-dir=./partition_temp_qpc \
  -aic-perf-metrics -aic-perf-warnings -stats-level=50 -ddr-stats \
  -sub-functions -retained-state \
  -custom-IO-list-file=../Qwen3_5MoeForCausalLM/Qwen3_5MoeForCausalLM-77ec2736ade84c5b/qpc-0f8ab6e0154ed8d9/custom_io.yaml \
  -network-specialization-config=../Qwen3_5MoeForCausalLM/Qwen3_5MoeForCausalLM-77ec2736ade84c5b/qpc-0f8ab6e0154ed8d9/specializations.json \
  -mdp-dump-partition-config=./mdp_dump_partition.json
```

They then call `generate_disagg_mdp_config()` with:

```python
generate_disagg_mdp_config(
    onnx_path=".../Qwen3_5MoeForCausalLM.onnx",
    compile_dir=Path("qpcs"),
    mdp_ts_num_devices=96,
    mdp_num_partitions=6,
    mdp_strategy=MdpStrategy.INTERSECTION,
    mdp_compiler_dump_path=".../mdp_dump_partition.json",
    num_cores=4,
    num_layers=92,
)
```

## Attached Compiler Dump

Attachment: `mdp_dump_partition.json` (20,776 bytes).

The attached dump has two partitions and 311 node references. It contains a
mix of already-semantic operation names, semantic layer helper names, and
generated retained-state names. Examples:

```text
/model/layers.0/QEffQwen3_5MoeLinearDecoderLayer__node_invoke_subgraph__2
model.layers.0.linear_attn._ones_lower
recurrent_state.0
/lm_head/linear__node_linear
```

The layer-specific `_ones_lower` helper names do not have the slash-prefixed
ONNX naming convention used by the semantic Dynamo name transform. The
`recurrent_state.N` values are compiler/runtime retained-state identifiers,
not ordinary ONNX operator names. Both categories require explicit treatment
in the MDP node-matching logic; semantic `NodeProto.name` normalization alone
cannot guarantee those identifiers are present in the QEff ONNX node set.

## Relationship to QEff Naming Work

`DynamoSemanticNodeNameTransform` addresses generic Dynamo node names such as
`node_linear` by promoting `pkg.torch.onnx.name_scopes` into `NodeProto.name`.
It improves trace and MDP matching for ordinary ONNX operators and top-level
subfunction callsites. This ticket demonstrates the remaining boundary:
compiler-generated helper and retained-state identifiers must either be
represented in the QEff MDP model or deliberately ignored/translated before
set-intersection validation.

## Confirmed Root Cause And Fix

`generate_disagg_mdp_partition_config()` built its MDP node list from
`NodeProto.name` only. The missing ticket identifiers are not ONNX nodes:

- `recurrent_state.N` is a retained-state graph input.
- `model.layers.N.linear_attn._ones_lower` is a layer-scoped ONNX initializer.

Both are valid compiler partition entries. They were absent before the
intersection, so `INTERSECTION` discarded them and reported them as unmatched.
The issue is independent of semantic Dynamo node naming.

The minimal generator fix:

1. Recognizes layer indices in dotted `model.layers.N...` names and retained
   state names.
2. Adds retained-state graph inputs and layer-scoped initializers immediately
   before the first emitted node for the corresponding decoder layer.
3. Continues to exclude weight-free parameter graph inputs, which are external
   model data rather than compiler partition nodes.
4. Keeps unmatched compiler-generated names visible in the warning; it does
   not silence arbitrary name mismatches.

## Safe Reproduction And Validation

The local reproducer `repro/repro_QRANIUMSW-64771.py` builds a minimal ONNX
graph and compiler dump containing these two classes. It requires no model
weights, QAIC hardware, or QPC output. Before the fix, its expected MDP entries
are absent; after the fix it prints:

```text
QRANIUMSW-64771 regression check passed
```

The focused regression test is:

```bash
pytest -q tests/unit_test/base/test_modeling_qeff_base.py::TestMdpExternalGraphValues::test_intersection_keeps_layer_scoped_initializers_and_retained_states
```

It verifies retained states and layer initializers land in their matching
pipeline partitions, preserves their order before the decoder callsite, keeps
an external weight input excluded, and still warns for a deliberately unknown
compiler-only name.

The attached JIRA compiler dump was also compared in memory with the available
16-layer Qwen ONNX. Before the fix, 24 relevant dump entries were missing: 12
retained-state inputs and 12 `_ones_lower` initializers. After the fix, all 121
entries relevant to layers 0 through 15 were represented in the QEff MDP.

## Remaining Validation

Run the reporter's full 92-layer export and TS16 x PP6 MDP compile using the
same QEff/QAIC revisions. That remains a hardware and multi-terabyte-storage
validation; it is intentionally not part of the safe development loop.
