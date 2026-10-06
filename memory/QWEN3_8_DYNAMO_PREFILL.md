# Qwen3.8 Dynamo Prefill Sequence Shapes

## Current Finding

The Qwen3.8 weight-free Dynamo export requests a dynamic `seq_len` for
prefill, but the emitted ONNX graph specializes the token axis to its example
length. A synthetic four-layer prefill export requested
`input_ids[1] = Dim("seq_len", max=128)` and produced `input_ids` with shape
`[batch_size, 32]`.

This is not an ONNX-axis configuration omission. The Qwen Gated Delta Network
(GDN) prefill implementation needs the sequence length as Python control flow:

```python
for i in range(total_sequence_length // chunk_size):
```

The loop is in
`QEfficient/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` within
`torch_chunk_gated_delta_rule_qeff`. Dynamo cannot preserve a symbolic loop
trip count in the normal ONNX dataflow graph, so it specializes the input to
the example sequence length.

## Recommended Direction

Use fixed prefill specializations (`prefill_seq_len` values such as 64, 128,
or 256) and runtime prompt chunking for the initial QAIC prefill enablement.
This preserves GDN numerical behavior and matches QAIC's fixed-shape QPC
specializations.

A truly dynamic prefill ONNX needs a separate design: a compiler-supported
symbolic scan/control-flow implementation or a GDN custom op. Do not merely
rewrite the ONNX input dimension metadata because internal chunk operations
remain specialized to the example length.

## Follow-up

Investigate why a default decode export may be observed with `seq_len=8`.
Decode-only Qwen export with `prefill_seq_len=1` is expected to produce a
static one-token ONNX input.

## Decode KV-Head-Parallel Export Fix

The decode `seq_len=8` observation was caused by shared CausalLM export code,
not Qwen Dynamo. Before creating the example inputs,
`QEFFAutoModelForCausalLM.export` rounded `seq_len` up to the largest blocking
count for every blocking mode. Thus `KV_HEADPAR` with `num_kv_blocks=8`
incorrectly changed the Qwen decode example from one token to eight.

KV-head parallelism partitions the cache/head axis only; it has no query-token
loop that requires the decode sequence dimension to be divisible by the KV
block count. The export path now reserves that early sequence-length change for
paged attention, whose cache layout genuinely needs a fixed KV block size.
Query-axis blocking continues to apply its existing padding later in export.

Validated with a fresh synthetic four-layer Qwen3.8 Dynamo, weight-free,
ONNX-subfunction export using `KV_HEADPAR`, `num_kv_blocks=8`,
`headpar_split=4`, and replicated KV heads. The emitted ONNX inputs were:

```text
input_ids    [batch_size, 1]
position_ids [4, batch_size, 1]
```

## Example Artifact Directory Caveat

The example's `--qeff-home` argument currently updates `os.environ` too late
to select the export directory. `QEfficient.utils.cache.QEFF_HOME` is resolved
when the package is imported, which happens before argument parsing. Set
`QEFF_HOME` in the shell before starting Python until the example's CLI option
is corrected:

```bash
export QEFF_HOME=.qeff_qwen_run
python examples/dynamo/causal_lm/qwen3_8_2_4t_a95b_decode_dynamo.py ...
```

## Decode Parity Record

The fresh four-layer synthetic weight-free export from the decode KV-head
parallel fix was compared directly against the original HF PyTorch checkpoint
for four retained-state decode steps. Both produced:

```text
[112, 5, 3, 112]
```

The largest absolute logit difference across the four steps was
`1.94e-7` in float32. This validates HF PyTorch to weight-free ONNXRuntime
parity for the fixed `seq_len=1` graph.

QAIC runtime was subsequently run from a `newgrp qaic` shell using the QPC
compiled from the same artifact. Its first four generated token IDs were also
`[112, 5, 3, 112]`, completing HF PyTorch -> ONNXRuntime -> QAIC token parity
for this synthetic four-layer configuration.
