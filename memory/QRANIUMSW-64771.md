# QRANIUMSW-64771: Qwen3.8 QPC Constant Packaging

Ticket: <https://jira-dc.qualcomm.com/jira/browse/QRANIUMSW-64771>

## Access Status

On 2026-10-01, the local shell was redirected to Qualcomm SSO when requesting
the JIRA REST endpoint. A subsequent Bearer-token request also returned an SSO
redirect (`HTTP 302`) with no ticket JSON. The ticket's title, fields,
description, comments, and attachments were therefore not retrieved. Add those
details from an authenticated JIRA session before treating this note as a
complete ticket summary.

## Confirmed Local Evidence

This record is associated with the Qwen3.8-2.4T-A95B 16-layer weight-free
decode compile investigated locally. It uses:

```text
dtype:              float16
layers:             16
weight-free:        enabled
ONNX subfunctions:  enabled
blocking:           kv_headpar
num_kv_blocks:      8
headpar_split:      4
replicate KV heads: enabled
devices:            16
cores/device:       4
ctx_len:            262144
prefill_seq_len:    1
```

The Dynamo ONNX export and the rebuilt weight-free checkpoint preparation
completed. QAIC then emitted 16 compiled decode functions, `Decode_slice00`
through `Decode_slice15`. Each function's QPC `constants.bin` was about
568.5 GB:

```text
568,525,212,116 bytes per slice
```

The compiler's planned `programqpc.bin` offset after all 16 slices was about
9.10 TB. The compile failed while writing `constants.bin` once the workspace
filled, with:

```text
Failure while writing constants.bin to QPC.
```

This was not a Dynamo export, ONNX correctness, or numerical-parity failure.
`ctx_len` affects retained KV-state capacity, but the dominant artifact size is
the repeated model constants materialized for all 16 compiled MDP slices.

## Related Checkpoint Cache Incident

An earlier 16-layer run failed before compilation because its prepared
weight-free checkpoint had a completion sentinel but was incomplete: its index
referenced 42 shards while only 12 existed. Removing that prepared directory
allowed a correct rebuild with all 42 required shards. This is a separate
checkpoint-cache validation issue, not the QPC packaging failure described
above.

## Open Questions

- Is duplication of approximately 568.5 GB of constants per MDP decode slice
  expected for this compiler configuration?
- Can the compiler share or deduplicate immutable constants across the 16
  slices while preserving device placement and runtime semantics?
- What QPC storage estimate should users apply before attempting this class of
  multi-device compile?
