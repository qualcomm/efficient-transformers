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

## Prefill Enablement Plan

### Goal

Enable the normal causal-LM serving shape for Qwen3.8: one QPC containing a
fixed-size prefill specialization and a one-token decode specialization. At
runtime, arbitrary prompt lengths are padded and chunked into the configured
prefill size, then generation continues through decode. This is the same
servable contract used by other causal-LM models; it is not a promise that one
ONNX graph accepts arbitrary token-axis lengths.

The initial target is Dynamo export, then ONNXRuntime, then QAIC. Weight-free,
ONNX subfunctions, KV blocking/head parallelism, and KV replication are added
only after the preceding baseline is numerically verified.

### What Already Exists

- `QEffQwen3_5MoeForCausalLM` exposes retained state for both layer families:
  `past_key`/`past_value` for full attention and `conv_state`/`recurrent_state`
  for GDN linear attention.
- `QEffQwen3_5MoeGatedDeltaNet.forward` computes a chunked GDN result for
  prefill and a recurrent GDN result for decode. The selected branch is based
  on `position_ids`, and both paths return the retained state needed by the
  next call.
- Causal-LM runtime input preparation already pads prompts and dispatches
  fixed-size prefill chunks before decode. Padding positions use `position_id
  = -1`, which the Qwen cache and GDN paths must preserve as invalid tokens.
- `get_onnx_export_seq_len` keeps the decode export at one token only when
  `prefill_seq_len == 1`; a prefill export can retain its requested fixed
  specialization length.

### Progress Log

#### 2026-10-08: Tiny PyTorch Prefill Baseline

The first direct construction of `QEffQwen3_5MoeForCausalLM` failed before
execution because QEff model replacement/initialization must be performed by
`QEFFAutoModelForCausalLM`. This is expected usage, not a prefill defect.

The supported QEff auto-model path then exposed the first actual prefill
defect: `QEffQwen3_5MoeDynamicCache` creates `None` GDN state slots, while
`QEffQwen3_5MoeGatedDeltaNet.forward` assumed retained `conv_state` and
`recurrent_state` were already present. This caused eager PyTorch prefill to
fail at `conv_state.ndim` before any export.

The fix lazily initializes zero GDN state on the first eager cache update,
using the same shapes and dtypes as the retained-state export contract. The
export/runtime path remains unchanged because it supplies concrete state input
tensors. Follow-up validation must prove this produces HF-equivalent prefill
and subsequent decode logits before Dynamo work begins.

Validation completed for a synthetic four-layer model with three GDN layers,
one full-attention layer, a 64-token prefill, and one decode token:

```text
HF PyTorch vs QEff PyTorch prefill last-logit max diff: 2.68e-7
HF PyTorch vs QEff PyTorch decode logit max diff:       1.34e-7
prefill greedy token:                                  87 == 87
decode greedy token:                                   78 == 78
```

The QEff side used runtime-shaped retained inputs: a 128-token preallocated
full-attention KV cache and concrete GDN states. This is intentional: QEff's
shared `QEffDynamicLayer` is a fixed-capacity retained-state cache for
export/runtime, rather than an appendable eager cache. A 64-token GDN-only
regression test also now verifies lazy cache initialization and matches
upstream GDN output within `5.22e-7`.

#### 2026-10-08: Dynamo Prefill Export Length

Synthetic Dynamo export with `prefill_seq_len=64` initially produced static
`input_ids` and `position_ids` token dimensions of 32. The Qwen
`get_onnx_export_seq_len` override returned the generic export default for all
prefill lengths greater than one, so it discarded the requested specialization
length before dummy-input creation.

The override now returns the requested `prefill_seq_len` whenever supplied.
This deliberately creates a fixed 64-token prefill graph; it does not claim
the GDN graph is dynamic. A regression assertion covers both decode length 1
and prefill length 64. The next validation is a fresh Dynamo export confirming
the ONNX input dimensions are 64, followed by ONNXRuntime retained-state
parity.

The fresh export completed with the expected interfaces:

```text
input_ids:     [batch_size, 64]
position_ids:  [4, batch_size, 64]
```

ONNXRuntime was run with the same fixed retained-state layout used by QAIC: a
128-token full-attention cache plus GDN conv/recurrent inputs. It matched QEff
PyTorch as follows:

```text
logits:                         2.38e-7 max absolute difference
all GDN conv states:            <= 1.49e-7
all GDN recurrent states:       <= 1.80e-9
full-attention key cache:       1.19e-6
full-attention value cache:     2.98e-7
```

This completes fixed-prefill Dynamo -> ONNXRuntime parity. The next boundary
is compiler specialization: determine whether a graph statically unrolled for
64 tokens can validly provide both 64-token prefill and one-token decode
specializations in one QPC. If it cannot, prefill and decode require distinct
ONNX/QPC artifacts (or a future symbolic GDN implementation).

#### 2026-10-08: Single-QPC Specialization Result

The compiler test answered this decisively. The ONNX graph exported for
`prefill_seq_len=64` has literal token dimensions of 64, while compilation was
asked for these two specializations:

```json
{
  "Prefill": {"seq_len": 64, "ctx_len": 128, "batch_size": 1},
  "Decode":  {"seq_len": 1,  "ctx_len": 128, "batch_size": 1}
}
```

QAIC rejected it with `No input that uniquely identifies specialization`.
There is no symbolic `seq_len` input dimension left after Dynamo specialized
the GDN Python loop, so QAIC cannot distinguish those entries. This is not a
compiler bug and must not be worked around by relabeling ONNX shape metadata.

The correct near-term serving design is two fixed graphs/QPCs: a 64-token
prefill QPC and a one-token decode QPC. They exchange the identical retained
state tensors. The next experiment is to compile both independently, then add
the minimal causal-LM runtime dispatch that owns one QAIC session per QPC and
passes retained-state outputs from prefill into decode.

The prefill-only compile was then run with `prefill_only=True` and completed
successfully in 12.64 seconds. Its specialization file contains exactly one
entry:

```json
{"Prefill": {"batch_size": 1, "ctx_len": 128, "seq_len": 64}}
```

This is now the required prefill invocation. Do not compile Qwen GDN with
`prefill_seq_len > 1` and `prefill_only=False`, because that requests an
invalid mixed prefill/decode specialization list from one static ONNX graph.
Decode remains independent: it uses `prefill_seq_len=1`, produces its existing
one-token graph/specialization, and its focused export regression test still
passes after the prefill changes.

The prefill QPC was executed on QAIC with the same 64-token input and
128-token retained-state buffers used for ONNXRuntime. It produced the same
greedy token and top-5 token ordering as ORT:

```text
ORT top-5:  [69, 63, 18, 111, 92]
QAIC top-5: [69, 63, 18, 111, 92]
```

The QAIC QPC was compiled with fp16 conversion, while the ORT reference graph
is float32. The maximum logit difference was `0.0125`; GDN recurrent-state
differences stayed below `1.95e-4`. The larger full-attention key-cache
difference (`0.175`) is an fp16 accumulation difference, not an I/O contract
mismatch. The matching greedy/top-k result is the relevant prefill acceptance
signal at this dtype. A future end-to-end test must feed these QAIC prefill
states into the separate decode QPC and verify generated tokens across both
sessions.

#### 2026-10-08: Decode Non-Regression Check

After the prefill changes, a fresh synthetic decode artifact was exported with
`prefill_seq_len=1`. It retained the established one-token interface:

```text
input_ids:     [batch_size, 1]
position_ids:  [4, batch_size, 1]
```

Its separate one-token QPC compiled and executed on QAIC. ORT and QAIC both
selected greedy token `112`; their float32-ORT to fp16-QPC maximum logit
difference was `0.0106`. This confirms the prefill-specific export-length and
first-cache-update changes do not change the decode artifact contract.

### Constraints That Drive The Design

1. GDN's `torch_chunk_gated_delta_rule_qeff` pads to `gdn_chunk_size` and has
   a Python loop over `total_sequence_length // chunk_size`. Dynamo therefore
   specializes that loop to the example prefill length. Marking ONNX metadata
   as dynamic cannot make this graph correct for a different number of GDN
   chunks.
2. Use fixed prefill lengths that are multiples of the GDN chunk size. Start
   with `gdn_chunk_size=64` and `prefill_seq_len=64`; add `128` only after the
   64-token path is verified. This makes the captured chunk count explicit and
   avoids an additional internal pad case in the main specialization.
3. A QPC must contain both `(batch, 64)` prefill and `(batch, 1)` decode
   specializations. A prefill-only QPC is insufficient because generation must
   continue from the states produced during prefill.
4. Preserve the current decode contract and artifact identity. `prefill_seq_len`
   and other compile options must remain part of ONNX/QPC identity. Prepared
   weight-free checkpoint identity remains independent of layer count but
   continues to differ for dtype and KV-replication layout.

### Staged Work

#### 1. Establish a Tiny PyTorch Oracle

Use the synthetic four-layer configuration because it contains both three GDN
layers and one full-attention layer. Exercise:

- a single 64-token prefill;
- a 128-token prompt, proving multiple GDN chunks;
- non-multiple prompt lengths such as 65 and 129 through the runtime padding
  path; and
- at least several one-token decode steps after each prefill.

Compare HF PyTorch with QEff PyTorch at the last valid prefill logit and every
decode logit. Also compare retained state by layer family: full-attention KV
cache, GDN convolution state, and GDN recurrent state. This isolates semantic
errors before Dynamo or QAIC is involved.

#### 2. Prove Fixed-Shape Dynamo and ONNXRuntime Parity

Export the same tiny model with `dynamo=True`, `prefill_seq_len=64`, and no
subfunctions, weight-free, blocking, or KV replication initially. Verify that
the ONNX specialization list contains both sequence lengths 64 and 1, and that
the prefill graph input is statically 64, not incorrectly marked as generally
dynamic.

Run ONNXRuntime prefill once, feed its retained-state outputs into ONNXRuntime
decode calls, and compare the same logits/states from step 1. Inspect the
graph for the expected GDN unrolled 64-token chunk behavior and for stable
retained-state input/output names.

#### 3. Compile and Run the Baseline QPC

Compile the tiny artifact without optional optimizations. Run prompts of 64,
65, 128, and 129 tokens so runtime dispatch proves both a full prefill chunk
and a padded final chunk. Compare QAIC prefill/decode token IDs and saved
logits against the PyTorch oracle. Confirm the runtime selects the prefill
specialization for chunks and the decode specialization afterwards.

#### 4. Add Features One Boundary at a Time

For every successful baseline, repeat the exact prefill-plus-decode parity
case before moving to the next row:

| Stage | Additions |
| --- | --- |
| A | ONNX subfunctions |
| B | Weight-free checkpoint loading |
| C | KV blocking |
| D | KV-head parallelism |
| E | Replicated KV heads |

At stages C through E, inspect retained-state custom I/O, MDP partitions, and
the GDN conv/recurrent-state intersections. Replication must remain limited to
full-attention K/V projections; GDN states are not replicated.

#### 5. Validate the Real Checkpoint Incrementally

Use the actual Qwen3.8 checkpoint with four configured layers first, the same
64-token prefill plus decode scenario, and the intended final dtype. Capture
CPU logits before QAIC execution. Then validate 16 devices with the desired
blocking/KV-head-parallel settings. Only after that passes, increase the export
depth and finally run all 92 layers.

Set `QEFF_WF_HOME` to a shared high-capacity location before real weight-free
runs. The full converted checkpoint is intentionally reused across 4/16/92
layer exports, but normal and KV-replicated layouts require separate full
prepared checkpoints.

### Required Code Review Points Before Implementation

- Trace the causal-LM `compile` path to verify it emits both specializations
  for this text-only wrapper, rather than borrowing the VLM-only specialization
  code later in the same modeling file.
- Verify `prepare_inputs_for_generation` sends the correct Qwen MRoPE
  `position_ids` and preserves `-1` padding through GDN state updates.
- Confirm the model's last-valid-token logit selection uses the same position
  convention after a padded prefill chunk.
- Keep any Qwen-specific changes in `modeling_qwen3_5_moe.py`; promote only
  genuinely generic prefill dispatch or export fixes into `modeling_auto.py`.
- Do not solve the dynamic-axis limitation by changing ONNX shape metadata.
  That would create a graph whose declared input contract disagrees with its
  unrolled GDN computation.

### Acceptance Criteria

For a fixed `prefill_seq_len=64`, HF PyTorch, QEff PyTorch, ONNXRuntime, and
QAIC agree on last-valid prefill and subsequent decode logits within the
dtype-appropriate tolerance, and generate identical greedy token IDs. The
same holds for padded prompt lengths. The produced QPC contains valid 64-token
prefill and one-token decode specializations, including the final enabled
weight-free/subfunction/KV-head-parallel configuration.

### Deferred Work

True arbitrary dynamic prefill sequence length in a single ONNX graph remains
deferred. It requires replacing the Python GDN chunk loop with a compiler-
supported symbolic control-flow/scan design or a dedicated GDN custom op; it
is not required for normal fixed-chunk serving.

## Latest Validation

### 2026-10-08: Disaggregated Synthetic Runtime Example

Added `examples/dynamo/causal_lm/qwen3_8_2_4t_a95b_disagg_dynamo.py` for the
fixed-shape disaggregated path. It compiles two independent Dynamo QPCs: a
`prefill_only=True` QPC at the requested fixed prefill length and a
`prefill_only=False`, one-token decode QPC.

The runner initializes retained-state inputs from the QPC binding metadata and
hands state through host NumPy buffers. It does not hard-code the Qwen hybrid
cache shapes. The transferred state comprises `past_key`/`past_value` for
full-attention layers and `conv_state`/`recurrent_state` for GDN layers.

The initial validation used the normal weighted baseline. Weight-free
prefill was enabled and validated subsequently.

Validated on QAIC with the synthetic four-layer mixed layout, float32 source
weights, `prefill_seq_len=64`, `ctx_len=128`, and fp16 QPC compilation:

```text
prompt length 2:  HF PyTorch == QAIC == [26, 92, 78, 79]
prompt length 65: HF PyTorch == QAIC == [127, 33, 33, 33]
```

The 65-token case runs two prefill invocations before decode, so it verifies
both same-worker retained-state reuse and the prefill-to-decode handoff.

### 2026-10-08: Disaggregated Artifact Layout

The disaggregated example follows the standard process-level `QEFF_HOME`
contract. It does not accept a separate compile output directory: `QEFF_HOME`
must be set before Python starts, because QEff resolves the artifact root at
import time. Prefill and decode have different export hashes, so their ONNX
and QPC artifacts naturally remain separate under that one root. Passing a
custom `compile_dir` split QPCs away from their ONNX artifacts and was removed.

### 2026-10-08: Disaggregated ONNX Inspection

The normal weighted synthetic artifacts under `.new_testing_dissag` have the
expected distinct contracts:

- prefill (`Qwen3_5MoeForCausalLM-40bfcbec4c02e1c4`): `input_ids` is
  `[batch_size, 64]` and `position_ids` is `[4, batch_size, 64]`;
- decode (`Qwen3_5MoeForCausalLM-803261805afad19e`): `input_ids` is
  `[batch_size, 1]` and `position_ids` is `[4, batch_size, 1]`.

Both have the same ten hybrid retained-state inputs and matching outputs:
three GDN `conv_state`/`recurrent_state` pairs plus full-attention
`past_key.3`/`past_value.3`. Structural ONNX validation passes for both, and
both compile and run on QAIC.

Raw standard ONNX Runtime loads decode directly, but rejects prefill during
shape inference. The shared `CtxGather3D`/`CtxScatter3D` ONNX function bodies
feed `int32` indices to standard `GatherND`/`ScatterND`, whose ONNX contract
requires `int64`. QEff's existing ORT quickcheck helper works around this by
casting those function-local indices and replacing invalid `INT32_MAX` padded
indices with zero. Applying that normalization in memory made both graphs pass
full ONNX checking and generated `[26, 92, 78, 79]`, matching QAIC and HF
PyTorch. This is a shared custom-op export compliance issue, not a Qwen model
or disaggregated state-handoff issue; fix it in `ctx_scatter_gather.py` rather
than modifying generated ONNX files.

The graph-level optimization audit for this baseline found:

- both exports use Dynamo and semantic node names, but module-level ONNX
  subfunction export is disabled;
- prefill has 8,907 graph nodes and no `Loop`/`If`: the 64-token GDN path is
  statically unrolled as required; decode has 8,623 nodes and a static
  one-token input for recurrent GDN execution;
- `CustomRMSNorm` is used fourteen times, MoE dispatch uses the QEff custom
  context gather/scatter operators, and `KVCacheExternalModuleMapperTransform`
  produces explicit hybrid retained state;
- the output is already reduced to the last valid token
  `[batch_size, 1, vocab_size]`, rather than returning logits for every input
  token;
- this run has no ONNX module subfunctions, weight-free external weight data,
  MXFP6, MXINT8 cache, KV blocking, KV-head parallelism, KV replication, or
  multi-device MDP. `num_replicate_kv_heads=1`, `mdp_num_partitions=1`, and
  `mdp_ts_num_devices=1` confirm that it is the clean baseline;
- compiler conversion to fp16 is enabled, but hybrid retained-state custom I/O
  remains `float`, matching the ONNX float32 source-state contract.

### 2026-10-08: Disaggregated ONNX Subfunction Validation

The synthetic disaggregated example was re-run with
`--use-onnx-subfunctions`. Both QPCs compiled with `-sub-functions` and the
QAIC prefill/decode handoff generated `[26, 92, 78, 79]`, matching the
non-subfunction run and HF PyTorch.

Both exported ONNX graphs contain the expected local decoder functions and
call them once per configured layer: three calls to
`QEffQwen3_5MoeLinearDecoderLayer` and one call to
`QEffQwen3_5MoeFullAttentionDecoderLayer`. ONNX represents these as local
function-call nodes with the function name as their operator type, not as an
`InvokeSubgraph` node. Prefill retains the static 64-token interface and
decode retains the static one-token interface; their hybrid retained-state
I/O contracts remain identical.

### 2026-10-08: Weight-Free Disaggregated Prefill Validation

The branch-only `NotImplementedError` that rejected
`weight_free=True, prefill_only=True` was removed to align with mainline.
The disaggregated example now defaults to weight-free mode and supports
`--no-weight-free` for the original weighted baseline. Synthetic weight-free
checkpoint generation uses a deterministic, dtype/layer/seed-specific source
checkpoint directory so independent prefill and decode model instances share
the same weights.

The four-layer synthetic run with `--use-onnx-subfunctions` produced a
weight-free prefill ONNX/QPC and a separate weight-free decode ONNX/QPC. Both
compiled with `-sub-functions`, each has a `weight_spec.json`, and direct
QAIC host-state handoff generated:

```text
HF PyTorch == weight-free QAIC == [26, 92, 78, 79]
```

This establishes that weight-free disaggregated prefill is no longer blocked
by infrastructure and works with the Qwen hybrid retained-state contract.

### 2026-10-08: Fully Optimized Decode Regression

The existing decode-only example was validated with the full currently
supported decode configuration: Dynamo, weight-free export, ONNX subfunctions,
KV-head-parallel blocking (`num_kv_blocks=8`, `headpar_split=4`), and
replicated KV heads. The synthetic four-layer QPC was compiled for four
devices and executed explicitly on devices `0 1 2 3` with `ctx_len=128`.

```text
HF PyTorch original == optimized weight-free QAIC == [112, 5, 3, 112]
```

This is decode-only validation; the KV-head-parallel and replicated-KV cache
layout has not yet been validated for the disaggregated prefill-to-decode
handoff.
