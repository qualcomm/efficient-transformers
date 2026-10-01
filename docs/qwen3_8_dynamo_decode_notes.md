# Qwen3.8 Dynamo Decode Engineering Log

This document records the Qwen/Qwen3.8-2.4T-A95B decode-only onboarding work.
Update it when an issue is reproduced, a root cause is confirmed, or a fix is
validated. Keep unverified hypotheses out of the confirmed-fix sections.

## Scope

- Model: `Qwen/Qwen3.8-2.4T-A95B`
- Architecture: Qwen3.5 MoE text model with hybrid decoder layers.
- Initial scope: decode-only Dynamo export, then weight-free export, then QAIC
  compilation and runtime parity.
- Target parity chain: HF PyTorch -> QEff PyTorch -> ONNX Runtime -> QAIC QPC.

## Validated Baseline

### Synthetic four-layer model, FP32

The synthetic hybrid model passed HF PyTorch -> QEff PyTorch -> ORT -> QAIC
parity with weight-free export, ONNX subfunctions, `kv_headpar`, and replicated
KV heads enabled.

Observed generated tokens:

```text
[112, 5, 3, 112]
```

Observed maximum differences:

```text
HF vs QEff PyTorch: 1.30e-7
HF vs ORT:          1.79e-7
QEff PyTorch vs ORT: 1.49e-7
```

### Original four-layer model, FP16

The following reduced configuration compiled and generated matching CPU HF and
QAIC tokens. QAIC automatic device selection must be used; do not supply
`--device-ids`.

```bash
python examples/dynamo/causal_lm/qwen3_8_2_4t_a95b_decode_dynamo.py \
  --no-use-synthetic-tiny \
  --hf-hub-cache /local/mnt/workspace/michchen/hf/hub \
  --torch-dtype float16 --num-hidden-layers 4 \
  --weight-free --use-onnx-subfunctions \
  --enable-blocking --blocking-mode kv_headpar \
  --num-kv-blocks 2 --headpar-split 4 --replicate-kv-heads \
  --ctx-len 128 --prefill-seq-len 1 --generation-len 4 \
  --num-devices 16 --num-cores 4
```

Observed tokens:

```text
HF CPU: [220, 220, 220, 220]
QAIC:   [220, 220, 220, 220]
```

## Confirmed Issues And Fixes

### 1. Hybrid decoder layers had incompatible nested-subfunction signatures

**Symptom:** Dynamo capture of decoder-layer ONNX subfunctions could fail or
fall back because Qwen3.5 mixes linear-attention and full-attention decoder
layers. Linear layers carry `conv_state` and `recurrent_state`; full-attention
layers carry `past_key` and `past_value`.

**Fix:** Use separate nested subfunction identities for linear and full-attention
decoder layers. Return updated linear-attention state explicitly across the
nested-function boundary and update it in the parent text-model forward path.

**Why:** A single decoder-layer identity cannot safely describe two captured
input/output signatures.

### 2. PyTorch invoke-subgraph cache reused an incompatible nested graph

**Symptom:** During Dynamo nested-subfunction export, PyTorch could reuse a
stale generated `subgraph_0` body for a later graph with different lifted
inputs. This prevented the Qwen graph from completing the normal
`torch.export.export(..., strict=False)` path.

**Fix:** `isolate_invoke_subgraph_cache()` temporarily disables the three
`torch._guards.InvokeSubgraphCache` lookups during `torch.onnx.export`. It is
locked and restores the original PyTorch methods after export.

**Scope:** Export only. It has no inference or QPC-runtime effect.

**Follow-up:** This is a PyTorch workaround. It can be removed when PyTorch
keys this cache by graph identity and lifted-input signature rather than a
locally reused generated subgraph identifier.

### 3. Prefill position IDs did not match the exported Qwen input contract

**Symptom:** The exported Qwen graph expects rank-3 `position_ids` with
position sections, while the prefill runtime path had not formatted them.

**Fix:** Call `_format_position_ids_for_session()` on the prefill path as well
as the decode path.

### 4. Four-layer long-context QPC packaging exhausted artifact storage

**Symptom:** A four-layer FP16 compile with `ctx_len=262144`,
`num_kv_blocks=8`, `kv_headpar`, replicated KV heads, and 16 devices failed
after graph compilation with:

```text
Error: failed to write payload to QPC segment constants.bin
```

**Root cause:** The compiler was packaging approximately 110 GB of constants
for each of 16 decode slices. The incomplete QPC was already about 1.5 TB, the
expected final `programqpc.bin` was about 1.77 TB, and the workspace had only
24 MB free.

**Resolution:** Use a fresh `QEFF_HOME` with more than 2 TB free for this
configuration. Compilation cannot resume from a partially written QPC.

**Status:** Storage limitation, not an ONNX export or numerical-parity failure.

### 5. Explicit QAIC device IDs caused runtime timeouts

**Symptom:** The successful 16-device QPC timed out when generation was given
an explicit `--device-ids 0 ... 15` list.

**Resolution:** Omit `--device-ids` and let QAIC choose devices automatically.

### 6. Original FP16 ORT loading has a CustomRMSNorm mixed-dtype gap

**Symptom:** The real four-layer FP16 ONNX passes structural checking but ORT
type inference/loading can fail around `CustomRMSNorm`, where FP16 variance is
combined with an FP32 epsilon.

**Status:** Open. QAIC QPC parity is currently the priority. Upstream Qwen
normalizes in FP32 and casts the result back to the input dtype; the custom-op
contract should eventually match that behavior.

### 7. A stale weight-free prepared checkpoint can be marked complete with missing shards

**Symptom:** The original 16-layer FP16 weight-free command exported Dynamo
ONNX successfully, then failed before compilation with:

```text
Could not resolve model initializer 'model.embed_tokens.weight' to a
safetensors checkpoint key
```

**Confirmed cause:** The prepared checkpoint directory
`...-qeff-prepared-float16-layers16-v2` had a completion sentinel and a
matching manifest, but its `model.safetensors.index.json` referenced 42 shards
while only 12 existed on disk. `model.embed_tokens.weight` was mapped to the
missing `model-00025-of-00213.safetensors`. The original HF snapshot contained
all 213 shards, so this was not a model-download or ONNX-export failure.

**Recovery:** Delete only the incomplete prepared-checkpoint directory and
rerun. A correct 16-layer prepared checkpoint needs about 844 GiB for the
referenced source shard set; ensure sufficient free workspace capacity before
rebuilding.

**Generic follow-up:** `CheckpointTransformPipeline` currently trusts a
matching manifest plus `.checkpoint_prepared`. It should additionally validate
that every shard named by the prepared `model.safetensors.index.json` exists
before reusing a cached prepared checkpoint. When validation fails, it should
clear and rebuild that cache automatically.

## Open Issue: Dynamo ONNX Node Names For MDP And Trace Analysis

### Reported behavior

The full-model Dynamo export produces generic ONNX node names such as:

```text
node_linear
node_add_279
node_invoke_subgraph_17__2
```

This happens both with and without ONNX subfunctions. It prevents creating
layer-aware MDP partition files and makes compiler traces hard to map to
PyTorch operations.

### Confirmed cause

This is a Dynamo FX-to-ONNX naming behavior, not a QAIC rename and not a
weight-free rewrite. Dynamo functionalizes module calls, names FX nodes after
operator targets such as `aten.linear`, then emits unique ONNX names such as
`node_linear`.

The meaningful information is present in ONNX node metadata, for example:

```text
pkg.torch.onnx.name_scopes:
['', 'model', 'model.layers.17',
 'model.layers.17.linear_attn.in_proj_qkv', 'linear']
```

The QAIC compiler and QEff MDP generator consume `NodeProto.name`, not this
metadata. `preserve_subfunction_source_lines()` keeps metadata while retracing
nested functions but does not rewrite `NodeProto.name`.

### Subfunction constraint

The top-level ONNX subfunction callsite has the actual layer scope and can be
named, for example:

```text
/model/layers.17/decoder_layer
```

The reused ONNX function body must remain layer-independent. Its nodes cannot
be named `layers.0/...`, because that same function is invoked by many layers.
Use semantic names such as:

```text
decoder_layer/linear_attn/in_proj_qkv/linear
```

Compiler traces can combine the semantic callsite and semantic function-body
node to identify the concrete PyTorch operation.

### Implemented generic fix

`DynamoSemanticNodeNameTransform` is a final post-Dynamo ONNX transform. It:

1. Reads `pkg.torch.onnx.name_scopes`.
2. Renames `NodeProto.name` only; tensor names, operator names, function names,
   metadata, and weights remain unchanged.
3. Produces unique compiler-safe names using the semantic scope plus the
   original generated node name as a suffix.
4. Applies to normal Dynamo and weight-free Dynamo exports, with and without
   ONNX subfunctions.
5. Keeps reused function-body names layer-agnostic and gives top-level
   callsites their actual `layers.N` path.

It is exporter-level and applies to every Dynamo model, including weight-free
export, without a Qwen-specific class, layer count, cache, or architecture
condition. A Dynamo-only export-hash version invalidates existing ONNX files
that contain generic node names.

Validated in memory against the four-layer Qwen ONNX: callsites become names
such as `/model/layers.0/QEffQwen3_5MoeLinearDecoderLayer/...`, while reused
function bodies remain layer-independent.

### Four-layer QAIC validation

The following fresh weight-free export, compile, and QAIC run passed on 16
auto-selected devices. `HF_HUB_CACHE` must be set before Python imports
QEfficient, because weight-free checkpoint resolution reads the cache location
at import time.

```bash
sg qaic -c 'HF_HUB_CACHE=/local/mnt/workspace/michchen/hf/hub \
HF_HUB_OFFLINE=1 \
QEFF_HOME=/local/mnt/workspace/amitraj/Pmodels/efficient-transformers/.qeff_qwen_node_names_4l \
/local/mnt/workspace/amitraj/Pmodels/env/qeff_qwen_3/bin/python \
/local/mnt/workspace/amitraj/Pmodels/efficient-transformers/examples/dynamo/causal_lm/qwen3_8_2_4t_a95b_decode_dynamo.py \
--qeff-home /local/mnt/workspace/amitraj/Pmodels/efficient-transformers/.qeff_qwen_node_names_4l \
--no-use-synthetic-tiny --hf-hub-cache /local/mnt/workspace/michchen/hf/hub \
--torch-dtype float16 --num-hidden-layers 4 --weight-free --use-onnx-subfunctions \
--enable-blocking --blocking-mode kv_headpar --num-kv-blocks 2 --headpar-split 4 \
--replicate-kv-heads --num-devices 16 --num-cores 4 \
--ctx-len 128 --prefill-seq-len 1 --generation-len 4'
```

Results:

- Dynamo used `torch.export.export(..., strict=False)` and completed in
  46.10 seconds; checkpoint preparation completed in 0.02 seconds.
- QAIC compilation completed in 588.75 seconds. The QPC is at
  `.qeff_qwen_node_names_4l/Qwen3_5MoeForCausalLM/Qwen3_5MoeForCausalLM-f1d9c835c8ff950a/qpc-3b48e7d709d61364/qpc`.
- The saved ONNX passes `onnx.checker` and has 86 inputs, 9 outputs, 11
  initializers, 6 local functions, and four semantic top-level subfunction
  calls. The callsites identify layers 0 through 3; no local function node
  contains a layer-specific scope.
- QAIC produced `[220, 220, 220, 220]`, exactly matching the original CPU
  reference for the same prompt and generation length.

Remaining validation: generate a reduced QAIC compiler dump, then verify the
semantic names are preserved in the compiler IR before attempting a full
92-layer PP/TS run.

## Configuration Notes

- `kv_headpar` with no explicit `--headpar-split` defaults to four for the
  current example, matching `--num-cores 4`; pass `--headpar-split 4`
  explicitly in reproducibility commands.
- Replicated KV heads are required for the validated 16-device hybrid Qwen
  configuration. Without replication, compiler MDP split planning reported a
  retained-state split-count mismatch for `conv_state.2`.
- A four-device QPC configuration is capacity-limited for this model. The
  observed requirement was roughly 58 GB/device versus about 31 GB available.
- A TS16 x PP6 deployment requires 96 devices. The JIRA PP2 example uses two
  host-connected devices and the reported `--num-devices 16` command by itself
  does not describe TS16 x PP6.
