# Qwen3.8 Flash four-layer QSA example

This example exports, compiles, and executes a four-layer Qwen3.8-Flash-Next
decode model through QEfficient. It uses the Qwen4-Exp wrapper and the migrated
QSA path.

The tested configuration is:

- four leading language-model layers;
- decode-only, `batch_size=16`, `ctx_len=65536`;
- indexer DP16/CP1 and attention DP16/CP1;
- 16 QAIC devices and 4 cores per device;
- FP16 matmul compilation (`mxfp6_matmul=False`);
- MXINT8 custom IO for every GDN, PLE, and QSA retained-state input/output;
- weight-free export with host-provided `ngram_embeddings`.

The 256K configuration does not compile on the current layout because the
compiler exhausts per-core VA space in the QSA retained-state quantization
buffers. 64K is the compiled configuration used by the commands below.

## Environment and local snapshot

Use the Qwen3.8 virtual environment:

```bash
VENV=/home/ochougul/envs/qwen38f
SNAPSHOT=/home/huggingface_hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540
```

These commands expect a completed local snapshot. They do not download model
weights. If the snapshot is not already available, configure the approved
`HF_HUB_CACHE` before obtaining it.

## Step 1: create the four-layer checkpoint

The pruning helper reads the safetensors index shard-by-shard, keeps the first
four language-model layers and `lm_head`, and excludes the host-side PLE table.
It also copies `config.json` into the output directory so the pruned checkpoint
is independently loadable.
Run:

```bash
"$VENV/bin/python" examples/text_generation/qwen4_exp_create_4layer_checkpoint.py \
  --snapshot "$SNAPSHOT" \
  --output qwen3.8_flash_experiment/example_repro_20261013/qwen38_flash_4layer_checkpoint_v2
```

The resulting checkpoint is intentionally local and is not a source-controlled
artifact; it is approximately 23 GB for this model window.

## Step 2: compile and execute the four-layer model

To export and compile a fresh QPC with the configuration above, run:

```bash
sg qaic -c '/home/ochougul/envs/qwen38f/bin/python examples/text_generation/qwen4_exp_dp16_decode_qpc.py \
  --snapshot /home/huggingface_hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --checkpoint qwen3.8_flash_experiment/example_repro_20261013/qwen38_flash_4layer_checkpoint_v2 \
  --compile-dir qwen3.8_flash_experiment/wf_export_debug_boundary_v3'
```

The script uses `.compile()` with `dynamo=True`, `weight_free=True`, 16 devices,
4 cores per device, `-retained-state`, FP16 matmul compilation,
`-allow-mxint8-mdp-io`, and MXINT8 custom IO for all retained-state pairs.
The generated ONNX contains `CtxPagedScatter` and `CtxGatherRows` function
definitions whose bodies lower to `ScatterND` and `GatherND`.

## Step 3: generate opstats

The profiling QPC is compiled directly from the freshly generated ONNX. The
runner must use the profiling QPC, and the selected 16-device group must have a
working P2P fabric. On the tested host, devices `0..15` were available:

```bash
sg qaic -c '/home/ochougul/envs/qwen38f/bin/python \
  examples/text_generation/qwen4_exp_dp16_decode_qpc_opstats.py \
  --qpc qwen3.8_flash_experiment/wf_export_debug_boundary_v3/profiling_qpc \
  --onnx qwen3.8_flash_experiment/wf_export_debug_boundary_v3/Qwen4ExpForCausalLM/Qwen4ExpForCausalLM-1be5535ee55e962b/Qwen4ExpForCausalLM.onnx \
  --device-list 0..15 \
  --work-dir qwen3.8_flash_experiment/wf_export_debug_boundary_v3/opstats_run_final \
  --iterations 30 \
  --profile-start-iter 20 \
  --profile-samples 2 \
  --batch-size 16 \
  --ctx-len 65536 \
  --pooled-ctx-len 16384'
```

Opstats are written to
`qwen3.8_flash_experiment/wf_export_debug_boundary_v3/opstats_run_final/opstats`.

To execute an existing QPC without recompiling, use `--qpc` and run under the
QAIC group. The example passes `device_ids=None` to `QAICInferenceSession`, so
the runtime selects devices instead of hardcoding a device list:

```bash
sg qaic -c '/home/ochougul/envs/qwen38f/bin/python \
  examples/text_generation/qwen4_exp_dp16_decode_qpc.py \
  --snapshot /home/huggingface_hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --qpc qwen3.8_flash_experiment/wf_export_debug_boundary_v3/qpc-ea581d119d396fd2/qpc'
```

The script supplies zero-valued example `input_ids`, `position_ids`, and host
`ngram_embeddings`, skips retained-state buffers so they remain device-owned,
and performs one decode step. Successful execution requires the selected QAIC
devices and their P2P fabric to be available to the user.

## Relevant files

- `examples/text_generation/qwen4_exp_create_4layer_checkpoint.py` — checkpoint creation entry point;
- `examples/text_generation/qwen4_exp_dp16_decode_qpc.py` — weight-free export, compile, and one-step QAIC execution;
- `examples/text_generation/qwen4_exp_dp16_decode_qpc_opstats.py` — profiling-QPC compilation, dummy IO generation, and opstats;
- `examples/text_generation/qwen4_exp_decode_parity.py` — shared Qwen3.8 window configuration and shard-pruning implementation;
- `QEfficient/transformers/models/qwen4_exp/modeling_qwen4_exp.py` — QSA wrapper and retained-state custom IO;
- `QEfficient/base/modeling_qeff.py` — QSA `qaic_config` and blocking-transform plumbing.
