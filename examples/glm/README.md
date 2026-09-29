# GLM-5.3 Attention Example

`glm53_four_layer_decode_compile_generate.py` exercises the four-layer GLM-5.3 model with the dense MLA and DSA
attention configurations represented by the standalone GLM attention benchmark. It supports weight-free ONNX export,
QAIC compilation, and decode generation.

The example always transforms the QEff model with a single-token decode graph. Presets named `prefill_parallel` select
the corresponding benchmark attention implementation, but they do not change this example into a multi-token prefill
workload.

## Setup

Run from the repository root. Set the Hugging Face cache before loading the model and enable accelerated Hub transfer
when a download is required:

```bash
export HF_HUB_CACHE=/home/huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME=/tmp/qeff_glm53
```

For an offline run against an already populated cache, also set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Export one four-layer configuration without compiling it:

```bash
python examples/glm/glm53_four_layer_decode_compile_generate.py \
  --weight-free --export-only \
  --attention-preset dsa_cp1 \
  --prompt-len 4 --generation-len 1 \
  --qeff-home "$QEFF_HOME"
```

Remove `--export-only` to compile. By default the script then generates tokens and compares them with a CPU Hugging
Face decode; use `--skip-generate` to stop after compilation.

## Attention presets

The selected preset is copied into `qaic_config` before `QEFFAutoModelForCausalLM.transform()` is called. Dense presets
also set all four validation layers to `full_attention`; DSA presets retain the checkpoint's native sparse layers.

| Example preset | Benchmark variant selected | Important `qaic_config` values |
|---|---|---|
| `dense` | dense MLA, no absorption | `blocking_mode=none`, absorption disabled |
| `dense_offline` | dense MLA, offline absorption | `blocking_mode=none`, absorption enabled, `online=false` |
| `dense_online` | dense MLA, online absorption | `blocking_mode=none`, absorption enabled, `online=true` |
| `dense_parallel` | `par`, two KV blocks, split 16 | `blocking_mode=par`, `num_kv_blocks=2`, `par_num_split=16` |
| `dense_prefill_parallel` | `prefill_par_blocks2` | `blocking_mode=prefill_par`, offline absorption |
| `dense_prefill_parallel_online` | `prefill_par_online_blocks2` | `blocking_mode=prefill_par_online`, online absorption |
| `dsa_cp1` | DSA DP1/CP1 smoke topology | indexer DP1/CP1, attention DP1/CP1 |
| `dsa_cp2` | DSA CP2 folded-cache topology | indexer DP1/CP2, attention DP1/CP2 |
| `dsa_ts16` | benchmark TS16 topology | indexer DP1/CP16, attention DP16/CP1, 16 indexer blocks |

Run the complete preset matrix with:

```bash
python examples/glm/glm53_four_layer_decode_compile_generate.py \
  --weight-free --export-only --all-attention-configs \
  --prompt-len 4 --generation-len 1 \
  --qeff-home "$QEFF_HOME"
```

`dsa_ts16` defaults to batch 16, context 4096, and 16 devices. Other presets default to batch 1, context 2176, and
their required device count. Explicit `--batch-size`, `--ctx-len`, and `--num-devices` values override those defaults.

## Passing custom `qaic_config`

Use `--attention-qaic-json` to shallow-merge a JSON object over the selected preset. For example, this selects the
benchmark `par` implementation but changes its KV block and split counts:

```bash
python examples/glm/glm53_four_layer_decode_compile_generate.py \
  --weight-free --export-only \
  --attention-preset dense_parallel \
  --attention-qaic-json '{"num_kv_blocks": 4, "par_num_split": 8}'
```

The merge is shallow. When overriding `mla_absorption`, pass the complete nested object:

```bash
--attention-qaic-json \
  '{"mla_absorption":{"absorption":true,"online":true,"cache_compressed":true}}'
```

The dense benchmark modes accept these fields:

```json
{
  "blocking_mode": "none | par | prefill_par | prefill_par_online",
  "num_kv_blocks": 2,
  "par_num_split": 16,
  "mla_absorption": {
    "absorption": true,
    "online": false,
    "cache_compressed": true
  }
}
```

DSA topology is controlled independently for the indexer and sparse attention:

Top-K is not a QAIC setting. It comes from the model's `index_topk` configuration field (2048 for GLM-5.3) and is
bounded by the selected context length.

```json
{
  "indexer_dp": 1,
  "indexer_cp": 16,
  "indexer_kvp": 1,
  "attn_dp": 16,
  "attn_cp": 1,
  "attn_kvp": 1,
  "indexer_num_blocks": 16,
  "num_cores_per_device": 16
}
```

For a custom four-device folded topology:

```bash
python examples/glm/glm53_four_layer_decode_compile_generate.py \
  --weight-free --export-only \
  --attention-preset dsa_cp1 \
  --batch-size 4 --ctx-len 2176 --num-devices 4 \
  --attention-qaic-json \
  '{"indexer_dp":1,"indexer_cp":4,"indexer_kvp":1,"attn_dp":4,"attn_cp":1,"attn_kvp":1,"indexer_num_blocks":1,"num_cores_per_device":16}'
```

The script validates that DP × CP equals `--num-devices`, batch is divisible by DP, context is divisible by CP, and
the indexer local context is divisible by `indexer_num_blocks * num_cores_per_device`.

To reproduce the benchmark-scale TS16 layout, keep the `dsa_ts16` topology and override only the context:

```bash
python examples/glm/glm53_four_layer_decode_compile_generate.py \
  --weight-free --export-only \
  --attention-preset dsa_ts16 \
  --ctx-len 262144 --prompt-len 4 --generation-len 1
```

Avoid combining topology-specific JSON overrides with `--all-attention-configs`: the same override is applied to every
preset and can make otherwise valid presets inconsistent with their runtime dimensions.
