# GLM-5.3 Attention Example

`glm53_four_layer_decode_compile_generate.py` exercises the four-layer GLM-5.3 model with single-token decode dense MLA
and DSA attention configurations. It supports weight-free ONNX export, QAIC compilation, and decode generation.

`glm53_four_layer_prefill_compile.py` exports and compiles a separate multi-token prefill-only graph for dense
prefill-parallel or folded DSA modes. Use it for disaggregated-serving experiments where prefill and decode run as
separate QPCs.

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

The selected preset is copied into `qaic_config` before `QEFFAutoModelForCausalLM.transform()` is called. The
`baseline` preset passes an exact empty dictionary. Dense presets also set all four validation layers to
`full_attention`; DSA presets and the baseline retain the checkpoint's native sparse layers.

| Example preset | Benchmark variant selected | Important `qaic_config` values |
|---|---|---|
| `baseline` | native checkpoint attention | exactly `{}` |
| `dense` | dense MLA, no absorption | `blocking_mode=none`, absorption disabled |
| `dense_offline` | dense MLA, offline absorption | `blocking_mode=none`, absorption enabled, `online=false` |
| `dense_online` | dense MLA, online absorption | `blocking_mode=none`, absorption enabled, `online=true` |
| `dense_parallel` | `par`, two KV blocks, split 16 | `blocking_mode=par`, `num_kv_blocks=2`, `par_num_split=16` |
| `dsa_cp1` | DSA DP1/CP1 smoke topology | indexer DP1/CP1, attention DP1/CP1 |
| `dsa_cp2` | DSA CP2 folded-cache topology | indexer DP1/CP2, attention DP1/CP2 |
| `dsa_ts16_smoke` | reduced TS16 smoke topology | indexer DP1/CP16, attention DP16/CP1, 16 indexer blocks |

Run the complete preset matrix with:

```bash
python examples/glm/glm53_four_layer_decode_compile_generate.py \
  --weight-free --export-only --all-attention-configs \
  --prompt-len 4 --generation-len 1 \
  --qeff-home "$QEFF_HOME"
```

`dsa_ts16_smoke` defaults to batch 16, context 4096, and 16 devices. It is a reduced smoke graph, not the benchmark
context-262144 TS16 graph. Other presets default to batch 1, context 2176, and their required device count. Explicit
`--batch-size`, `--ctx-len`, and `--num-devices` values override those defaults.

For weight-free compilation, use `--cache-io-dtype mxint8` to store `compressed_kv`, `k_pe`, and `indexer_key` custom
I/O (including their retained-state bindings) as MXINT8. The default remains `float16`.

## Prefill-only export

Run dense prefill-parallel export and compile with the dedicated prefill script:

```bash
python examples/glm/glm53_four_layer_prefill_compile.py \
  --attention-preset dense_prefill_parallel \
  --prefill-seq-len 128 \
  --ctx-len 128 \
  --qeff-home "$QEFF_HOME"
```

Use `dense_prefill_parallel_online` for the online prefill variant. This script intentionally uses a regular loaded
model path. Weight-free disaggregated compile is currently blocked by the core export/compile contract.

Run the folded DSA prefill path with:

```bash
python examples/glm/glm53_four_layer_prefill_compile.py \
  --attention-preset dsa_prefill_cp1 \
  --prefill-seq-len 512 \
  --ctx-len 65536 \
  --qeff-home "$QEFF_HOME"
```

The DSA preset computes per-query Top-K indices in query blocks, gathers selected compressed-KV and RoPE rows from
the folded cache, and evaluates sparse MLA with KV blocking and online-softmax accumulation.

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
  "blocking_mode": "none | par",
  "num_kv_blocks": 2,
  "par_num_split": 16,
  "mla_absorption": {
    "absorption": true,
    "online": false,
    "cache_compressed": true
  }
}
```

The prefill script accepts `blocking_mode=prefill_par` and `blocking_mode=prefill_par_online` through its dense
presets. Its `dsa_prefill_cp1` preset additionally uses `indexer_ql_chunk`, `indexer_q_block_size`,
`indexer_topk_blocking`, `indexer_prefill_parallel`, `sparse_q_block_size`, and `sparse_kv_num_blocks`. All prefill
presets require `prefill_only=True` with a multi-token `seq_len`.

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

Benchmark-scale TS16 requires context 262144, which should be exported as its own artifact and validated with compile
placement evidence. Do not report the `dsa_ts16_smoke` artifact as benchmark TS16 parity.

```bash
python examples/glm/glm53_four_layer_decode_compile_generate.py \
  --weight-free --export-only \
  --attention-preset dsa_ts16_smoke \
  --ctx-len 262144 --prompt-len 4 --generation-len 1
```

Avoid combining topology-specific JSON overrides with `--all-attention-configs`: the same override is applied to every
preset and can make otherwise valid presets inconsistent with their runtime dimensions.
