# DeepSeek-V4-Flash

[`deepseek_v4_flash_decode.py`](deepseek_v4_flash_decode.py) is the folder's
export, compile, and runtime entry point. By default it exports and compiles a
one-token decode graph, then runs generation. Add `--prefill-only` to export
and compile a prefill graph; prefill runs stop after compilation.

The companion scripts are also kept here:

- [`deepseek_v4_flash_parity.py`](deepseek_v4_flash_parity.py) compares HF and
  QAIC logits for chunked prefill and decode using precompiled QPCs.
- [`deepseek_v4_prefill_mad.py`](deepseek_v4_prefill_mad.py) compares saved HF
  and QAIC prefill logits and reports mean absolute difference and top-1 match.

Run the commands below from the repository root. Set `MODEL_DIR` to a local
DeepSeek-V4-Flash checkpoint and set `HF_CACHE` and `ARTIFACT_ROOT` to writable
directories:

```bash
MODEL_DIR=/path/to/DeepSeek-V4-Flash
HF_CACHE=/path/to/huggingface-cache
ARTIFACT_ROOT=/path/to/qeff-artifacts
```

## One-token decode

This full 43-layer AI100 configuration uses batch 16, context length 262144,
and 16 devices. The script defaults to all 43 DeepSeek-V4-Flash layers; use
`--num-hidden-layers 4` for a reduced four-layer run:

```bash
python examples/text_generation/deepseekV4/deepseek_v4_flash_decode.py \
    --model-id "$MODEL_DIR" \
    --hf-cache "$HF_CACHE" \
    --artifact-root "$ARTIFACT_ROOT" \
    --ctx-len 262144 \
    --num-hidden-layers 43 \
    --num-cores 16 \
    --batch-size 16 \
    --device-group "$(seq -s, 0 15)" \
    --attn-dp 16 \
    --indexer-cp 16 \
    --hca-compressed-kv-cp 2 \
    --hca-attn-blocks 8 \
    --hw-version ai100 \
    --ffn-blocking-mode token \
    --ffn-token-block-size 4
```

## Prefill with pipeline-parallel MDP

This 43-layer AI200 configuration uses 16 pipeline partitions across devices
16–31, a context length of 1048576, and a prefill sequence length of 1024. The
CSA prefill gather subchunk size and indexer tile size are configurable from
the runner's CLI:

```bash
python examples/text_generation/deepseekV4/deepseek_v4_flash_decode.py \
    --model-id "$MODEL_DIR" \
    --hf-cache "$HF_CACHE" \
    --artifact-root "$ARTIFACT_ROOT" \
    --prefill-only \
    --ctx-len 1048576 \
    --prefill-seq-len 1024 \
    --num-hidden-layers 43 \
    --num-cores 4 \
    --batch-size 1 \
    --device-group "$(seq -s, 16 31)" \
    --attn-dp 1 \
    --indexer-cp 1 \
    --hca-attn-blocks 1 \
    --hw-version ai200 \
    --mdp-num-partitions 16 \
    --mdp-strategy onnx \
    --csa-prefill-attn-gather-sub-qc 4 \
    --csa-prefill-indexer-kv-block-size 512 \
    --use-onnx-subfunctions
```

The runner does not currently expose an HCA prefill query-block-size option;
the former wrapper's `hca_prefill_q_block_size` default was not applied to the
model. Use `--help` to see all supported options. `--export-only` and
`--compile-only` can be used to stop after the respective stage.
