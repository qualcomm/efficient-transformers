# MiniMax-M3 examples

These scripts demonstrate different MiniMax-M3 execution layouts:

- `minimax_m3_dp_cp_decode_only.py` exports and runs a text-only, one-token decode graph with configurable data/context parallelism for the dense and sparse attention paths.
- `minimax_m3_paged_dp_decode_only_configurable_pages.py` exercises the MiniMax paged-decode cache layouts with independently configurable GQA, MSA indexer, and MSA attention pages. It is intended for cache-shape and paged-attention testing, not as an accuracy benchmark.
- `minimax_m3_prefill_only.py` exports and runs a chunked prefill-only graph and reports time to first token (TTFT).

Run the full-checkpoint decode-only DP/CP example from the repository root. It uses static export dimensions and MXINT8 KV-cache I/O. The script uses all checkpoint layers by default; add `--num-layers 4` for a reduced four-layer run.

```bash
python examples/text_generation/minimax/minimax_m3_dp_cp_decode_only.py \
    --ctx-len 1048576 \
    --batch-size 1 \
    --num-devices 16 \
    --num-cores 4 \
    --generation-len 32 \
    --num-kv-blocks 8 \
    --indexer-num-blocks 4 \
    --msa-num-kv-blocks 4 \
    --attn-dp 1 \
    --attn-cp 4 \
    --msa-indexer-dp 1 \
    --msa-indexer-cp 16 \
    --msa-attn-dp 1 \
    --msa-attn-cp 4 \
    --device-ids 72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87 \
    --mxint8-kv-cache \
    --node-precision-info \
    --static-export-dims
```

Run the full-checkpoint prefill-only example from the repository root with this
two-device MDP configuration. It uses all checkpoint layers by default; add
`--num-layers 4` for a reduced four-layer run:

```bash
python examples/text_generation/minimax/minimax_m3_prefill_only.py \
  --model-id MiniMaxAI/MiniMax-M3 \
  --ctx-len 65536 \
  --batch-size 1 \
  --num-devices 2 \
  --device-group "[17,18]" \
  --mdp-num-partitions 2 \
  --prefill-seq-len 1024 \
  --num-cores 4 \
  --num-kv-blocks 32 \
  --q-head-block-chunk 1 \
  --q-block-size 256 \
  --q-block-chunk 1024 \
  --msa-num-kv-blocks 1 \
  --msa-q-chunk 4 \
  --indexer-num-blocks 8 \
  --indexer-q-chunk 1024 \
  --indexer-q-size 256 \
  --use-onnx-subfunctions \
  --mxint8-kv-cache \
  --msa-indexer-dp 1 \
  --msa-indexer-cp 1 \
  --msa-attn-dp 1 \
  --msa-attn-cp 1 \
  --indexer-n-head 1 \
  --time-passes \
  --static-export-dims
```

Run the configurable paged decode example from the repository root with this
full-checkpoint, 16-device configuration. It uses all checkpoint layers by
default; add `--num-layers 4` for a reduced four-layer run:

```bash
python examples/text_generation/minimax/minimax_m3_paged_dp_decode_only_configurable_pages.py \
  --ctx-len 1048576 \
  --batch-size 16 \
  --num-devices 16 \
  --num-cores 4 \
  --generation-len 32 \
  --num-kv-blocks 8 \
  --indexer-num-blocks 4 \
  --msa-num-kv-blocks 4 \
  --attn-dp 1 \
  --attn-cp 4 \
  --msa-indexer-dp 1 \
  --msa-indexer-cp 16 \
  --msa-attn-dp 1 \
  --msa-attn-cp 4 \
  --gqa-page-block-size 128 \
  --gqa-num-pages 8192 \
  --msa-indexer-page-block-size 128 \
  --msa-indexer-num-logical-pages 8192 \
  --msa-attn-page-block-size 128 \
  --msa-attn-num-logical-pages 8192 \
  --device-ids 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
```

### What the paged example exercises

The example compiles the MiniMax decode graph with paged KV enabled and passes
three independent block tables into generation:

| Cache | Contents | Page settings |
|-------|----------|---------------|
| GQA | Dense-attention key/value cache | `--gqa-page-block-size`, `--gqa-num-pages` |
| MSA indexer | Sparse-attention index keys used to select blocks | `--msa-indexer-page-block-size`, `--msa-indexer-num-logical-pages` |
| MSA attention | Sparse-attention key/value cache | `--msa-attn-page-block-size`, `--msa-attn-num-logical-pages` |

The GQA and MSA-attention K/V pools use the shape
`[physical_pages, DP * CP * num_key_value_heads, page_block_size, head_dim]`.
The MSA index-key pool uses
`[physical_pages, DP * CP * indexer_n_head, page_block_size, index_head_dim]`.
The first axis is physical pool capacity, separate from the logical-page
count configured by the page-count flags.

Each logical page is `page-block-size` tokens. In the command above, all three
caches use 128-token pages and 8192 logical pages per sequence, providing
capacity for the configured 1,048,576-token context. The separate logical-page
counts let the three caches be sized independently; optional
`--gqa-physical-pages`, `--msa-indexer-physical-pages`, and
`--msa-attn-physical-pages` flags configure physical-pool capacities.
Each logical capacity must cover `--ctx-len`, and page sizes must divide the
context length; the MSA indexer page size must also match the model's index
block size. The script validates GQA and indexer page-count divisibility
against their CP, KV-block, and core layouts.

The block tables have DP, local-batch, and CP-page-group axes. CP groups
logical pages by `logical_page // CP`, with the page's owner lane given by
`logical_page % CP`; the table maps each group to a physical page. Thus the
example's CP factors (GQA 4, MSA indexer 16, MSA attention 4) exercise distinct
table and cache layouts. The script creates reproducible, randomized
logical-to-physical mappings (seed 17 by default) and prints table shapes and
a short mapping preview before compilation.

This is a cache-layout and paged-attention dataflow test - accuracy is not guaranteed
for this script.  Use `--layout-only` to inspect generated tables without loading the 
model or compiling, or `--skip-generate` to compile without running generation. By
default, the runtime prompt is repeated to at least 129 tokens so decode starts on a 
later logical page and can exercise a nonzero CP owner lane with the 128-token page 
size in the command above.
