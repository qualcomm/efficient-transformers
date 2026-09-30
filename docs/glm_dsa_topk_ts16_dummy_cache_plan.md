# Complete GLM DSA Top-K and TS16 Export Integration

## Summary

Replace full-context indexer reconstruction with benchmark-equivalent blocked local Top-K/global merge. Make validation presets choose valid dimensions, with TS16 defaulting to batch 16, 16 devices, and context 4096. Ensure export dummy caches use compile context length and each layer's actual layout.

## Implementation Changes

### Blocked DSA Top-K

- Add a production blocked-indexer helper under `QEfficient/blocking/glm_attention.py`.
- Keep the folded indexer cache in `[B/DP, DP*CP, T/CP, D]`; do not reconstruct `[B, T, D]`.
- Scatter the current indexer key, then reshape into CP/block/core tiles.
- For every block:
  - Compute per-head QK scores.
  - Apply scale, ReLU, learned head reduction, causal validity, and supplied attention mask in HF order.
  - Select `min(dsa_topk, block_width)` local candidates.
  - Convert local positions to global token indices.
- Concatenate candidates from every block and CP shard, then perform one final `dsa_topk` merge.
- Return global INT32 indices; shared-indexer layers continue reusing the preceding full-indexer result.
- Precompute and hash context-dependent topology values such as tokens per core and block Top-K.
- Preserve the existing non-folded CP1 path and tuple-cache behavior.

### TS16 Validation Dimensions

- Add preset runtime defaults separate from QAIC attention settings.
- Resolve CLI precedence as: explicit CLI value, then preset runtime default.
- Use:
  - `dsa_ts16_smoke`: batch 16, context 4096, 16 devices.
  - Other presets: current lightweight batch/context defaults.
- Default generation to a bounded validation length, rather than filling the entire context, while allowing `--generation-len` overrides.
- Make `--all-attention-configs` resolve dimensions independently for each child preset.
- Reject explicit invalid values instead of silently rounding them.
- Pass resolved batch size and device count into weight-free specializations and `_compile`; remove the hardcoded batch size of 1.
- Document that larger production runs use explicit contexts such as 262144 or 1048576.

### Dummy Cache Length and Layout

- Add a model-provided GLM export-dimension hook so generic export plumbing does not contain GLM layout rules.
- Persist the compile-resolved batch size and full context length during `BlockingAttentionTransform`.
- Keep trace `seq_len` and cache `context_length` separate:
  - Inputs and position IDs use trace `seq_len`.
  - Compressed and indexer retained states use full `context_length`.
- Construct cache tensors from `pkv_cache[layer_idx]`, not `pkv_cache[0]`, so mixed dense/DSA layers retain distinct shapes.
- Use the resolved TS16 batch for export examples, ensuring folded DP16 caches are valid.
- Keep folded cache dimensions on distinct symbolic axes; do not equate `B/DP` or `T/CP` with full batch/context symbols.
- Ensure standalone weight-free validation explicitly applies the compile-time transform before export.

## Test Plan

- Blocked Top-K parity:
  - Compare blocked and monolithic indexer results for CP1, CP2, and a reduced TS16-compatible fixture.
  - Cover multi-step cache updates, future-token masking, padded attention masks, full/shared indexers, and INT32 indices.
  - Verify sparse-attention outputs match when driven by either Top-K implementation.
- ONNX topology:
  - Export the folded indexer and assert one Top-K per block plus one final merge.
  - Assert the graph does not materialize a full-context indexer tensor.
  - Retain the passing Dynamo integer division/modulo export test.
- Configuration:
  - Verify TS16 accepts context 4096, batch 16, and 16 devices.
  - Verify context 2176, batch 1, incorrect device counts, and divisibility violations fail clearly.
- Dummy caches:
  - Use `seq_len=1` and `context_length=4096`; assert cache capacity is 4096.
  - Verify every mixed dense/DSA layer receives its own cache shape.
  - Verify folded attention and indexer shapes, retained-state names, and symbolic axes.
- Validation:
  - Run focused GLM transform, cache, quickcheck, and weight-free suites.
  - Run the nine-mode script matrix with the TS16 smoke defaults.
  - When the checkpoint is available, run export, compile, and generation through `sg qaic -c`.
  - Run manual `ruff format` and `ruff check`; do not run `pre-commit`.

## Assumptions

- TS16's default context is 4096 for practical correctness validation; benchmark-scale contexts remain explicit overrides.
- Benchmark code remains an untracked reference and is neither imported nor modified.
- No `pyproject.toml` changes or new public attention-mode flags.
- Export hashes include all resolved topology values that can change the generated graph.
