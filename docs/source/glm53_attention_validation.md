# GLM-5.3 Attention Integration Validation

This report tracks production integration of dense MLA and DeepSeek sparse attention (DSA) for the four-layer
GLM-5.3 validation model. Generated ONNX files, QPCs, logs, graph images, and benchmark outputs are intentionally
kept outside the repository.

## Environment

- Model: `zai-org/GLM-5.3`
- Cached revision: `aca966e4e02791568aa6a4ced368624b3d897f42`
- Python environment: `/home/ochougul/envs/xf`
- Hugging Face cache: `/home/huggingface_hub`
- PyTorch: `2.13.0+cpu`
- Transformers: `5.15.1`
- ONNX: `1.18.0`
- QAIC Graph API: `11.5.invalid.DBG`
- QAIC runtime: `LRT.AIC.13.2.1.23.0.33`
- Hardware discovery: 46 AI100 devices reported `Ready` through `sg qaic -c '/opt/qti-aic/tools/qaic-util -q'`.

## Commands

CPU regression and parity coverage:

```bash
HF_HUB_CACHE=/home/huggingface_hub HF_HUB_ENABLE_HF_TRANSFER=1 \
  /home/ochougul/envs/xf/bin/python -m pytest -q \
  tests/unit_test/transforms/test_blocking_transform.py \
  tests/unit_test/models/test_cache_correctness.py \
  tests/weight_free/test_transforms.py \
  tests/unit_test/models/test_model_quickcheck.py \
  -k 'glm_moe_dsa or glm_attention or glm_dsa_folded or reduced_glm or blocking_transform'
```

Real four-layer export probe:

```bash
HF_HUB_CACHE=/home/huggingface_hub HF_HUB_ENABLE_HF_TRANSFER=1 \
QEFF_HOME=/home/ochougul/efficient-transformers/artifacts/glm53_attention \
  /home/ochougul/envs/xf/bin/python examples/glm/glm53_four_layer_decode_compile_generate.py \
  --weight-free --export-only --attention-preset dsa_cp1 \
  --prompt-len 4 --ctx-len 2048 --generation-len 1 \
  --qeff-home /home/ochougul/efficient-transformers/artifacts/glm53_attention \
  --export-dir /home/ochougul/efficient-transformers/artifacts/glm53_attention/dsa_cp1_export
```

The graph was captured and translated to ONNX successfully. Checkpoint preparation then attempted to fetch the
incomplete 408 GB local snapshot (93 of 146 files). The run was stopped to avoid exhausting the 656 GB remaining
filesystem space, so no ONNX/QPC artifact was retained and no hardware execution was claimed.

## QAIC Configuration Matrix

| Preset | Layer topology | Configuration summary | CPU parity | Export | Compile/generate |
|---|---|---|---|---|---|
| `dense` | validation-only full attention | no absorption | Pass | Not run | Not run |
| `dense_offline` | validation-only full attention | offline absorption | Pass | Not run | Not run |
| `dense_online` | validation-only full attention | online absorption | Pass | Not run | Not run |
| `dense_parallel` | validation-only full attention | parallel decode, 2 KV blocks, TS16 split | Covered by dispatch/cache tests | Not run | Not run |
| `dsa_cp1` | native sparse attention | DP1/CP1 indexer and attention | Pass | Graph translation pass; artifact incomplete | Not run |
| `dsa_cp2` | native sparse attention | CP2 folded caches | Folded cache pass | Not run | Not run |
| `dsa_ts16_smoke` | native sparse attention | reduced indexer DP1/CP16, attention DP16/CP1 | Configuration/shape coverage | Not run | Not run |

The decode script resolves runtime dimensions after selecting each preset. `dsa_ts16_smoke` defaults to batch 16,
context 4096, and 16 devices; the other presets keep the lightweight batch 1/context 2176 defaults. Generation defaults to 32
tokens instead of filling the entire context. Explicit `--batch-size`, `--ctx-len`, `--num-devices`, and
`--generation-len` values take precedence and are rejected when they violate the selected topology. Production-scale
runs should pass their intended context explicitly, for example `--ctx-len 262144` or `--ctx-len 1048576`.

Dense comparisons use native HF DSA as the reference and keep the live context within `index_topk`, making the
sparse selection equivalent to full attention. The validation-only QEff model changes only
`layer_types=["full_attention"] * 4`; checkpoint weights are unchanged.

## Cache and Top-K Parity

- Legacy compressed cache tuples remain `((compressed_kv, k_pe), ...)`.
- Dense MLA retains `[B, Hkv, T, D]` cache layout.
- Folded DSA cache scatter/gather round trips pass for `[B/DP, DP*CP, T/CP, D]`.
- Folded indexer caches remain in `[B/indexer_DP, indexer_DP*indexer_CP, T/indexer_CP, D_index]`; blocked local
  Top-K candidates are merged without reconstructing `[B, T, D_index]`.
- The four-layer quickcheck passes HF-to-QEff logits for native DSA and verifies three full-indexer cache entries
  followed by a shared-indexer layer reusing the preceding Top-K indices.
- Dense no-absorption, offline-absorption, and online-absorption PyTorch paths match native HF DSA when Top-K covers
  the complete live context.

## Graph Inspection

| Layer | Native role | Expected production graph | Inspection status |
|---|---|---|---|
| 0 | full indexer | indexer Q/K, ReLU-before-head-reduction, Top-K, sparse MLA | Structural/unit coverage |
| 1 | full indexer | independent retained indexer cache and Top-K | Structural/unit coverage |
| 2 | full indexer | independent retained indexer cache and Top-K | Structural/unit coverage |
| 3 | shared indexer | consumes layer 2 Top-K; no indexer projection/cache | Quickcheck pass |

Visual graph-parity verdicts for final QAIC artifacts remain pending. The production implementation defines the
required folded-row gather, paged scatter, sparse scatter, integer division/modulo, sparse gather/reduction, output
projection, and distinct folded dynamic-axis symbols, but operator attributes and device partition placement must be
confirmed from a completed real export/compile.

## Known Limitations

- The complete GLM-5.3 snapshot was not locally available. Completing it risked filling the shared filesystem.
- Consequently, the full nine-mode export/compile/generation matrix and on-device token comparison are not yet
  measured in this checkout.
- No claim of QAIC or ONNX Runtime parity is made from export translation alone.
- Use `--all-attention-configs` for the complete matrix once the checkpoint is available. `--attention-qaic-json`
  can override any preset value, and `--results-json` records artifact and token-comparison metadata.
