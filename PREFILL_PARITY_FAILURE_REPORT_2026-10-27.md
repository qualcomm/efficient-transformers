# Offline Prefill Parity Failure Report — October 27, 2026

## Scope

Models: GPT-OSS, Gemma4, Qwen3-VL-MoE, and Qwen3.5-MoE.

Modes: `prefill_q`, `prefill_kv`, and `prefill_qkv`.

Both paths compile with `use_onnx_subfunctions=True`:

- **Weight-free:** `weight_free=True`, which uses Dynamo.
- **Legacy:** `weight_free=False`, `dynamo=False`.

## Feature Branch Results

Branch: `weight_free_prefill`.

### Weight-free

| Model | `prefill_q` | `prefill_kv` | `prefill_qkv` |
| --- | --- | --- | --- |
| GPT-OSS | Pass: full 40-token parity | Pass: full 40-token parity | Pass: full 40-token parity |
| Gemma4 | Pass: full 10-token parity | Pass: full 10-token parity | Pass: full 10-token parity |
| Qwen3-VL-MoE | Pass: full 10-token parity | Pass: full 10-token parity | Pass: full 10-token parity |
| Qwen3.5-MoE | Fail: first three tokens match, then decode diverges | Fail: first three tokens match, then decode diverges | Fail: first three tokens match, then decode diverges |

Qwen3.5-MoE now reaches export, compile, and QAIC inference. Its earlier HF-reference error caused by forcing all hybrid layers to full attention was corrected; the remaining failures are genuine parity failures.

### Gemma4 Resolution

Gemma4 `prefill_kv` previously diverged from the first generated token when the test used two KV blocks. The failure was not caused by random weights, the Gemma4 vision-bidirectional attention mask, or the prefill-to-decode retained-state handoff:

- The same test passes when only one KV block is used.
- A focused PyTorch comparison showed the offline-KV kernel agrees with regular attention for Gemma4-shaped heads and its bidirectional vision mask.
- The two-block QAIC failure therefore isolated the problem to the offline-KV cross-block reduction emitted into the export graph.

`blocked_kv_attention_forward_prefill_headpar_offline` now merges KV blocks and head-parallel splits with the equivalent stable running-max recurrence instead of stacked `max`/`sum` reductions. This preserves the numerical attention formula while avoiding the compiler-sensitive multi-block reduction pattern.

The passing test still uses `num_kv_blocks=2` (`ctx_len=512`, two 256-token cache blocks) and `headpar_split=1`. The one-block result was a runtime-only diagnostic that proved the cross-block merge was responsible; it did not modify the test configuration or introduce an unblocked fallback. Each block still computes its local maximum, softmax denominator, and weighted-value output, then the stable recurrence rescales both block results to their shared maximum before combining them.

Validated with `weight_free=True`, Dynamo, `use_onnx_subfunctions=True`, and the existing ten-token full e2e parity assertion:

- Gemma4 `prefill_kv`: **pass** — `/tmp/qeff_gemma4_prefill_kv_weight_free_fixed_20261027.log`.
- Gemma4 `prefill_qkv`: **pass** — `/tmp/qeff_gemma4_prefill_qkv_weight_free_20261027.log`.
- Gemma4 `prefill_q`: previously passed under the same matrix.
- Focused blocked-attention unit tests: **7 passed**.

The one-KV-block diagnostic pass is recorded at `/tmp/qeff_gemma4_prefill_kv_one_block_20261027.log`; it was a runtime-only test override and did not change branch configuration.

### Qwen3-VL-MoE Resolution

Qwen3-VL-MoE `prefill_q` originally produced an incorrect first QAIC token. Reducing the test from two query blocks to one did not change the failure, ruling out the Q-block split itself. The actual issue was chunked prefill: the runtime executes the prompt in 64-token chunks, but `blocked_q_attention_forward_prefill` attended only to the current chunk's K/V tensors. Later chunks must attend to the full prefix already written to retained KV cache.

`prefill_q` now reads the just-written layer cache before attention and regenerates a full-context causal mask when the per-chunk input mask is shorter than the retained K/V context. This preserves direct Q blocking while restoring causal-prefix visibility for chunked prefill. A focused unit regression covers cached-K/V use for later prompt chunks.

- Qwen3-VL-MoE `prefill_q`: **pass**, full ten-token QAIC/HF parity — `/tmp/qeff_qwen3_vl_moe_prefill_q_weight_free_fixed_20261027.log`.
- Focused blocked-attention unit tests after this change: **8 passed**.

### Qwen3.5-MoE Remaining Failure

The same chunked-prefill correction required a narrow `read_only(layer_idx, cache_kwargs)` adapter on Qwen3.5-MoE's hybrid cache. It fixes its initial prefill failure: all three offline modes now match the first three generated tokens. The remaining mismatch is identical for `prefill_q`, `prefill_kv`, and `prefill_qkv`, beginning at generated token index 3:

```text
HF   = [60941, 38638, 82454, 236485, 178243, 77303, 200690, 238326, 135927, 240627]
QAIC = [60941, 38638, 82454, 157303, 1049, 204466, 195274, 228191, 201916, 172976]
```

Because all three prefill implementations now reach the exact same first three tokens and fail on the same later decode token, the remaining problem is not mode-specific blocked prefill attention. It is a shared Qwen3.5 hybrid decode-state evolution/handoff issue involving its linear-attention convolution and recurrent retained states. No parity shortcut, reduced-token assertion, or mode fallback was applied.

Post-fix full e2e logs:

- `prefill_q`: `/tmp/qeff_qwen3_5_moe_prefill_q_weight_free_fixed_20261027.log`.
- `prefill_kv`: `/tmp/qeff_qwen3_5_moe_prefill_kv_weight_free_20261027.log`.
- `prefill_qkv`: `/tmp/qeff_qwen3_5_moe_prefill_qkv_weight_free_20261027.log`.

### Legacy

| Model | `prefill_q` | `prefill_kv` | `prefill_qkv` |
| --- | --- | --- | --- |
| GPT-OSS | **Compile failure** | Pass: full 40-token parity | Pass: full 40-token parity |
| Gemma4 | Pass: full 10-token parity | Fail: QAIC/HF token mismatch | **Compile failure** |
| Qwen3-VL-MoE | Fail: QAIC/HF token mismatch | Pass: full 10-token parity | **Compile failure** |
| Qwen3.5-MoE | Fail: QAIC returns ten `0` tokens | Fail: QAIC returns ten `0` tokens | Fail: QAIC returns ten `0` tokens |

Exact legacy failure signatures:

- GPT-OSS `prefill_q`: `qaic-compile` exits with `CalledProcessError` and exit status `255`.
- Gemma4 `prefill_kv`: `HF=[138838, 121390, 66852, 241217, 110802, 23102, 84589, 166796, 42216, 166796]`; `QAIC=[27114, 166796, 61134, 33050, 12628, 42846, 123327, 156545, 166796, 166796]`.
- Gemma4 `prefill_qkv`: `qaic-compile` exits with status `1`.
- Qwen3-VL-MoE `prefill_q`: `HF=[45519, 17920, 17920, 17920, 17920, 17920, 17920, 17920, 17920, 17920]`; `QAIC=[0, 56830, 1933, 72078, 17920, 17920, 17920, 17920, 17920, 17920]`.
- Qwen3-VL-MoE `prefill_qkv`: `qaic-compile` exits with status `1`.
- Qwen3.5-MoE all modes: expected `HF=[60941, 38638, 82454, 236485, 178243, 77303, 200690, 238326, 135927, 240627]`; received `QAIC=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.

Raw legacy VLM log: `/tmp/qeff_vlm_legacy_prefill_parity_20261026.log`.

## Mainline Baseline Results

Worktree: `/home/sabhis/blocking/quic_abhishek_legacy_mainline`.

Branch: `legacy_prefill_mainline_validation`.

Base: `origin/main` commit `3736940` (`Add blocking support with weightfree and dynamo for CausalLM models (#1403)`).

The baseline matrix ran 18 VLM cases: three models × three modes × both paths.

| Model | `prefill_q` | `prefill_kv` | `prefill_qkv` |
| --- | --- | --- | --- |
| Gemma4 | Weight-free export failure; legacy parity failure | Weight-free and legacy parity failure | Weight-free and legacy compile/runtime failure |
| Qwen3-VL-MoE | Weight-free export failure; legacy parity failure | **Passes in both paths** | Weight-free and legacy compile failure |
| Qwen3.5-MoE | Fails before QAIC reference generation in both paths | Fails before QAIC reference generation in both paths | Fails before QAIC reference generation in both paths |

Mainline result: **2 passed, 16 failed**.

Exact mainline failure signatures:

- Nested subfunction export: `Could not uniquely resolve the nested past_key.0 state writer ... expected one CtxScatter*, found 0`.
- Gemma4 legacy `prefill_q`: QAIC/HF token parity mismatch.
- Gemma4 legacy `prefill_qkv`: decode binding mismatch: `input_ids shape (1, 1) does not match input_ids binding shape (1, 32)`.
- Qwen3-VL-MoE legacy `prefill_q`: QAIC/HF token parity mismatch.
- Qwen3-VL-MoE `prefill_qkv`: QAIC compiler failures, including exit status `1` and `SIGABRT`.
- Qwen3.5-MoE: `ValueError: has_previous_state can only be called on LinearAttention layers, and the current Cache seem to only contain Attention layers`.

Raw mainline log: `/tmp/qeff_mainline_offline_prefill_parity_20261026.log`.

## Commands

Feature legacy VLM matrix:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_CACHE=/tmp/qeff_prefill_hf_hub \
QEFF_HOME=/tmp/qeff_prefill_mode_exports PYTHONPATH="$PWD" \
/home/sabhis/dynamo_and_weightfree/current_changes/efficient-transformers/dynamo/bin/pytest -q --tb=short \
tests/weight_free/test_on_qaic_vlm.py::test_vlm_offline_prefill_hf_qaic_parity -k legacy
```

Mainline baseline matrix:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_CACHE=/tmp/qeff_prefill_hf_hub \
QEFF_HOME=/tmp/qeff_prefill_mainline_exports PYTHONPATH="$PWD" \
/home/sabhis/dynamo_and_weightfree/current_changes/efficient-transformers/dynamo/bin/pytest -q --tb=short \
tests/weight_free/test_on_qaic_vlm.py::test_vlm_offline_prefill_hf_qaic_parity
```
