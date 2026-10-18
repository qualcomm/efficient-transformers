# K2 Horizon: end-to-end test plan on a Cloud AI 100 server

Every check that has to pass before a dense K2 Horizon size (`IFM/K2-Horizon-0.9B`, `-3.7B`, `-7B`, `-32B`) counts as onboarded, in the order to run them, with the expected result of each. Everything here uses files in this branch only. Run from the repo root unless a step says otherwise. Commands are bash; in csh use `setenv NAME value`, `source qeff_env/bin/activate.csh`, and keep the quotes on the device lists (`'[0]'`), which are harmless in bash.

Tick the table at the end as you go. Keep the log file path QEfficient prints at the end of every run.

## 0. Environment, once

```bash
cd efficient-transformers
python3 -m venv qeff_env && source qeff_env/bin/activate
pip install -e ".[test]"

# caches on a local disk with room, not a quota-limited home
export HF_HOME=/local/mnt/workspace/$USER/hf
export QEFF_HOME=/local/mnt/workspace/$USER/qeff_cache
export QEFF_LOG_PATH=/local/mnt/workspace/$USER/qeff_logs
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $HF_HOME $QEFF_HOME $QEFF_LOG_PATH

/opt/qti-aic/tools/qaic-util -q | grep -E "QID|Status"     # 4 devices, all Ready
df -h /local/mnt/workspace                                 # 0.9B 8 GB, 3.7B 35 GB, 7B 60 GB, 32B 250 GB
free -g                                                    # fp32 export: 7B 36 GB, 32B 140 GB; full lane: double that
```

Download the weights once, in the background, so no later step waits on the network:

```bash
python3 -c "from huggingface_hub import snapshot_download; [snapshot_download(f'IFM/K2-Horizon-{s}') for s in ['0.9B','3.7B','7B','32B']]"
```

Nothing else may use the card while a step runs. Use tmux for anything over a few minutes.

## 1. Offline checks, minutes, no card

```bash
pytest tests/unit_test/utils/test_modeling_registry.py tests/unit_test/transforms/test_transform_accuracy.py \
       tests/unit_test/models/test_new_arch_accuracy.py -k k2 -v
```
Expected: 12 passed.

```bash
cd examples/text_generation/k2_horizon && python3 tools/k2_tiny_check.py && cd ../../..
```
Expected: four `RESULT ... HF==KV True HF==ORT True` lines, exit 0. This is each size's real config reduced to 2 layers with random weights through HF PyTorch, QEff PyTorch and ONNX Runtime.

## 2. Repo lanes on the card, per size

Three lanes per size plus the per-PR lane. Never pass `-n auto`; the card tests run one at a time. The Hugging Face reference tokens come from `tests/golden_outputs/goldens.json`; a missing entry is computed on the CPU and written on first use.

### 2a. dummy: random weights, 2 layers, real config features
```bash
for size in 0.9B 3.7B 7B 32B; do
  pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-$size and dummy" -v
done
```
Expected per size: 2 passed, 1 skipped (the repeat-KV test skips models not in its list). About a minute each. The generated text is gibberish; only the token match counts.

### 2b. few: real checkpoint, first layer
```bash
for size in 0.9B 3.7B 7B 32B; do
  QEFF_RUN_CPU_REFERENCES=1 pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-$size and few" -v
done
```
Expected per size: 4 passed (plain, continuous batching, 2-partition MDP compile with and without ONNX subfunctions). `QEFF_RUN_CPU_REFERENCES=1` runs the QEff PyTorch and ONNX Runtime legs live as well, so the plain test compares all four stages, not only HF vs card (the continuous-batching test compares HF vs card only). Minutes each; the 32B loads one layer plus the 250k-row embeddings, about 10 GB.

### 2c. full: real checkpoint, all layers, fp32, 4 devices
```bash
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-0.9B and full" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-3.7B and full" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-7B and full" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-32B and full" -v   # needs about 300 GB of host RAM
```
Expected per size: 2 passed (plain and continuous batching). The fp32 checkpoint is held twice on the host (HF reference plus the export copy): 7B about 80 GB, 32B about 300 GB. Skip the 32B if `free -g` says no; then 2b plus section 4 stand in for it, and say so in the results.

### 2d. per-PR feature lanes, 7B dummy shape
```bash
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "k2_horizon_text" -v
```
Expected: 4 passed, 1 xfail, 1 failed on an AI 100-only host. The xfail is speculative decoding (does not apply to a class-keyed external model). The failure is the bf16 case: it compiles for AI 200 because the test picks the target from the dtype; the compiler error is `Kernel lookup failed for: libjit_element_select_splat_kernel_f`. On a host with the AI 200 toolchain it passes.

### 2e. goldens
```bash
git status --short tests/golden_outputs/goldens.json
```
If the file changed (new entries for the 3.7B, 32B and the real-weight lanes), commit it; it is part of the PR:
```bash
git add tests/golden_outputs/goldens.json && git commit -s -m "K2 Horizon: goldens for the real-weight and 3.7B/32B lanes"
```
Set `QEFF_REGENERATE_GOLDEN=1` only after a change that alters the model output; otherwise an existing golden must not change.

## 3. Compiled QPC against Hugging Face at the shipped shape, per size

`tools/qpc_vs_hf.py` compiles (or reuses from `QEFF_HOME`) a QPC at the given shape, runs the prompt greedy on the card and on the CPU in bf16, and reports the first token where they differ. First the fp16 run, which is the precision check; then the production precision.

```bash
cd examples/text_generation/k2_horizon

# fp16, one device (32B: four devices)
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-0.9B --prefill-seq-len 128 --ctx-len 4096 --device-group '[0]'
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-3.7B --prefill-seq-len 128 --ctx-len 4096 --device-group '[0]'
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-7B   --prefill-seq-len 128 --ctx-len 4096 --device-group '[0]'
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-32B  --prefill-seq-len 128 --ctx-len 4096 --device-group '[0,1,2,3]' --use-onnx-subfunctions

# production precision, four devices: MXFP6 weights + MXINT8 KV (the 0.9B keeps fp16 weights)
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-0.9B --prefill-seq-len 128 --ctx-len 4096 --device-group '[0,1,2,3]' --mxint8-kv-cache --use-onnx-subfunctions --min-match 8
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-3.7B --prefill-seq-len 128 --ctx-len 4096 --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --min-match 8
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-7B   --prefill-seq-len 128 --ctx-len 4096 --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --min-match 8
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-32B  --prefill-seq-len 128 --ctx-len 4096 --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --min-match 8
cd ../../..
```

Expected: `RESULT {... "pass": true}` and `first differing token` of 16 or more in fp16 (usually 20+), 8 or more at MXFP6. The 7B fp16 answer is ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome.`; the 32B's is ` Paris, a city known for its rich history, culture, and iconic landmarks.` A different first token, or text that stops reading like language, is a failure; at MXFP6 that means the size needs sensitive layers kept in fp16 (the 0.9B does, which is why it runs without `--mxfp6`).

Each command also prints `PerfMetrics(...)` (TTFT, decode tokens/s for one user). Record them; they go into the status table. The first run of each shape exports the ONNX (minutes, 140 GB of RAM for the 32B) and compiles (7B 20 to 40 minutes, 32B 1 to 2 hours); reruns reuse the cache. The CPU leg loads the model in bf16: 2 to 70 GB of RAM.

## 4. Long context, per size: needle at 32K and at the 64K cap

The rotary tables are baked for 65536 positions, so 64K is the most a QPC can take, and nothing before this step has exercised rope or the cache past a few thousand positions. The 0.9B's YaRN scaling is identity below 8192, so this is the only test of it.

```bash
cd examples/text_generation/k2_horizon
for size in 3.7B 7B 32B; do
  python3 tools/qpc_vs_hf.py IFM/K2-Horizon-$size --needle-tokens 30000 --prefill-seq-len 1024 --ctx-len 32768 \
      --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --skip-hf
  python3 tools/qpc_vs_hf.py IFM/K2-Horizon-$size --needle-tokens 60000 --prefill-seq-len 1024 --ctx-len 65536 \
      --device-group '[0,1,2,3]' --mxfp6 --mxint8-kv-cache --use-onnx-subfunctions --skip-hf
done
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-0.9B --needle-tokens 30000 --prefill-seq-len 1024 --ctx-len 32768 \
    --device-group '[0,1,2,3]' --mxint8-kv-cache --use-onnx-subfunctions --skip-hf
python3 tools/qpc_vs_hf.py IFM/K2-Horizon-0.9B --needle-tokens 60000 --prefill-seq-len 1024 --ctx-len 65536 \
    --device-group '[0,1,2,3]' --mxint8-kv-cache --use-onnx-subfunctions --skip-hf
cd ../../..
```

Expected: `"pass": true`, the QPC text starts with ` marigold`. Each is a new compile (the 64K ones are long) plus a 30K or 60K-token prefill. `--skip-hf` leaves out the CPU reference, which would take hours at this length; a pass needs no reference because the answer is known. A fail at 64K with a pass at 32K points at the rotary tables or the cache indexing, not the weights.

Optional, once, for the record: the 7B at 32K with the CPU reference (`--generation-len 8`, no `--skip-hf`), to show the first tokens agree with HF at long range too.

## 5. What to collect

- `tests/golden_outputs/goldens.json` committed if it changed (2e).
- The pytest summary lines of 2a to 2d per size.
- `examples/text_generation/k2_horizon/tools/results/qpc_vs_hf_*.json` from sections 3 and 4 (one file per run; `pass`, both texts, `first_diff`).
- The `PerfMetrics` lines from section 3 and the `Model compiled to:` paths.
- `/opt/qti-aic/tools/qaic-version-util --apps` output for the SDK version.

Then fill `docs/source/validate.md` (one row per size that passed everything, footnote with the precision) and the table below.

## 6. Sign-off

| Check | 0.9B | 3.7B | 7B | 32B |
|---|---|---|---|---|
| 1. unit tests + tiny check | | | | |
| 2a. dummy lane | | | | |
| 2b. few lane (4 stages live) | | | | |
| 2c. full lane | | | | n/a if RAM < 300 GB |
| 2d. per-PR lane | n/a | n/a | | n/a |
| 3. QPC vs HF, fp16, 128/4096 | | | | |
| 3. QPC vs HF, production precision | fp16 + MXINT8 | MXFP6 + MXINT8 | MXFP6 + MXINT8 | MXFP6 + MXINT8 |
| 4. needle 32K | | | | |
| 4. needle 64K | | | | |
| decode tok/s, 1 user, 4 devices | | | | |
| row in `validate.md` | | | | |

A size is onboarded when every cell in its column is filled with a pass (or a justified n/a).
