# K2 Horizon on Cloud AI 100 Ultra

How to compile and run `IFM/K2-Horizon-7B` on a server with Cloud AI 100 Ultra cards, check that the card output matches Hugging Face, and collect the numbers for the benchmark.

## Status

| Stage | Where | Result |
|---|---|---|
| HF PyTorch vs QEff PyTorch, full 36 layers (bf16) | CPU | Tokens match over 24 generated tokens |
| HF vs QEff PyTorch vs ONNX Runtime, first 8 layers (fp32) | CPU | Tokens match |
| ONNX export, full model | Server | To do |
| Compile and run on AI 100 Ultra | Server | To do |

The wrapper covers the dense sizes (0.9B, 3.7B, 7B, 32B). The 36B and 375B MoVA sizes have routed value experts and a sparse MoE that are not mapped yet.

## Prerequisites

- A server with Cloud AI 100 Ultra cards, Platform SDK and Apps SDK installed (`/opt/qti-aic` exists). Check the cards with `/opt/qti-aic/tools/qaic-util -q`.
- Python 3.10 to 3.12. The repo pins `transformers==5.5.4`.
- Internet access to Hugging Face. The model is public, no token needed.
- Disk: about 18 GB for the bf16 checkpoint, about 36 GB for the fp32 ONNX, plus the QPC.

One AI 100 Ultra card is 4 devices of 32 GB. The 7B needs about 18 GB in fp16 and about 7 GB with MXFP6 weights, so one device is enough. Use more devices only for throughput.

## Install

```bash
git clone -b k2-horizon-onboarding https://github.com/Nouman945/efficient-transformers.git
cd efficient-transformers
python3 -m venv qeff_env && source qeff_env/bin/activate
pip install -U pip && pip install -e ".[test]"

export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME=/path/with/space/qeff_cache   # ONNX and QPC files go here
```

The model is remote code, so every load passes `trust_remote_code`. Without it transformers stops at an interactive prompt.

## Step 1: compile and run

Start without MXFP6 so the first run is a precision check, not a performance run.

```bash
python examples/text_generation/k2_horizon/k2_horizon_inference.py \
    --model-name IFM/K2-Horizon-7B \
    --prompt "The capital of France is" \
    --prefill-seq-len 128 --ctx-len 4096 \
    --num-cores 16 --device-group [0]
```

The same thing through the CLI:

```bash
python -m QEfficient.cloud.infer --model_name IFM/K2-Horizon-7B --trust_remote_code \
    --batch_size 1 --prompt_len 128 --ctx_len 4096 --num_cores 16 --device_group [0] \
    --prompt "The capital of France is" --mos 1 --aic_enable_depth_first
```

Expected output for this prompt, greedy: ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The`

The first run exports the ONNX (a few minutes) and compiles the QPC (longer). Both are cached under `QEFF_HOME`, so later runs with the same settings skip straight to generation. Add `--use-onnx-subfunctions` to cut export and compile time.

## Step 2: token match against Hugging Face

The repo test does this for a reduced random model, on the card:

```bash
pytest -n auto tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon and dummy"
```

For the real weights, compare the card output with a greedy Hugging Face run of the same prompt:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "IFM/K2-Horizon-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.bfloat16)
inputs = tokenizer("The capital of France is", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:]))
```

Compare the first 20 or so tokens. fp16 on the card and bf16 on CPU can drift on a long tail, which is normal. A different first token, or repeated junk, is not.

## Step 3: performance runs

Switch on the production precision and read the metrics that `generate()` prints:

```bash
python examples/text_generation/k2_horizon/k2_horizon_inference.py \
    --prefill-seq-len 128 --ctx-len 4096 --generation-len 256 \
    --num-cores 16 --device-group [0] --mxfp6 --mxint8-kv-cache
```

`perf_metrics` gives prefill time (time to first token), decode tokens per second, and total throughput. Repeat for the context lengths and batch sizes the customer cares about. Batching is set at compile time (`batch_size` or continuous batching with `full_batch_size` in `compile()`), see `examples/text_generation/continuous_batching.py`.

Precision on AI 100 Ultra: fp16 compute, MXFP6 weights (`--mxfp6`), MXINT8 KV cache (`--mxint8-kv-cache`). bf16 and FP8 are not AI 100 features, so use the bf16 Hub repo and let the compiler quantize; skip the `-FP8` repos.

## Step 4: Performance per Watt and per Dollar

- Watts: read card power from `/opt/qti-aic/tools/qaic-util -q` while a decode run is in progress, or use the SDK's device telemetry for a time series. Record idle power as well.
- Performance per Watt = decode tokens per second divided by the measured card power during decode.
- Performance per Dollar = decode tokens per second divided by the card price (or hourly cost) used for the comparison. The price is an input, agree it with the customer.

Report the compile settings next to every number: prompt length, context length, batch size, cores, devices, MXFP6, MXINT8.

## Troubleshooting

- `Do you wish to run the custom code? [y/N]`: `trust_remote_code` was not passed.
- The ONNX is about 0.5 GB larger than the weights alone: the rotary tables are baked in for the full 524288 context. Harmless.
- Out of memory during export: the export runs in fp32 and needs about 36 GB of host RAM for the 7B.
- Compile takes long: add `--use-onnx-subfunctions`, and keep `ctx_len` to what the benchmark needs.
- `[ERROR] cache_position is part of K2HorizonModel.forward's signature, but not documented`: printed by transformers when it loads the remote code. Harmless.

## What the wrapper changes

- `QEfficient/transformers/models/k2_horizon/modeling_k2_horizon.py`: KV cache attention (with the optional gate and QK norm used by larger sizes), grouped RMSNorm on the compiler custom op, static rotary tables.
- `QEfficient/transformers/models/pytorch_transforms.py`: the K2 classes are mapped by name through `KVCacheExternalModuleMapperTransform`.
- `QEfficient/transformers/modeling_utils.py`: `K2HorizonConfig` added to the external class mapping.
- `tests/configs/causal_model_configs.json`, `QEfficient/utils/test_utils.py`: dummy-layer test config.
