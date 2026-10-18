# Onboarding a remote-code CausalLM: K2 Horizon, step by step

[README.md](README.md) explains the general process with a made-up `Blueprint` model. This file is the same process on a real one: the dense K2 Horizon family (`IFM/K2-Horizon-0.9B`, `-3.7B`, `-7B`, `-32B`), which ships its model class as remote code (`trust_remote_code=True`) instead of living in `transformers`. Every step names the file it touched, the command that checks it, and what "pass" looks like. Use it as the template for the next remote-code model.

Remote code changes three things compared with an in-tree model:

1. The HF classes cannot be imported, so the wrapper is registered by class name through `KVCacheExternalModuleMapperTransform`, not by class through `KVCacheModuleMapperTransform`.
2. The config class has to be routed to the right auto class in `EXTERNAL_MODEL_CLASS_MAPPING`.
3. Every test loader needs to know to pass `trust_remote_code`, which is what `ModelConfig.EXTERNAL_MODELS` is for.

Everything else is the same four-stage pipeline: HF PyTorch -> QEff PyTorch -> ONNX Runtime -> Cloud AI 100, tokens equal at every stage.

---

## Step 0: learn the model before writing code

Read the Hub config of every size you intend to support, not just one:

```python
from transformers import AutoConfig
for size in ["0.9B", "3.7B", "7B", "32B"]:
    c = AutoConfig.from_pretrained(f"IFM/K2-Horizon-{size}", trust_remote_code=True)
    print(size, c.num_hidden_layers, c.hidden_size, c.num_attention_heads, c.num_key_value_heads,
          c.head_dim, c.vocab_size, c.layernorm_num_groups, c.rope_parameters, c.max_position_embeddings)
```

What this found for K2 Horizon, and why each item mattered later:

| Fact | Consequence |
|---|---|
| `auto_map` in config.json points at `modeling_k2_horizon.K2HorizonForCausalLM` | remote code: string-keyed registration (Step 2) |
| the remote file is generated from transformers' `qwen3_moe` modular file | Qwen3 skeleton (GQA, RMSNorm, SwiGLU, RoPE); start from the Llama/Qwen2 wrappers |
| `layernorm_num_groups` is 1, 2, 4, 4 across the sizes | grouped RMSNorm needs its own custom-op path, and a test per group count |
| `head_dim` is independent of `hidden_size / num_attention_heads` (32B: 5120 vs 64 x 128) | never derive one from the other in the wrapper or in dummy configs |
| `max_position_embeddings` 524288 (131072 on the 0.9B), YaRN on the 0.9B | baking a 512K rotary table into the ONNX is too large; cap it |
| `num_experts`, `query_key_norm`, `attention_gate_func` exist in the config but are off in all four | support the dense path only and refuse experts explicitly |
| `vocab_size` 250624 (64256 on the 0.9B), `eos_token_id` `[1, 250019]` | dummy configs must keep the real vocabulary, or the lm_head path differs |

Decide the scope now and write it in the wrapper docstring. K2 Horizon: the four dense sizes; the MoVA 36B is a different attention and out of scope.

---

## Step 1: the wrapper

File: `QEfficient/transformers/models/k2_horizon/modeling_k2_horizon.py` (+ `__init__.py`).

One `QEff` class per HF class whose `forward` must change. The HF math stays; only input plumbing, cache handling and export-unfriendly constructs change.

| Class | What it does |
|---|---|
| `QEffK2HorizonRMSNorm` | `forward` calls the compiler RMSNorm custom op. With `n_groups > 1` it reshapes to groups, normalises against a unit weight, and multiplies by the full weight after. `__qeff_init__` registers that unit weight once as a non-persistent buffer so `forward` allocates nothing. |
| `QEffK2HorizonAttention` | `forward` takes `position_ids`, `past_key_value` (`QEffDynamicCache`) and `batch_index`, writes K/V through `past_key_value.update`, runs the eager attention loop. `__qeff_init__` keeps the optional gate and QK norm wired. |
| `QEffK2HorizonDecoderLayer` | threads the new arguments through attention and MLP. |
| `QEffK2HorizonModel` | `__qeff_init__` bakes `cos_cached` / `sin_cached` for `min(max_position_embeddings, K2_HORIZON_MAX_POSITION_EMBEDDINGS)` positions and raises `NotImplementedError` for a config with experts, partial rotary or a sliding window. `forward` builds the causal mask from `position_ids` and returns the cache in the tuple layout the exporter expects. |
| `QEffK2HorizonForCausalLM` | `forward` gathers the last logits. `get_submodules_for_export` names the decoder layer for ONNX subfunctions. `get_specializations` returns `[prefill_spec, decode_spec]` and rejects `prefill_seq_len` or `ctx_len` beyond `self.model.cos_cached.shape[0]`, the real table size. |

Rules that came out of review:

- Refuse what is out of scope at transform time with a clear error. A silent fallback hides a wrong model.
- Validate against the tensor that was built, not against the constant that sized it.
- Build once in `__qeff_init__` what `forward` would otherwise rebuild on every call.
- Do not leave code for a path you refuse (the router-logits unpack for experts was removed).

---

## Step 2: registration

| File | Change |
|---|---|
| `QEfficient/transformers/models/pytorch_transforms.py` | entries in `KVCacheExternalModuleMapperTransform._match_string_replace_method`, keyed by HF class name: `"K2HorizonRMSNorm": {"forward": QEffK2HorizonRMSNorm.forward, "__qeff_init__": QEffK2HorizonRMSNorm.__qeff_init__}` and the same for the attention, decoder layer, model and causal LM. |
| `QEfficient/transformers/modeling_utils.py` | `EXTERNAL_MODEL_CLASS_MAPPING["K2HorizonConfig"] = "QEFFAutoModelForCausalLM"`. |
| `QEfficient/utils/constants.py` | `K2_HORIZON_MAX_POSITION_EMBEDDINGS = 65536`. |
| `QEfficient/utils/test_utils.py` | `ModelConfig.EXTERNAL_MODELS` gets one entry per size. Without it the test loaders call `from_pretrained` without `trust_remote_code` and stop at an interactive prompt. |

Check, offline:

```bash
pytest tests/unit_test/utils/test_modeling_registry.py tests/unit_test/transforms/test_transform_accuracy.py -k k2 -v
```

---

## Step 3: unit tests, offline, seconds

File: `tests/unit_test/models/test_new_arch_accuracy.py`, class `TestK2HorizonTransformStructure`. It asserts:

- every mapper entry exists and is callable
- a config with experts raises at transform time
- `get_specializations` returns two specs and rejects lengths beyond the baked table (a `SimpleNamespace` with `cos_cached=torch.zeros(128, 32)` is enough)
- the grouped RMSNorm equals the reference formula for `n_groups` 1 and 4 (`atol=1e-5`)

```bash
pytest tests/unit_test/models/test_new_arch_accuracy.py -k k2 -v
```

---

## Step 4: stages 1 to 3 on a reduced config, CPU, minutes

Each real Hub config reduced to 2 layers / 4 heads / small hidden with random weights, through HF PyTorch -> QEff PyTorch -> ONNX Runtime, tokens compared. One run per size, because the sizes differ in features (norm groups, head_dim, rope type, vocabulary):

```python
config = AutoConfig.from_pretrained(name, trust_remote_code=True)
config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads = 2, 4, 2
config.hidden_size, config.intermediate_size = 4 * config.head_dim, 512
config.mlp_only_layers, config.max_position_embeddings = [0, 1], 256
model_hf = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager").eval()
runner = ApiRunner(1, tokenizer, config, ["My name is"], 8, 24)
hf_tokens = runner.run_hf_model_on_pytorch_CB(copy.deepcopy(model_hf))[0]
qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=name)
kv_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)
ort_tokens = runner.run_kv_model_on_ort(str(qeff_model.export()))
```

Pass: `hf_tokens == kv_tokens == ort_tokens` for every size. A KV mismatch is the wrapper; an ORT mismatch is the export (an op that traces differently, a shape-derived slice).

---

## Step 5: stages 1 to 3 on real weights, CPU

Same three stages with the checkpoint, first N layers only so it fits in RAM (`num_hidden_layers=N` on `from_pretrained` loads only those layers). The fp32 ONNX leg needs about 4 bytes per parameter of host RAM: ~36 GB for the full 7B, ~140 GB for the full 32B. Pass: identical tokens at all three stages. After this the wrapper is correct; everything that follows is the card.

---

## Step 6: stage 4, first run on Cloud AI 100, fp16

```bash
python examples/text_generation/basic_inference.py --model-name IFM/K2-Horizon-7B --trust-remote-code \
    --prompt "The capital of France is" --prefill-seq-len 128 --ctx-len 4096 --num-cores 16 --device-group '[0]'
```

Sizes above about 15B parameters need all four SoCs of an Ultra card (`--device-group '[0,1,2,3]'`); the 32B is 70 GB in fp16. Add `--use-onnx-subfunctions` to shorten export and compile.

Pass: the text equals a greedy Hugging Face run on CPU of the same prompt (`do_sample=False`, bf16) for the first ~20 tokens; later drift between fp16 on the card and bf16 on the host is normal. Seen and harmless: the `aten::index_put_` export warnings, transformers' `[ERROR] cache_position ... not documented` on remote code.

---

## Step 7: the repo test lanes on the card

Register the model in `tests/configs/causal_model_configs.json`:

- `causal_lm_models`: one entry per size with `additional_params` that reduce the config to 2 layers but keep the features that differ between sizes (the 32B entry uses 8 query heads over 1 KV head with `hidden_size != heads * head_dim`, as the real config does). This feeds the `dummy` (random weights), `few` (real checkpoint, first layer) and `full` (real checkpoint, all layers, 4 devices) lanes.
- `per_pr_causal_text_models`: one `k2_horizon_text` entry with `config_overrides` for the continuous-batching, ONNX-subfunction, CCL, fp16/fp32/bf16 and speculative lanes.

Run from the repo root; the card tests run one at a time, so no `-n auto`:

```bash
for size in 0.9B 3.7B 7B 32B; do
    pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-$size and dummy" -v
    pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-$size and few" -v
done
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "K2-Horizon-7B and full" -v
pytest tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k "k2_horizon_text" -v
```

Expected: dummy 2 passed 1 skipped (the repeat-KV test skips models not in its list); few 4 passed; full 2 passed; per-PR 4 passed, 1 xfail (speculative decoding does not apply to a class-keyed external model), and on an AI 100-only host 1 failed (the bf16 case compiles for AI 200, see `test_causal_lm_models.py` where `aic_hw_version` follows the dtype). The `full` lane holds the fp32 checkpoint twice on the host; skip sizes that do not fit.

Goldens: the HF reference tokens of every lane live in `tests/golden_outputs/goldens.json`. A missing entry is computed on the CPU on first use and written; commit the file. Set `QEFF_REGENERATE_GOLDEN=1` only after a change that alters the model output. `QEFF_RUN_CPU_REFERENCES=1` runs the QEff PyTorch and ONNX Runtime legs live instead of trusting the golden.

`tests/transformers/models/causal_lm_models/check_causal_models.py` was changed so external models run the HF reference leg unless their `EXTERNAL_MODELS` entry sets `skip_hf_reference` (grok-1 keeps the skip).

---

## Step 8: production precision

Rerun Step 6 with `--mxfp6 --mxint8-kv-cache`. Read the text, not only the numbers: quantization changes them, so the output is close to HF, not identical. If it degrades, simulate MXFP6 on the CPU per layer or weight group to find the sensitive part and keep that in fp16. For K2 Horizon: the 3.7B, 7B and 32B take MXFP6; the 0.9B degrades (its first four layers are the sensitive ones) and is validated in fp16 weights with MXINT8 KV.

---

## Step 9: docs

| File | What |
|---|---|
| `docs/source/validate.md` | one row per size validated on hardware, with a footnote naming the precision. Only sizes that ran on a card. |
| `examples/text_generation/README.md` | a short section: the `--trust-remote-code` flag, the sizes covered, the rotary cap, quoting the device list in csh/zsh. |

---

## Step 10: the PR

1. `ruff check --fix . && ruff format .` (or `pre-commit run --all-files`); `pyproject.toml` stays untouched.
2. Signed-off commits: `git commit -s`.
3. Rebase on `main`, rerun Steps 2 to 4 offline.
4. The PR contains the wrapper, registration, tests, goldens and docs only; server guides and benchmark tooling stay out of it.
5. The PR description lists every test command with its result, the hardware and SDK used, the known gaps, and discloses any AI assistance.

---

## Checklist per size

A size is onboarded when all four hold:

1. HF PyTorch == QEff PyTorch == ONNX Runtime on the reduced config and on real weights (Steps 4, 5).
2. The fp16 card output equals the Hugging Face greedy text (Step 6).
3. The `dummy`, `few` and per-PR lanes pass on the card with goldens committed (Step 7).
4. The production-precision output still reads right (Step 8), and the size has its row in `validate.md` (Step 9).

K2 Horizon files in this PR: `QEfficient/transformers/models/k2_horizon/`, `pytorch_transforms.py`, `modeling_utils.py`, `constants.py`, `test_utils.py`, `tests/configs/causal_model_configs.json`, `tests/golden_outputs/goldens.json`, `check_causal_models.py`, the three unit test files, `docs/source/validate.md`, `examples/text_generation/README.md`.
