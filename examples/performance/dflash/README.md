# DFlash SPD Examples

Two entry points wrap the DFlash and DFlash-2 SPD (speculative-decoding) compile + run pipeline for
**text-only** language models and **vision-language (VLM, multimodal)** models. Each
resolves/compiles the QPCs (or reuses pre-built ones), then runs SPD inference on a
single prompt in-process via `QEfficient.generation.dflash_generation`. The three model adapters share the same internal `_run_spd_core` flow.

| | front-end (`--model_name`, builds QPCs) | runs via |
|---|---|---|
| text — single prompt | `basic_inference_text.py` | `dflash_generation.run_spd_inference_single` |
| vision — single prompt/image | `basic_inference_vision.py` | text, Gemma4, or Qwen3-VL generation adapter |

---

## Text (dense language models)

**Supported models:** Llama-3.1-8B-Instruct, Qwen3-4B, Qwen3-8B (see `--help` for the full list).

### Single prompt

```bash
python basic_inference_text.py --model_name Qwen3-4B \
    --prompt "Explain speculative decoding in two sentences."
```

---

## Vision-language models (VLM, multimodal)

**Supported models:** `gemma-4-31B-it`, `Qwen3-VL-32B-Instruct` (more VLMs are added to
`MODEL_MAP` over time — run `--help` for the current list). The VLM family (gemma4 vs.
qwen3-vl) is auto-detected from the TLM's `config.model_type`, so `basic_inference_vision.py`
works for either — gemma4 entries have no default TLM HF path (pass `--tlm_hf_path`);
`Qwen3-VL-32B-Instruct` does, so `--tlm_hf_path` is optional for it.

The vision path compiles **three** QPCs: the language decoder (TLM), the **vision
encoder** (`pixel_values -> vision_embeds`), and the DFlash draft (DLM). Because the
TLM and DLM fill their own cards, the vision encoder usually needs its **own** devices.

> Set `QEFF_HOME` to a filesystem with free space before compiling — the VLM TLM QPC
> is large (tens of GB):
> ```bash
> export QEFF_HOME=/local/mnt/workspace/<user>/qeff_home
> ```

### Single prompt (text through the VLM)

```bash
# gemma4
python basic_inference_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --prompt "Tell me about the Taj Mahal."

# qwen3-vl
python basic_inference_vision.py --model_name Qwen3-VL-32B-Instruct \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --prompt "Tell me about the Taj Mahal."
```

### Single image + text prompt

```bash
# gemma4
python basic_inference_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --image \
    --image_prompt "Describe this image in detail."

# qwen3-vl
python basic_inference_vision.py --model_name Qwen3-VL-32B-Instruct \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --image \
    --image_prompt "Describe this image in detail."
```

`--height`/`--width` (qwen3-vl only) set the vision encoder's compiled input resolution;
they default to `354`/`536` when omitted.

---

## `--model_name`

Accepts either the short key or the full HF repo path (case-insensitive):

```
Qwen3-4B          Qwen/Qwen3-4B          qwen3-4b
gemma-4-31B-it
Qwen3-VL-32B-Instruct
```

Run either script with `--help` to see the full supported list.

## Skipping compile (reuse QPCs)

Whichever QPC side you supply skips its compile step; the rest still compiles.

```bash
# text
python basic_inference_text.py --model_name Qwen3-4B \
    --tlm_qpc /path/to/tlm/qpc --dlm_qpc /path/to/dlm/qpc --prompt "Hello"

# vision (--tlm_qpc + --vision_qpc must be given together to skip the VLM build)
python basic_inference_vision.py --model_name gemma-4-31B-it \
    --tlm_qpc /path/to/lang/qpc --vision_qpc /path/to/vision/qpc --dlm_qpc /path/to/dlm/qpc \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --prompt "Hello"
```

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--tlm_devices` | `0,1,2,3` | TLM device IDs |
| `--dlm_devices` | `0,1,2,3` | DLM device IDs |
| `--tlm_cores` / `--dlm_cores` | `8` | per-side core count |
| `--ctx_len` | `4096` (text) / `2048` (vision) | |
| `--prefill_seq_len` | `128` | |
| `--generation_len` | `256` | |
| `--iteration` | `300` | max SPD iterations |
| `--hf_token` | `$HF_TOKEN` | required for gated repos |
| `--tlm_hf_path` | from `MODEL_MAP` | required when the map entry has `None` (VLM entries) |

`basic_inference_text.py` only:

| Flag | Default |
|---|---|
| `--prompt` | *(required)* |
| `--category` | `""` (math / coding / reasoning / …) |
| `--format_prompt` | off — wraps `--prompt` with the category template when set |

`basic_inference_vision.py` adds:

| Flag | Default | Notes |
|---|---|---|
| `--vision_devices` | `0,1,2,3` | vision-encoder device IDs (usually a separate group) |
| `--image` | off | run an image+text prompt instead of text |
| `--image_url` | — | image URL for `--image` |
| `--image_prompt` | — | prompt text for `--image` |
| `--height` / `--width` | `354` / `536` | qwen3-vl only — compiled vision-encoder input resolution |
| `--dflash-method` | `dflash` | select `dflash` or `dflash-2` |

## DFlash-2

`basic_inference_vision.py --dflash-method dflash-2` compiles missing TLM and DLM QPCs automatically.
The host-side top-k candidate selector is disabled by default, so draft tokens use
per-position logits argmax (`top_k_selector=False`).

```bash
python examples/performance/dflash/basic_inference_vision.py \
    --dflash-method dflash-2 \
    --model_name Qwen3.8-27B \
    --tlm_devices 34,35,36,37 \
    --dlm_devices 38,39,40,41 \
    --tlm_cores 16 \
    --no-top-k-selector \
    --prompt "Write a Python function that returns the nth Fibonacci number."
```

Pass `--tlm_qpc` and `--dlm_qpc` to reuse compiled QPCs, or
`--target-model` and `--draft-model` to override the model pair. To enable the
DFlash-2 bigram selector, pass `--top-k-selector`; that mode requires a DLM QPC
compiled with the `selector_hidden` output. DLM retained KV-cache I/O defaults
to MXINT8 for throughput; pass `--no-dlm-mxint8-kv-cache` to compile it as FP16.

## Adding a new model

Edit `MODEL_MAP` in `utils.py`:

```python
"<short-name>": ("<tlm-hf-repo or None>", "<dlm-hf-repo>"),
```

Both entry points reuse the same map automatically.
