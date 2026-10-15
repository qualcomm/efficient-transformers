# DFlash SPD Examples

Two entry points wrap the SPD (speculative-decoding) compile + run pipeline for
**text-only** language models and **vision-language (VLM, multimodal)** models. Each
resolves/compiles the QPCs (or reuses pre-built ones), then runs SPD inference on a
single prompt in-process via `QEfficient.generation.dflash_generation`. The three model adapters share the same internal `_run_spd_core` flow.

| | front-end (`--model_name`, builds QPCs) | runs via |
|---|---|---|
| text — single prompt | `basic_inference_text.py` | `dflash_generation.run_spd_inference_single` |
| vision — single prompt/image | `basic_inference_vision.py` | `run_spd_inference_gemma4` or `run_spd_inference_qwen3_vl` |

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

### Qwen3.5-4B

Qwen3.5 uses the VLM entry point with its DFlash target and draft checkpoints:

```bash
python basic_inference_vision.py --model_name Qwen3.5-4B \
    --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \
    --ctx_len 4096 --prefill_seq_len 128 \
    --image --image_prompt "Describe this image in detail."
```

Use `--prompt "..."` instead of `--image --image_prompt ...` for text-only input.
Prefill length can be any positive multiple of the draft block size; verification and replay remain block-sized.

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

DFlash targets use the existing SpD `num_logits_to_keep` input to project only
the last valid position during prompt prefill. Verification still predicts every
block position, and all hidden states needed by the DLM are preserved. This
applies automatically to Qwen3, Llama, Gemma4, Qwen3-VL, Qwen3.5, and Qwen3.5-MoE.
Existing TLM QPCs remain usable but need re-export/compilation for this change;
DLM and vision QPCs are unchanged.

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

## Adding a new model

Edit `MODEL_MAP` in `utils.py`:

```python
"<short-name>": ("<tlm-hf-repo or None>", "<dlm-hf-repo>"),
```

Both entry points reuse the same map automatically.
