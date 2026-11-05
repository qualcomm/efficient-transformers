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

## Text models

**Supported models:** Llama-3.1-8B-Instruct, Qwen3-4B, Qwen3-8B, GPT-OSS-20B, GPT-OSS-120B (see `--help` for the full list).

### Single prompt

```bash
python basic_inference_text.py --model_name Qwen3-4B \
    --prompt "Explain speculative decoding in two sentences."
```

---

### GPT-OSS

GPT-OSS uses the existing expert-parallel implementation with chunked, full-context
KV caches. Prefill returns one prediction, while verification returns the draft
block's predictions (8 tokens for GPT-OSS-20B, 10 for GPT-OSS-120B).

```bash
python basic_inference_text.py --model_name gpt-oss-20b \
    --tlm_devices 0,1,2,3 --dlm_devices 4,5,6,7 \
    --ctx_len 4096 --prefill_seq_len 128 --generation_len 256 \
    --prompt "Explain why the sky appears blue in three sentences."
```

Standalone QEfficient target QPCs return token IDs; vLLM target QPCs return
vocabulary logits. Use QPCs compiled for the appropriate runtime.

### Separate MoE TLM QPCs

For GPT-OSS and Qwen3.5-MoE, `--disagg-tlm` is enabled by default in the
single-prompt examples. The TLM has separate prefill and verification QPCs,
each with one specialization. Prefill uses `--prefill_seq_len` and keeps one
prediction; verification uses the draft block size and keeps every prediction.
Both retain the existing expert-parallel computation. The DLM is unchanged.

Use `--no-disagg-tlm` to compile or run the previous combined TLM QPC.
Dense models keep the combined-QPC default. Split TLM currently supports BS=1.
After the last prefill chunk, KV state (plus convolution and recurrent state
for Qwen3.5-MoE) is transferred through host buffers to the verification QPC
on its first call. Later decode calls keep state on-device; Qwen3.5 dense/MoE
targets concatenate the accepted-prefix replay and new verification block in one call.

To reuse a split pair, pass `--tlm-prefill-qpc /path/to/prefill/qpc` and
`--tlm_qpc /path/to/verification/qpc`. Prefill uses `--tlm_devices` unless
`--tlm-prefill-devices` is supplied; both groups must have the same device count.
When they share devices, both TLM QPCs are resident, so their combined core
and memory requirements must fit. For example, two 8-core QPCs use all 16 cores
per TLM device. Existing combined QPCs remain usable with `--no-disagg-tlm`.

## Vision-language models (VLM, multimodal)

**Supported models:** `gemma-4-31B-it`, `Qwen3-VL-32B-Instruct` (more VLMs are added to
`MODEL_MAP` over time — run `--help` for the current list). The VLM family (gemma4 vs.
qwen3-vl) is auto-detected from the TLM's `config.model_type`, so `basic_inference_vision.py`
works for either — gemma4 entries have no default TLM HF path (pass `--tlm_hf_path`);
`Qwen3-VL-32B-Instruct` does, so `--tlm_hf_path` is optional for it.

The combined-TLM vision path compiles **three** QPCs: the language decoder (TLM), the **vision
encoder** (`pixel_values -> vision_embeds`), and the DFlash draft (DLM). Because the
TLM and DLM fill their own cards, the vision encoder usually needs its **own** devices.
Qwen3.5-MoE's default split TLM adds a fourth QPC for language prefill.

> Set `QEFF_HOME` to a filesystem with free space before compiling — the VLM TLM QPC
> is large (tens of GB):
> ```bash
> export QEFF_HOME=/local/mnt/workspace/<user>/qeff_home
> ```

### Independent language and vision resources

With `kv_offload=True`, the dual-QPC API accepts `vision_num_devices` and
`vision_num_cores` independently of the language `num_devices` and `num_cores`.
Omitting either vision setting inherits the corresponding language setting.
These controls apply to both model transforms and compilation, including layerwise
compilation; they do not apply to a single combined vision/language QPC.

For a shared-card TS4C8 language / TS4C4 draft / TS4C4 vision layout:

```python
qpc_paths = target.compile(
    num_devices=4,
    num_cores=8,
    vision_num_devices=4,
    vision_num_cores=4,
    prefill_seq_len=128,
    ctx_len=4096,
    dflash_block_size=8,
)
```

Compile the DFlash-2 draft separately with `num_devices=4, num_cores=4`.
TS means QAIC devices, and C means cores per device. At runtime, place all
three QPCs on the same four devices: each uses `8 + 4 + 4 = 16` cores.
Check device availability and physical-card grouping before activation.

`compile_dflash2_tlm_qpc` also accepts the independent vision settings. Its
default `lang_only=True` preserves the language-path string return; use
`lang_only=False` to compile both components and retain their named QPC paths.

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
Qwen3.5 dense and MoE targets use concatenated replay and verification. Each decode call takes
`2 * block_size` positions: the previous accepted prefix (padded with position IDs
of `-1`), followed by the next draft block. It commits the prefix's convolution
and recurrent states, as well as full-attention KV, while verifying the new block
without retaining that block's cache writes.
Only the new block's token predictions and projected features are used by DFlash.
The first decode call has an entirely masked replay half. After each verification,
the runtime queues the seed plus accepted draft tokens for the next call; the new
replacement/bonus seed belongs to the verification half, not the replay half.
The example reports this combined call as `TLM Replay+Verify`, with no separate
device replay time. SPD wall-time throughput includes DLM calls and host orchestration;
the rate based on time spent in model calls is shown separately, not as device-only profiling.
Prefill length must be a multiple of the draft block size and at least twice that size.
For a block size of 16, decode takes 32 positions and returns 16 token predictions.

**Recompilation required to enable fused replay:** Re-export and recompile the TLM;
older TLM ONNX/QPC artifacts cannot gain fused replay without rebuilding. The language
export hash distinguishes the new cache protocol; existing artifacts are not deleted.
Omit both `--tlm_qpc` and `--vision_qpc` to let this example select/compile compatible artifacts.
The existing vision and DLM QPCs do not need to change for fused replay.

The full-attention retained-KV fix uses the `fused_replay_kv_commit_v2` cache
protocol. Earlier concatenated TLM artifacts also need re-export/recompilation
to pick up this fix. Explicit `--tlm_qpc` paths must be replaced with rebuilt
artifacts; their input shapes alone do not identify the retained-state behavior.

The previous two-call TLM QPCs are no longer supported; the example rejects them
with a recompilation message. There is no two-call fallback or mode switch.
MXINT8/MXFP6 settings are unchanged. External runtimes, including vLLM, must support
the new fused input/state contract before using these new TLM QPCs.

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
Apart from the incompatible Qwen3.5 artifacts noted above, existing TLM QPCs remain
usable; re-export/recompile to benefit from this optimization. DLM and vision QPCs
are unchanged by the target logits optimization.

New DLM compilations consume a full TLM prefill chunk in one call for all supported
text and vision-language models. There is still one DLM specialization: target
features use `--prefill_seq_len`, while noise inputs and draft outputs stay
block-sized. Decode pads unused target positions with `-1`; noise positions are
unchanged. Override with `--dlm-target-seq-len` (a block-size multiple that divides
the TLM prefill length); use the draft block size to compare with the old path.

Only the DLM needs re-export/recompilation for this wider-target optimization.
Omit `--dlm_qpc` to build it; existing TLM and vision QPCs can be reused.
A supplied DLM QPC uses its compiled target width automatically, including legacy
block-width QPCs. An explicitly requested width must match the supplied QPC.

### Prompt Formatting and Thinking

Single prompts are always wrapped in the model's native chat template, with a
user message and an assistant-generation prefix. No task-specific instruction
or extra system message is added by default. In the text example,
`--format_prompt --category math` (or `coding`) adds the existing dataset-style
instruction before chat formatting.

Dataset benchmarks preserve their task instructions: text datasets use
`utils.py`, and MathVision requests a boxed answer or option letter. Do not pass
an already chat-formatted prompt, because it would be wrapped twice.

Thinking follows the model's chat-template default in all DFlash examples and
benchmarks. This retains thinking for the Qwen3/Qwen3.5 templates used in our
runs. Use `--enable-thinking` or `--disable-thinking` to override it where the
model supports that option; underscore aliases are also accepted. Templates
without a thinking mode are not given synthetic thinking tokens.

Baseline and DFlash comparisons must use the same task instructions, chat
template, thinking setting, image processing, and generation limits. These
prompt settings do not require recompiling QPCs.

| Flag | Default | Notes |
|---|---|---|
| `--tlm_devices` | `0,1,2,3` | TLM device IDs |
| `--dlm_devices` | `0,1,2,3` | DLM device IDs |
| `--tlm_cores` / `--dlm_cores` | `8` | per-side core count |
| `--ctx_len` | `4096` | both scripts |
| `--prefill_seq_len` | `128` | |
| `--dlm-target-seq-len` | TLM prefill width | New DLM compilation; reused QPCs use their compiled width |
| `--generation_len` | `256` (text script) / `1800` (vision script) | |
| `--enable-thinking` / `--disable-thinking` | model template | Mutually exclusive; only affects models with a thinking option |
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
Despite the entry point's name, the DFlash-2 path currently accepts text prompts
only; image prompts are rejected. Add `--compile-only` to compile the DFlash-2
pair without running inference.
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

DFlash-2 draft checkpoints must use `.safetensors`, either as a single file or
multiple shards. Pickle-based `pytorch_model*.bin` checkpoints are not supported,
and Hub downloads only fetch safetensors weights for the draft.

The native DFlash-2 loader requires `DFlash2DraftModel` in the checkpoint's
`architectures` list. A configuration containing DFlash-2 convolution and selector
metadata without that marker is rejected with an actionable error rather than
sent to the HF AutoModel loader. Configuration lookup errors are preserved; an
explicit supported `config` can be supplied when loading a local checkpoint.

The inference report distinguishes `SPD Wall Speed`, including host orchestration,
from `SPD Model-call Speed`, based only on target and draft model-call time. These
rates must not be treated as interchangeable when reporting benchmark results.

## Adding a new model

Edit `MODEL_MAP` in `utils.py`:

```python
"<short-name>": ("<tlm-hf-repo or None>", "<dlm-hf-repo>"),
```

Both entry points reuse the same map automatically.
