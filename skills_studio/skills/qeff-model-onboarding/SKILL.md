---
name: qeff-model-onboarding
description: Add support for a new Hugging Face model in this QEfficient repo. Use when the user provides a Hub model id, pasted model card, config excerpt, or architecture name and wants the QEff equivalent implemented, mapped to the closest existing wrapper family, validated with a tiny or config-derived test model, and covered through `tests/test_model_quickcheck.py`.
---

# QEff Model Onboarding

## Overview
Use this skill to convert an upstream Hugging Face model into the nearest valid QEff implementation in this repo with minimal divergence. Given either a Hub model id or a pasted model card, classify the architecture, reuse the closest local modeling pattern, wire export/runtime/cache behavior, and prove support through the repo’s consolidated quickcheck.

## Inputs
- Accept any of these as starting material:
  - a Hugging Face model id
  - a pasted model card
  - a config snippet
  - an architecture class name
- Extract at least:
  - `model_type`
  - `architectures`
  - task category
  - attention style
  - cache style
  - RoPE or positional encoding details
  - MoE, MLA, hybrid, chunking, or sliding-window features
  - multimodal or audio components

## Workflow
1. Classify the model before editing anything.
Read `references/model-family-map.md` and map the incoming model onto the closest local QEff family.

2. Prefer the nearest existing wrapper over inventing a new one.
Reuse the smallest viable pattern from the closest family. Keep divergence at wrapper and cache/export boundaries whenever possible.

3. Touch the repo in the expected order.
Update the model wrapper first, then the module replacement map, then the auto/export/runtime glue, then tests.

4. Add cache changes only when the model actually needs new retained-state behavior.
If the new model works with existing cache types, do not create a new cache abstraction.

5. Select a validation model that exercises the feature you changed.
Prefer an existing tiny Hub checkpoint. If none exists, derive a dummy config that still activates the same architecture path.

6. Add or extend coverage in `tests/test_model_quickcheck.py`.
Use the existing coverage tier that best matches the model:
- runtime parity for stable CPU-supported paths
- export smoke for supported families without stable CPU parity yet

7. Validate before claiming support.
Read `references/validation-playbook.md`, run the narrowest relevant test slice first, then run the full quickcheck if shared causal/export/cache code changed.

## Repo Hotspots
- Model wrappers: `QEfficient/transformers/models/*/modeling_*.py`
- Replacement wiring: `QEfficient/transformers/models/pytorch_transforms.py`
- Auto/export/runtime glue: `QEfficient/transformers/models/modeling_auto.py`
- Cache abstractions: `QEfficient/transformers/cache_utils.py`
- Runtime input helpers: `QEfficient/utils/generate_inputs.py`
- Runtime cache conversion: `QEfficient/utils/run_utils.py`
- Acceptance gate: `tests/test_model_quickcheck.py`

## Decision Rules
- Start from Llama, Mistral, Qwen2, Granite, or Olmo2 when the model is a plain decoder-only RoPE/GQA causal LM.
- Start from GPT-OSS, Mixtral, Qwen3-MoE, or GLM4-MoE-Lite when routed experts materially affect forward, export, or cache behavior.
- Start from Gemma3 when hybrid cache or dynamic per-layer sequence behavior is present.
- Start from Qwen2.5-VL, InternVL, Gemma3, Mllama, or related VLM wrappers when non-text inputs are first-class.
- Start from GLM4-MoE-Lite when the model uses MLA, absorption, compressed KV, or rope-side compressed cache behavior.
- Start from Whisper, CTC, or embedding paths instead of causal-LM code for speech, encoder-decoder, or representation models.

## Non-Negotiables
- Preserve HF PyTorch to QEff PyTorch parity before treating ONNX or compile success as sufficient.
- Keep tuple-cache compatibility whenever shared export or runtime code still expects legacy cache layout.
- Prefer feature detection and boundary normalization over version-specific hacks.
- For MoE architectures, start with a weights-as-activation implementation as the default decode path. Use this as the first integration mode because it is faster to validate and simpler to debug; add alternate/prefill-specialized paths only after baseline parity and export stability are proven.
- In MoE/router paths, avoid `.sum(..., keepdim=True)` when it creates non-constant ONNX reduction axes under subfunctions; prefer `torch.einsum(...)`-based normalization for export/compile stability.
- For `use_onnx_subfunctions=True`, avoid shape-derived reduction/slice expressions in model logic (for example runtime `.sum(..., keepdim=True)` axes or `x[..., :tensor.shape[-1]]` bounds). Prefer compile-stable forms (for example `einsum` normalization and config-constant dimensions) so ONNX function-local axes/bounds remain constant and compile cleanly.
- Do not add a bespoke test file when `tests/test_model_quickcheck.py` can carry the regression.

## References
- Load `references/model-family-map.md` when deciding which local modeling file to copy or extend.
- Load `references/validation-playbook.md` when choosing tiny models, export commands, and quickcheck coverage.
