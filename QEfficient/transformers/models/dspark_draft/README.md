# DSpark draft model

DSpark checkpoints from `deepseek-ai` (Qwen3 4B/8B/14B and Gemma4 12B) and `RadixArk/Qwen3.8-27B-DSpark` use checkpoint-compatible diffusion backbones, a
target-side concatenation/projector for target features, a rank-256 vanilla Markov head,
and a Markov-conditioned confidence head. This implementation follows DeepSpec's
DSpark attention/head equations and the published checkpoint's flat weight layout.
It does not require SpecForge or `trust_remote_code=True`.

## Loading

The draft checkpoint does not contain token embeddings or an LM head. Supply a
compatible target checkpoint containing `model.embed_tokens.weight` (or
`model.language_model.embed_tokens.weight` for the Qwen3.5 vision-language target) and
`lm_head.weight` (or tied embeddings). These tensors must be unquantized and match the draft checkpoint's `vocab_size` and `hidden_size`; packed quantized heads are not supported.

```python
from QEfficient import QEFFAutoModelForCausalLM

draft = QEFFAutoModelForCausalLM.from_pretrained(
    "RadixArk/Qwen3.8-27B-DSpark",
    qaic_config={
        "dspark_dlm": True,
        "dspark_tlm_repo": "/path/to/compatible/target/checkpoint",
    },
)
onnx_path = draft.export()
```

Set `HF_HUB_CACHE` to your model cache and `HF_HUB_ENABLE_HF_TRANSFER=1`
before downloading checkpoints. Local checkpoint directories also work.

## Forward/export contract

- `input_ids`: known bonus/anchor token followed by MASK tokens,
  `[batch_size, block_size]`. The first draft output predicts the token **after**
  that anchor. Noise positions start at the anchor's absolute position.
- `target_hidden`: projected and normalized target features with shape
  `[batch_size, target_seq_len, hidden_size]`. The target concatenates the
  checkpoint's `target_layer_ids`, applies `fc.weight` and
  `hidden_norm.weight`, then returns the projected features.
- `position_ids_target` / `position_ids`: absolute target/noise positions,
  `[batch_size, block_size]`. Padded target entries use position `-1`.
- `prev_token_ids`: the known token immediately preceding the first proposal,
  `[batch_size, 1]`. Each subsequent Markov bias conditions on the previous
  proposal's greedy token.
- `past_key_values`: preallocated legacy tuple cache or `QEffDynamicCache`.
  Both target and noise keys/values are written at their absolute positions.
- Optional `batch_index` selects retained-cache rows for continuous batching.
- Optional `attention_mask` is a 4-D boolean mask; `True` means masked.

`forward()` returns float32 Markov-corrected logits and retained KV state.
Greedy proposals are `outputs.logits.argmax(dim=-1)`. Export and compilation use
the checkpoint's serving `block_size=7`, not `training_block_size=16`.
The compatibility argument `dflash_block_size`, if provided to compilation,
must equal the checkpoint's block size.

The confidence head is loaded and available through
`draft.model.model.predict_confidence_step(hidden_states, prev_token_ids)`;
this returns confidence **logits**, not sigmoid probabilities. Confidence is
not part of the exported causal-LM output contract and adaptive block-length
selection is not implemented. Generic sampling, stochastic DSpark decoding,
and weight-free/layerwise loading are not implemented. The DFlash-integrated performance runner in `examples/performance/dflash` with `--technique dspark` handles greedy verification. Qwen3 and Gemma4 use retained-KV overwrite; Qwen3.5 reuses fresh-main fused DFlash state management while supplying DSpark projected target features and `prev_token_ids`.

## Hybrid Qwen3.5 target

The published Qwen3.8 target uses the Qwen3.5 architecture, including linear
attention and vision. Its target transform uses the linear-attention SPD
technique from efficient-transformers PR #1472:

```python
from QEfficient import QEFFAutoModelForImageTextToText

target = QEFFAutoModelForImageTextToText.from_pretrained(
    "/path/to/compatible/target/checkpoint",
    kv_offload=True,
    qaic_config={
        "dspark_tlm": True,
        "dspark_dlm_repo": "RadixArk/Qwen3.8-27B-DSpark",
        "target_layer_ids": [6, 20, 34, 48, 62],
        "dspark_block_size": 7,
    },
)
```

Each target decode call has width sixteen: the first eight positions replay the prior accepted prefix and the second eight positions verify `[anchor] + seven proposals`. Rejected or padded replay positions use `-1`; `cache_commit=0` commits only valid positions from the first half while leaving the current proposal suffix uncommitted. The next round carries that current block into the replay half, so Qwen3.5 needs one target call per speculation round. Position zero resets recurrent state for a new prompt. Prefill width must be a positive multiple of eight and at least sixteen.

`DSparkTLMTransform` loads the trained `fc.weight` and `hidden_norm.weight` from `dspark_dlm_repo` into the target before enabling the existing DFlash target-layer collection. This reduces the target-to-draft feature width from 25,600 to 5,120. Fresh causal Qwen3.5 returns vocabulary logits; the performance adapter converts them to greedy token IDs. `num_logits_to_keep` is `[1, 1]` for prefill and `[8, 1]` for fused decode. Target batch size is one, without continuous batching or layerwise execution. See `examples/performance/dflash/dflash-dspark/README.md`
for compilation, text/vision entry points, and offline smoke commands.

## Validation

Focused CPU tests live alongside the DFlash tests:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q \
    tests/unit_test/transforms/test_dflash_transform.py
```

They cover checkpoint loading, GQA/YaRN attention reference parity, confidence
and sequential Markov heads, legacy-cache/continuous-batching parity, fixed
block specializations, and PyTorch-to-ONNXRuntime logits parity on tiny models.
The ONNXRuntime test uses the existing quickcheck session helper to normalize
QAIC's invalid cache-index sentinel for CPU execution; it does not change the
exported artifact. These checks do not establish full-checkpoint or QAIC
hardware validation.
