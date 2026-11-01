# Qwen3-ASR Transformers Changes

This PR adds the Transformers integration needed to compile and execute
`Qwen/Qwen3-ASR-0.6B-hf` through QEfficient.

## Transformers-side changes

- Loads Qwen3-ASR with `AutoModelForSpeechSeq2Seq`.
- Uses `QEFFAutoModelForSpeechSeq2Seq` to create the QEfficient model wrapper.
- Loads the matching `AutoProcessor` and `AutoConfig` from the same model ID.
- Builds processor input with the Qwen ASR audio chat template:
  `processor.apply_chat_template(..., add_generation_prompt=True)`.
- Uses eager attention during model loading for the Qwen3-ASR export path.
- Uses float32 model weights during export and applies QEfficient's Transformers
  quantizer replacement before compilation.
- Converts processor position IDs and input-feature masks to the integer types
  expected by the QEfficient/QPC graph.
- Handles the model's audio-feature context and audio-token sizing when setting
  the compiled encoder and decoder context lengths.

## Compatibility

- The model ID is `Qwen/Qwen3-ASR-0.6B-hf`.
- The environment must provide a Transformers version compatible with the
  installed QEfficient checkout and the Qwen3-ASR architecture.
- No upstream Transformers source files were modified by this PR; the integration
  is implemented through the standalone `qwen_asr_onefile.py` workflow.

## Verification note

Syntax and Cloud AI 100 smoke checks passed. These checks do not establish
numerical parity; correctness requires HF/QEfficient/ORT/QPC parity evidence.

## Fast AI100 QPC profile

`qwen_asr_onefile.py` now defaults to the `best` compile profile. From the repo
root, on a Cloud AI 100 host with the QEfficient dependencies and compiler set
up, run:

```bash
python qwen_asr_onefile.py compile \
  --profile best \
  --ctx-len 512 \
  --chunk-seconds 30 \
  --batch-size 1 \
  --num-cores 16 \
  --device-ids <assigned-device-id>
```

The command exports the model, checks that the ONNX graph has no decoder
`attention_mask` input, finds the MatMul feeding `logits`, and selects that
MatMul for MXFP6. It compiles the remaining graph with FP16 conversion and
FP16 KV cache/logits, a full vocabulary, batch size 1, 512-token context and
30-second audio windows. `input_features_mask` remains part of the model input
contract. The dynamically generated selector and QPC manifest are saved with
the build artifacts. Use the manifest path printed by compilation:

```bash
python qwen_asr_onefile.py run \
  --manifest /path/to/qpc_manifest.json \
  --audio-file /path/to/audio.wav \
  --device-ids <assigned-device-id>

python qwen_asr_onefile.py benchmark \
  --manifest /path/to/qpc_manifest.json \
  --audio-dir /path/to/audio-directory \
  --device-ids <assigned-device-id> \
  --output-json benchmark.json
```

The benchmark command reuses one QPC session and reports request RTFx, including
processor preprocessing, generation, and text decoding. Session creation and
audio file loading are outside the timed requests. Its timing procedure differs
from the measured result below, so compare runs only when using the same script,
device, data, and settings.

### Measured reference

The recorded mask-free QPC was built for AI100 with 16 cores, 512-token
context/prefill, single-token decode, 3,000 audio feature frames, a 30-second
window, FP16 KV/logits, and MXFP6 on the logits MatMul. In a serial direct-QEff
run over 50 LibriSpeech Test-Clean clips it measured 14.76x request RTFx and
15.52x generation-only RTFx, with 3.06% WER (28/915 words). Its transcripts and
token histories matched the optimized attention-mask reference.

The reference measured 14.38x request RTFx and 3.06% WER. It ran on device 60;
the mask-free QPC ran on device 1, and clocks were not locked. These single runs
do not establish a speedup from removing the mask. Treat 15x as a measured
reference point, not a performance guarantee; hardware, compiler/driver
versions, clocks, and benchmark timing scope affect results.
