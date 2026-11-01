# Audio Examples

Examples for running audio processing models on Qualcomm Cloud AI 100.

## Dependencies

Install required packages:
```bash
pip install librosa==0.10.2 soundfile==0.13.1
```

## Authentication

For private/gated models, export your HuggingFace token:
```bash
export HF_TOKEN=<your_huggingface_token>
```

## Supported Models

**QEff Auto Classes:** 
- `QEFFAutoModelForSpeechSeq2Seq` (for Whisper models)
- `QEFFAutoModelForCTC` (for Wav2Vec2 models)

For the complete list of supported audio models, see the [Validated Models - Audio Section](../../docs/source/validate.md#audio-models).

Popular models include:
- Whisper (tiny, base, small, medium, large, large-v3-turbo)
- Wav2Vec2 (base-960h)

## Available Examples

### Qwen3-ASR optimized AI100 profile

From the repository root, compile the mask-free 16-core profile on an AI100
device:

```bash
python qwen_asr_onefile.py compile \
    --profile best \
    --ctx-len 512 \
    --chunk-seconds 30 \
    --batch-size 1 \
    --num-cores 16 \
    --device-ids <assigned-device-id>
```

The compiler writes a QPC manifest path to stdout. Use it to transcribe one
audio file or benchmark a directory:

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

This profile uses FP16 KV/logits and MXFP6 on the logits MatMul. It checks that
the compiled ONNX input contract omits decoder `attention_mask`; the audio
`input_features_mask` is retained. The measured 50-clip reference reached
14.76x request RTFx and 3.06% WER. See the [Qwen3-ASR build notes](../../docs/source/qwen3_asr_changes.md)
for benchmark scope and comparison caveats.

### speech_to_text.py
Speech-to-text transcription using Whisper models.

**Usage:**
```bash
# With default parameters
python speech_to_text.py \

# With custom parameters
python speech_to_text.py \
    --model-name openai/whisper-tiny \
    --ctx-len 25 \
    --num-cores 16
```

**Parameters:**
- `--model-name`: HuggingFace Whisper model ID (default: `openai/whisper-tiny`)
- `--ctx-len`: Context length for generation (default: `25`)
- `--num-cores`: Number of cores (default: `16`)

This example:
- Loads a sample audio from the librispeech dataset
- Uses Whisper-tiny model by default
- Compiles and runs inference on Cloud AI 100
- Outputs the transcribed text

### wav2vec2_inference.py
Speech recognition using Wav2Vec2 models with CTC (Connectionist Temporal Classification).

**Usage:**
```bash
# With default parameters
python wav2vec2_inference.py

# With custom parameters
python wav2vec2_inference.py \
    --model-name facebook/wav2vec2-base-960h \
    --num-cores 16
```

**Parameters:**
- `--model-name`: HuggingFace CTC model ID (default: `facebook/wav2vec2-base-960h`)
- `--num-cores`: Number of cores (default: `16`)

This example:
- Loads a sample audio from the librispeech dataset
- Uses Wav2Vec2-base-960h model by default
- Compiles and runs inference on Cloud AI 100
- Outputs the recognized text

## Documentation

- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html)
- [Validated Audio Models](https://quic.github.io/efficient-transformers/source/validate.html#audio-models)
- [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html)
