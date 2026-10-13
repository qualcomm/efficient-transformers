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
- Cohere Transcribe

## Available Examples

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

### cohere_asr_inference.py

Compile a Cohere ASR QPC and transcribe a local audio file, or reuse an
existing QPC. See [Cohere ASR on Cloud AI 100](COHERE_ASR.md) for the complete
customer workflow.

## Documentation

- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html)
- [Validated Audio Models](https://quic.github.io/efficient-transformers/source/validate.html#audio-models)
- [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html)
