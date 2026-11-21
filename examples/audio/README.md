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
- `QEffChatterboxMultilingualTTS` (text-to-speech, Chatterbox Multilingual)

For the complete list of supported audio models, see the [Validated Models - Audio Section](../../docs/source/validate.md#audio-models).

Popular models include:
- Whisper (tiny, base, small, medium, large, large-v3-turbo)
- Wav2Vec2 (base-960h)

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

### text_to_speech.py
Text-to-speech with Resemble AI Chatterbox Multilingual (23 languages). The default config gives T3, the acoustic
language model, 16 cores and the four S3Gen graphs (flow encoder, flow estimator, HiFT F0 predictor and decoder) 16
cores together, so it needs two SoCs' worth of cores; the runtime picks the devices.

Multilingual V3 is only in upstream git (not in the `chatterbox-tts` 0.1.7 release), and Chatterbox pins older
torch/transformers releases, so install the pinned commit without its dependencies, then its runtime dependencies:
```bash
pip install --no-deps "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox@5de7a54aa4e5e2baadb0182dde554908b48b85c2"
# torchaudio must match QEfficient's torch 2.7.0 CPU wheel; install it first so s3tokenizer does not pull a newer torch
pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
pip install s3tokenizer==0.3.0 conformer==0.3.2 einops==0.8.1 omegaconf==2.4.0 librosa==0.10.2 \
    resemble-perth==1.0.1 pykakasi==2.3.0 spacy-pkuseg==1.0.2 "setuptools<81"
```

```bash
# With default parameters (English, built-in voice, default compile config)
python text_to_speech.py

# French, cloning the voice of a reference recording
python text_to_speech.py --language fr --text "Bonjour, comment allez-vous?" --audio-prompt reference.wav \
    --output bonjour.wav

# Pin devices or change cores: copy QEfficient/tts/pipelines/configs/chatterbox_config.json, edit
# execute.device_ids / compilation, and pass it
python text_to_speech.py --compile-config my_chatterbox_config.json
```

## Documentation

- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html)
- [Validated Audio Models](https://quic.github.io/efficient-transformers/source/validate.html#audio-models)
- [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html)
