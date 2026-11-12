# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import argparse

import soundfile as sf

from QEfficient import QEffChatterboxMultilingualTTS


def main():
    parser = argparse.ArgumentParser(description="Text-to-speech with Chatterbox Multilingual")
    parser.add_argument("--model-name", type=str, default="ResembleAI/chatterbox", help="HuggingFace model ID")
    parser.add_argument(
        "--t3-model", type=str, default="v3", help="Multilingual T3 checkpoint: v3 (validated) or v2 (untested)"
    )
    parser.add_argument("--text", type=str, default="Hello from Cloud AI 100. This voice was generated on the card.")
    parser.add_argument("--language", type=str, default="en", help="Language id, e.g. en, fr, de, zh, ja, ar, hi")
    parser.add_argument("--audio-prompt", type=str, default=None, help="Optional reference WAV to clone a voice")
    parser.add_argument("--compile-config", type=str, default=None, help="Optional JSON config (cores, devices)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="chatterbox.wav")
    args = parser.parse_args()
    if QEffChatterboxMultilingualTTS is None:
        raise ImportError(
            "QEffChatterboxMultilingualTTS is unavailable; see the warning above, and examples/audio/README.md for the "
            "install steps (text_to_speech.py)."
        )

    ## STEP 1 -- load the model on CPU (upstream ChatterboxMultilingualTTS, float32)
    pipeline = QEffChatterboxMultilingualTTS.from_pretrained(args.model_name, t3_model=args.t3_model)

    ## STEP 2 -- export and compile the five graphs (default config unless --compile-config is given)
    pipeline.compile(compile_config=args.compile_config)

    ## STEP 3 -- synthesize
    output = pipeline.generate(
        args.text, language_id=args.language, audio_prompt_path=args.audio_prompt, seed=args.seed
    )
    print(output)

    ## STEP 4 -- save the waveform
    sf.write(args.output, output.audio, output.sample_rate)
    print(f"Saved {output.audio_seconds:.2f} s of audio to {args.output}")


if __name__ == "__main__":
    main()
