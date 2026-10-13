# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run Cohere ASR QPC inference on selected dataset clips."""

import argparse
import re
import unicodedata
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "validation"


def parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("indices must contain non-negative integers")
    return indices


def reset_session(model: QEFFAutoModelForSpeechSeq2Seq) -> None:
    if model.qpc_session is not None:
        model.qpc_session.deactivate()
        model.qpc_session = None


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(character if character.isalnum() else " " for character in value)
    return re.sub(r"\s+", " ", value).strip()


def trim_at_eos(token_ids: list[int], eos_token_id: int) -> tuple[list[int], bool]:
    try:
        eos_index = token_ids.index(eos_token_id)
    except ValueError:
        return token_ids, False
    return token_ids[: eos_index + 1], True


def resolve_qpc_path(model: QEFFAutoModelForSpeechSeq2Seq, args: argparse.Namespace) -> Path:
    if args.qpc_path is not None:
        if not args.qpc_path.is_dir():
            raise ValueError(f"QPC directory does not exist: {args.qpc_path}")
        return args.qpc_path

    compile_dir = args.compile_dir
    qpc_path = model.compile(
        compile_dir=str(compile_dir),
        batch_size=args.batch_size,
        num_devices=1,
        num_cores=args.num_cores,
        encoder_ctx_len=args.encoder_ctx_len,
        ctx_len=args.ctx_len,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        use_onnx_subfunctions=False,
    )
    return Path(qpc_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Cohere ASR model ID or local snapshot")
    parser.add_argument("--qpc-path", type=Path, help="Existing QPC directory; skips compilation")
    parser.add_argument(
        "--compile-dir",
        type=Path,
        default=Path("cohere_asr_compile"),
        help="Directory for export and QPC compilation when --qpc-path is omitted",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Static QPC batch size")
    parser.add_argument("--num-cores", type=int, default=16, help="QPC accelerator cores")
    parser.add_argument("--encoder-ctx-len", type=int, default=438, help="Cohere encoder context length")
    parser.add_argument("--ctx-len", type=int, default=512, help="Cohere decoder context length")
    parser.add_argument("--language", required=True, help="Processor language prompt, for example en or ar")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--indices", type=parse_indices, default=parse_indices("0,1,2"))
    parser.add_argument("--audio-path", type=Path, help="Local WAV/FLAC audio file to transcribe")
    parser.add_argument("--reference-text", help="Optional reference transcript for a local audio file")
    parser.add_argument("--generation-len", type=int, default=500)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.generation_len <= 0 or args.batch_size <= 0 or args.num_cores <= 0:
        parser.error("--generation-len, --batch-size, and --num-cores must be positive")
    if args.encoder_ctx_len <= 0 or args.ctx_len <= 0:
        parser.error("--encoder-ctx-len and --ctx-len must be positive")
    if args.qpc_path is not None and not args.qpc_path.is_dir():
        parser.error(f"QPC directory does not exist: {args.qpc_path}")
    if args.audio_path is not None and not args.audio_path.is_file():
        parser.error(f"audio file does not exist: {args.audio_path}")

    if args.audio_path is None:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
        invalid = [index for index in args.indices if index >= len(dataset)]
        if invalid:
            parser.error(f"dataset indices out of range for {len(dataset)} clips: {invalid}")
        samples = [
            (
                str(index),
                dataset[index][args.audio_column]["array"],
                int(dataset[index][args.audio_column]["sampling_rate"]),
                dataset[index].get(args.text_column, ""),
            )
            for index in args.indices
        ]
    else:
        audio_values, sample_rate = sf.read(args.audio_path, dtype="float32", always_2d=False)
        if audio_values.ndim == 2:
            audio_values = audio_values.mean(axis=1)
        samples = [(args.audio_path.name, audio_values, sample_rate, args.reference_text or "")]

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        dtype="float32",
    )
    qpc_path = resolve_qpc_path(model, args)
    model.qpc_path = qpc_path

    print(f"model={args.model_name}")
    print(f"qpc={qpc_path}")
    if args.audio_path is None:
        print(f"dataset={args.dataset_name}/{args.dataset_config}:{args.dataset_split}")
    else:
        print(f"audio={args.audio_path}")
    print(f"language={args.language} device={args.device_id}")

    try:
        for clip_id, audio_values, sample_rate, reference in samples:
            inputs = processor(
                audio_values,
                sampling_rate=sample_rate,
                language=args.language,
                return_tensors="pt",
            )
            chunk_count = inputs["input_features"].shape[0]
            chunk_transcriptions = []
            token_ids = []
            eos_reached = True
            for chunk_index in range(chunk_count):
                chunk_inputs = {
                    name: value[chunk_index : chunk_index + 1]
                    if getattr(value, "shape", ()) and value.shape[0] == chunk_count
                    else value
                    for name, value in inputs.items()
                }
                execution = model.generate(
                    inputs=chunk_inputs,
                    generation_len=args.generation_len,
                    device_ids=[args.device_id],
                )
                metrics = execution.perf_metrics
                chunk_tokens, chunk_eos_reached = trim_at_eos(
                    execution.generated_ids[0].tolist(), model.model.config.eos_token_id
                )
                token_ids.extend(chunk_tokens)
                eos_reached = eos_reached and chunk_eos_reached
                chunk_transcriptions.append(
                    processor.batch_decode(execution.generated_ids, skip_special_tokens=True)[0].strip()
                )
                print(
                    f"chunk[{chunk_index}] prefill_s={metrics.prefill_time:.6f} "
                    f"decode_tokens_per_s={metrics.decode_perf:.6f} "
                    f"total_s={metrics.total_time:.6f}"
                )
                reset_session(model)
            transcription = " ".join(text for text in chunk_transcriptions if text)
            print(f"\nclip[{clip_id}]")
            print(f"processor_chunks:      {chunk_count}")
            print(f"desired_text:          {reference}")
            print(f"actual_generated_text: {transcription}")
            if reference:
                print(f"exact_text_match:      {transcription == reference}")
                print(f"normalized_text_match: {normalize_text(transcription) == normalize_text(reference)}")
            print(f"eos_reached:           {eos_reached}")
            print(f"generated_ids_to_eos:  {token_ids}")
    finally:
        reset_session(model)


if __name__ == "__main__":
    main()
