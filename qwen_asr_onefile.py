#!/usr/bin/env python3
"""Compile and run Qwen3-ASR QPCs from one portable script.

The script intentionally has no imports from the older Qwen helper scripts.
    Runtime dependencies (QEfficient, transformers, soundfile, scipy, and the
Cloud AI SDK) are imported only when the selected command needs them.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from time import perf_counter

import numpy as np


MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"
SAMPLE_RATE = 16_000
FEATURE_FRAMES_PER_SECOND = 100
FEATURE_CHUNK_LEN = 100
AUDIO_TOKENS_PER_FEATURE_CHUNK = 13
DEFAULT_AUDIO = "audio_test.flac"
DEFAULT_AUDIO_DIR = Path(
    "/prj/qct/aisyssol_scratch/users/mabusaa/cohere_asr/performance/"
    "production_manifests/wav/english"
)


def repo_root() -> Path:
    """Return the directory containing this standalone script."""
    return Path(__file__).resolve().parent


def prepare_qeff_imports(qeff_root: Path | None = None) -> None:
    """Use installed QEfficient, or an explicitly supplied checkout."""
    selected = qeff_root or (Path(os.environ["QEFF_ROOT"]) if os.environ.get("QEFF_ROOT") else None)
    if selected is not None:
        selected = selected.expanduser().resolve()
        if not (selected / "QEfficient").is_dir():
            raise SystemExit(f"--qeff-root does not contain QEfficient/: {selected}")
        sys.path.insert(0, str(selected))


def load_qaic_session_class(qeff_root: Path | None = None):
    """Load QAICInferenceSession without importing all of QEfficient when possible."""
    if qeff_root is not None:
        module_path = qeff_root.expanduser().resolve() / "QEfficient" / "generation" / "cloud_infer.py"
        if not module_path.is_file():
            raise SystemExit(f"missing cloud_infer.py under --qeff-root: {module_path}")
        spec = importlib.util.spec_from_file_location("qwen_asr_cloud_infer", module_path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"could not load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.QAICInferenceSession
    from QEfficient.generation.cloud_infer import QAICInferenceSession

    return QAICInferenceSession


def device_ids(value: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --device-ids value: {value}") from exc
    if not ids:
        raise SystemExit("--device-ids must contain at least one device")
    if 43 in ids:
        raise SystemExit("refusing to use device 43")
    return ids


def set_runtime_env(ids: list[int]) -> None:
    value = ",".join(str(item) for item in ids)
    os.environ["DEVICE_GROUP"] = value
    os.environ.pop("QAIC_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def manifest_from_arg(root: Path, value: Path | None) -> Path:
    if value is not None:
        path = value.expanduser().resolve()
    else:
        candidates = sorted(
            (root / ".archon/artifacts/runs/qwen3_asr").glob("**/qpc_manifest.json"),
            key=lambda item: item.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit("no Qwen3-ASR QPC manifest found; compile one first")
        path = candidates[-1].resolve()
    if not path.is_file():
        raise SystemExit(f"manifest not found: {path}")
    return path


def compile_qpc(args: argparse.Namespace, root: Path) -> Path:
    if args.chunk_seconds <= 0 or args.ctx_len < 1 or args.batch_size != 1:
        raise SystemExit("chunk-seconds and ctx-len must be positive; only batch size 1 is supported")
    ids = device_ids(args.device_ids)
    encoder_ctx_len = ceil_multiple(
        round(args.chunk_seconds * FEATURE_FRAMES_PER_SECOND), FEATURE_CHUNK_LEN
    )
    feature_chunks = encoder_ctx_len // FEATURE_CHUNK_LEN
    audio_tokens = feature_chunks * AUDIO_TOKENS_PER_FEATURE_CHUNK
    prefill_len = max(args.ctx_len, audio_tokens)
    if args.ctx_len < audio_tokens:
        raise SystemExit(f"ctx-len={args.ctx_len} is smaller than audio token count {audio_tokens}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = args.output_root or root / ".archon/artifacts/runs/qwen3_asr/onefile-qpc"
    run_dir = base.expanduser().resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    qeff_home = run_dir / "qeff_home"
    qeff_home.mkdir()
    set_runtime_env(ids)
    os.environ["QEFF_HOME"] = str(qeff_home)
    prepare_qeff_imports(args.qeff_root)

    import torch
    from transformers import AutoModelForSpeechSeq2Seq
    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSpeechSeq2Seq
    from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers

    replace_transformers_quantizers()
    request = {
        "model": args.model_id,
        "chunk_seconds": args.chunk_seconds,
        "chunk_samples": round(args.chunk_seconds * SAMPLE_RATE),
        "overlap_seconds": 0,
        "stride_seconds": args.chunk_seconds,
        "encoder_ctx_len": encoder_ctx_len,
        "audio_feature_chunks": feature_chunks,
        "audio_tokens": audio_tokens,
        "ctx_len": args.ctx_len,
        "prefill_seq_len": prefill_len,
        "batch_size": args.batch_size,
        "num_cores": args.num_cores,
        "num_devices": len(ids),
        "device_ids": ids,
        "qeff_home": str(qeff_home),
        "started_at_utc": stamp,
    }
    (run_dir / "compile_request.json").write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id, attn_implementation="eager", low_cpu_mem_usage=False, dtype=torch.float32
    ).eval()
    qeff = QEFFAutoModelForSpeechSeq2Seq(model, pretrained_model_name_or_path=args.model_id)
    qpc_dir = Path(qeff.compile(
        ctx_len=args.ctx_len,
        encoder_ctx_len=encoder_ctx_len,
        batch_size=args.batch_size,
        num_devices=len(ids),
        num_cores=args.num_cores,
    )).resolve()
    qpc_bin = qpc_dir / "programqpc.bin"
    if not qpc_bin.is_file():
        raise SystemExit(f"compile finished without programqpc.bin: {qpc_dir}")
    manifest = {
        **request,
        "qpc_dir": str(qpc_dir),
        "qpc_bin": str(qpc_bin),
        "size_bytes": qpc_bin.stat().st_size,
        "finished_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    manifest_path = run_dir / "qpc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "qpc_path.txt").write_text(str(qpc_dir) + "\n", encoding="utf-8")
    print(f"qpc_manifest={manifest_path}\nqpc_dir={qpc_dir}\nqpc_size_bytes={qpc_bin.stat().st_size}")
    return manifest_path


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"audio file not found: {path}")
    import soundfile as sf
    from scipy.signal import resample_poly

    source = path
    with tempfile.TemporaryDirectory(prefix="qwen3_asr_video_") as temp:
        if path.suffix.lower() not in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
            source = Path(temp) / "audio.wav"
            try:
                subprocess.run(["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), str(source)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except FileNotFoundError as exc:
                raise SystemExit("ffmpeg is required for video/non-audio inputs") from exc
        audio, source_rate = sf.read(str(source), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1)
    if not audio.size:
        raise SystemExit(f"audio stream is empty: {path}")
    if int(source_rate) != sample_rate:
        factor = gcd(int(source_rate), sample_rate)
        audio = resample_poly(audio, sample_rate // factor, int(source_rate) // factor).astype(np.float32)
    return audio


def processor_inputs(processor, audio: np.ndarray, sample_rate: int, chunk_samples: int) -> dict:
    import torch
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio"}]}], add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=prompt, audio=audio[:chunk_samples], sampling_rate=sample_rate, return_tensors="pt")
    length = inputs["input_ids"].shape[1]
    inputs["position_ids"] = torch.arange(length, dtype=torch.int64).view(1, length)
    if "input_features_mask" in inputs:
        inputs["input_features_mask"] = inputs["input_features_mask"].to(torch.int64)
    return dict(inputs)


def allowed_shapes(session, predicate) -> dict[str, tuple[int, ...]]:
    for allowed in getattr(session, "allowed_shapes", []):
        shapes = {binding.name: tuple(shape) for binding, (_, shape) in zip(session.bindings, allowed)}
        if predicate(shapes):
            return shapes
    raise SystemExit("QPC has no matching prefill/decode specialization")


def pad_prefill(inputs: dict, length: int) -> dict:
    import torch
    actual = int(inputs["input_ids"].shape[1])
    if actual > length:
        raise SystemExit(f"processor prompt length {actual} exceeds compiled prefill length {length}")
    result = dict(inputs)
    for name in ("input_ids", "attention_mask", "position_ids"):
        value = inputs[name]
        padded = torch.zeros((value.shape[0], length), dtype=value.dtype)
        padded[:, :actual] = value
        result[name] = padded
    return result


def retained(outputs: dict, inputs: dict, names: set[str]) -> None:
    suffix = "_RetainedState"
    for output_name, value in outputs.items():
        if output_name.endswith(suffix) and output_name[:-len(suffix)] in names:
            inputs[output_name[:-len(suffix)]] = value


def bind_output_buffers(session, output_names: set[str], shapes: dict[str, tuple[int, ...]]) -> None:
    """Bind buffers for every ordinary output in the selected specialization."""
    buffers = {}
    for binding in session.bindings:
        name = binding.name
        if name not in output_names or name.endswith("_RetainedState") or name not in shapes:
            continue
        dtype = session.aic_to_np_dtype_mapping.get(binding.type, np.dtype(np.float32))
        buffers[name] = np.zeros(shapes[name], dtype=dtype)
    if buffers:
        session.set_buffers(buffers)


def token_ids(value) -> set[int]:
    if value is None:
        return set()
    return {int(item) for item in value} if isinstance(value, (list, tuple)) else {int(value)}


def run_one(session, processor, audio: np.ndarray, sample_rate: int, chunk_samples: int, generation_len: int, stop_on_eos: bool, prefill_shapes, decode_shapes, input_names, eos_ids) -> dict:
    started = perf_counter()
    raw = processor_inputs(processor, audio, sample_rate, chunk_samples)
    padded = pad_prefill(raw, prefill_shapes["input_ids"][1])
    values = {
        "input_features": padded["input_features"].detach().cpu().numpy().astype(np.float16),
        "input_ids": padded["input_ids"].detach().cpu().numpy().astype(np.int64),
        "attention_mask": padded["attention_mask"].detach().cpu().numpy().astype(np.int64),
        "position_ids": padded["position_ids"].detach().cpu().numpy().astype(np.int64),
    }
    if "input_features_mask" in padded:
        values["input_features_mask"] = padded["input_features_mask"].detach().cpu().numpy().astype(np.int64)
    values = {name: value for name, value in values.items() if name in input_names}
    outputs = session.run(values)
    next_token = outputs["logits"].argmax(-1).astype(np.int64)
    generated = [int(next_token.reshape(-1)[0])]
    stopped = stop_on_eos and generated[-1] in eos_ids
    decode = {
        "input_features": np.zeros(decode_shapes["input_features"], dtype=np.float16),
        "attention_mask": np.ones(decode_shapes["attention_mask"], dtype=np.int64),
        "position_ids": np.full(decode_shapes["position_ids"], int(values["position_ids"].max()) + 1, dtype=np.int64),
    }
    if "input_features_mask" in input_names:
        decode["input_features_mask"] = np.ones(decode_shapes["input_features_mask"], dtype=np.int64)
    retained(outputs, decode, input_names)
    for _ in range(generation_len - 1):
        if stopped:
            break
        decode["input_ids"] = next_token
        outputs = session.run(decode)
        next_token = outputs["logits"].argmax(-1).astype(np.int64)
        retained(outputs, decode, input_names)
        generated.append(int(next_token.reshape(-1)[0]))
        stopped = stop_on_eos and generated[-1] in eos_ids
        if not stopped:
            decode["position_ids"] += 1
    tokens = np.asarray([generated], dtype=np.int64)
    return {"tokens": generated, "decoded": processor.batch_decode(tokens, skip_special_tokens=True)[0], "raw": processor.batch_decode(tokens, skip_special_tokens=False)[0], "stopped_on_eos": stopped, "elapsed_s": perf_counter() - started}


def open_runtime(manifest_path: Path, ids: list[int], qeff_root: Path | None = None):
    root = repo_root()
    prepare_qeff_imports(qeff_root)
    set_runtime_env(ids)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qpc_dir = Path(manifest["qpc_dir"]).expanduser().resolve()
    if not (qpc_dir / "programqpc.bin").is_file():
        raise SystemExit(f"missing QPC binary: {qpc_dir / 'programqpc.bin'}")
    from transformers import AutoConfig, AutoProcessor
    QAICInferenceSession = load_qaic_session_class(qeff_root)
    model_id = manifest.get("model", MODEL_ID)
    processor = AutoProcessor.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    session = QAICInferenceSession(str(qpc_dir), ids)
    names = set(session.input_names)
    prefill = allowed_shapes(session, lambda s: s.get("input_ids", (0, 0))[1] > 1 and s.get("input_features", (0, 0, 0))[-1] == int(manifest["encoder_ctx_len"]))
    decode = allowed_shapes(session, lambda s: s.get("input_ids") == (1, 1) and s.get("attention_mask") == (1, 1) and s.get("position_ids") == (1, 1))
    bind_output_buffers(session, set(session.output_names), decode)
    return manifest, processor, config, session, names, prefill, decode, token_ids(getattr(config, "eos_token_id", None))


def run_command(args: argparse.Namespace, root: Path) -> int:
    ids = device_ids(args.device_ids)
    manifest_path = manifest_from_arg(root, args.manifest)
    manifest, processor, config, session, names, prefill, decode, eos_ids = open_runtime(manifest_path, ids, args.qeff_root)
    chunk_seconds = float(args.chunk_seconds if args.chunk_seconds is not None else manifest.get("chunk_seconds", 30))
    chunk_samples = round(chunk_seconds * SAMPLE_RATE)
    overlap = float(args.overlap_seconds)
    if chunk_seconds <= 0 or overlap < 0 or overlap >= chunk_seconds:
        raise SystemExit("require chunk-seconds > 0 and 0 <= overlap-seconds < chunk-seconds")
    audio = load_audio(args.audio_file.resolve(), SAMPLE_RATE)
    stride = max(1, chunk_samples - round(overlap * SAMPLE_RATE))
    rows = []
    for index, start in enumerate(range(0, audio.size, stride)):
        chunk = audio[start:start + chunk_samples]
        if not chunk.size:
            break
        if chunk.size < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - chunk.size))
        result = run_one(session, processor, chunk, SAMPLE_RATE, chunk_samples, args.generation_len, args.stop_on_eos, prefill, decode, names, eos_ids)
        row = {"chunk_index": index, "chunk_start_s": start / SAMPLE_RATE, "chunk_end_s": (start + chunk_samples) / SAMPLE_RATE, **result}
        rows.append(row)
        print(f"chunk={index} start_s={row['chunk_start_s']:.3f} elapsed_s={result['elapsed_s']:.4f} token_ids={result['tokens']} transcript={result['decoded']!r}")
    transcript = "".join(row["decoded"] for row in rows)
    print(f"manifest={manifest_path}\nqpc_dir={manifest['qpc_dir']}\nchunks={len(rows)}\ntranscript={transcript!r}")
    return 0


def benchmark_command(args: argparse.Namespace, root: Path) -> int:
    ids = device_ids(args.device_ids)
    manifest_path = manifest_from_arg(root, args.manifest)
    manifest, processor, config, session, names, prefill, decode, eos_ids = open_runtime(manifest_path, ids, args.qeff_root)
    chunk_seconds = float(manifest.get("chunk_seconds", 30))
    overlap = float(args.overlap_seconds)
    if overlap < 0 or overlap >= chunk_seconds:
        raise SystemExit("overlap-seconds must be >= 0 and less than compiled chunk_seconds")
    chunk_samples = round(chunk_seconds * SAMPLE_RATE)
    stride = max(1, chunk_samples - round(overlap * SAMPLE_RATE))
    files = sorted(path for path in args.audio_dir.iterdir() if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".mp4", ".m4a"})
    if args.limit_files is not None:
        files = files[:args.limit_files]
    if not files:
        raise SystemExit(f"no audio files found under {args.audio_dir}")
    if args.warmup:
        warmup_start = perf_counter()
        run_one(session, processor, np.zeros(chunk_samples, dtype=np.float32), SAMPLE_RATE, chunk_samples, args.generation_len, args.stop_on_eos, prefill, decode, names, eos_ids)
        print(f"warmup_complete=True warmup_s={perf_counter() - warmup_start:.4f}", flush=True)
    rows = []
    for path in files:
        audio = load_audio(path, SAMPLE_RATE)
        for index, start in enumerate(range(0, audio.size, stride)):
            chunk = audio[start:start + chunk_samples]
            if chunk.size < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - chunk.size))
            result = run_one(session, processor, chunk, SAMPLE_RATE, chunk_samples, args.generation_len, args.stop_on_eos, prefill, decode, names, eos_ids)
            rows.append({"file": path.name, "chunk_index": index, "chunk_start_s": start / SAMPLE_RATE, "generated_tokens": len(result["tokens"]), "chunk_e2e_s": result["elapsed_s"], "decoded": result["decoded"]})
            print(f"{path.name} chunk={index} elapsed_s={result['elapsed_s']:.4f} tokens={len(result['tokens'])}", flush=True)
    summary = {"manifest": str(manifest_path), "audio_dir": str(args.audio_dir.resolve()), "file_count": len(files), "chunk_count": len(rows), "generation_len": args.generation_len, "stop_on_eos": args.stop_on_eos, "overlap_seconds": overlap, "qpc_session_reused": True, "total_elapsed_s": sum(row["chunk_e2e_s"] for row in rows)}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"summary": summary, "chunks": rows}, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audio-file", type=Path, default=repo_root() / DEFAULT_AUDIO)
    parser.add_argument("--generation-len", type=int, default=128)
    parser.add_argument("--device-ids", default="0")
    parser.add_argument("--qeff-root", type=Path, help="optional QEfficient checkout; otherwise use installed packages")
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--overlap-seconds", type=float, default=0)
    parser.add_argument("--stop-on-eos", dest="stop_on_eos", action="store_true")
    parser.add_argument("--no-stop-on-eos", dest="stop_on_eos", action="store_false")
    parser.set_defaults(stop_on_eos=True)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile", help="compile a QPC")
    compile_parser.add_argument("--model-id", default=MODEL_ID)
    compile_parser.add_argument("--chunk-seconds", type=float, default=30)
    compile_parser.add_argument("--ctx-len", type=int, default=512)
    compile_parser.add_argument("--batch-size", type=int, default=1)
    compile_parser.add_argument("--num-cores", type=int, default=8)
    compile_parser.add_argument("--device-ids", default="0")
    compile_parser.add_argument("--output-root", type=Path)
    compile_parser.add_argument("--qeff-root", type=Path, help="optional QEfficient checkout; otherwise use installed packages")
    run_parser = sub.add_parser("run", help="run one audio file")
    add_runtime_options(run_parser)
    benchmark_parser = sub.add_parser("benchmark", help="benchmark an audio directory")
    benchmark_parser.add_argument("--manifest", type=Path)
    benchmark_parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    benchmark_parser.add_argument("--generation-len", type=int, default=128)
    benchmark_parser.add_argument("--device-ids", default="0")
    benchmark_parser.add_argument("--qeff-root", type=Path, help="optional QEfficient checkout; otherwise use installed packages")
    benchmark_parser.add_argument("--overlap-seconds", type=float, default=0)
    benchmark_parser.add_argument("--stop-on-eos", dest="stop_on_eos", action="store_true")
    benchmark_parser.add_argument("--no-stop-on-eos", dest="stop_on_eos", action="store_false")
    benchmark_parser.set_defaults(stop_on_eos=True, limit_files=None)
    benchmark_parser.add_argument("--limit-files", type=int)
    benchmark_parser.add_argument("--output-json", type=Path)
    benchmark_parser.add_argument("--output-csv", type=Path)
    benchmark_parser.add_argument("--warmup", dest="warmup", action="store_true")
    benchmark_parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    benchmark_parser.set_defaults(warmup=True)
    compile_run = sub.add_parser("compile-run", help="compile, then run one audio file")
    compile_run.add_argument("--model-id", default=MODEL_ID)
    compile_run.add_argument("--ctx-len", type=int, default=512)
    compile_run.add_argument("--chunk-seconds", type=float, default=30)
    compile_run.add_argument("--batch-size", type=int, default=1)
    compile_run.add_argument("--num-cores", type=int, default=8)
    compile_run.add_argument("--output-root", type=Path)
    compile_run.add_argument("--device-ids", default="0")
    compile_run.add_argument("--audio-file", type=Path, default=root / DEFAULT_AUDIO)
    compile_run.add_argument("--generation-len", type=int, default=128)
    compile_run.add_argument("--overlap-seconds", type=float, default=0)
    compile_run.add_argument("--stop-on-eos", dest="stop_on_eos", action="store_true")
    compile_run.add_argument("--no-stop-on-eos", dest="stop_on_eos", action="store_false")
    compile_run.add_argument("--qeff-root", type=Path, help="optional QEfficient checkout; otherwise use installed packages")
    compile_run.set_defaults(stop_on_eos=True, output_root=None)
    args = parser.parse_args()
    if getattr(args, "generation_len", 1) < 1:
        raise SystemExit("--generation-len must be at least 1")
    if args.command == "compile":
        compile_qpc(args, root)
        return 0
    if args.command == "compile-run":
        manifest = compile_qpc(args, root)
        args.manifest = manifest
        return run_command(args, root)
    if args.command == "run":
        return run_command(args, root)
    return benchmark_command(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
