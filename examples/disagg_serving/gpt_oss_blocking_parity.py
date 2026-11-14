#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compare GPT-OSS offline-prefill blocking modes using fresh serial artifacts."""

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

MODEL_ID = "openai/gpt-oss-20b"
PROMPT = "The largest planet in our solar system is"
BLOCKING_MODES = {
    "prefill_q": {"blocking_mode": "prefill_q", "num_q_blocks": 2},
    "prefill_kv": {"blocking_mode": "prefill_kv", "num_kv_blocks": 2, "headpar_split": 1},
    "prefill_qkv": {
        "blocking_mode": "prefill_qkv",
        "num_q_blocks": 2,
        "num_kv_blocks": 2,
        "headpar_split": 1,
    },
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Run GPT-OSS blocking modes serially and compare generated tokens.")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local snapshot path")
    parser.add_argument("--mode", choices=("all", *BLOCKING_MODES), default="all")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=2048)
    parser.add_argument("--generation-len", type=int, default=10)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--mdp-num-partitions", type=int, default=2)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--expert-chunk-size", type=int, default=256)
    parser.add_argument("--hf-hub-cache", default="/home/huggingface_hub")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--_worker-mode", choices=tuple(BLOCKING_MODES), help=argparse.SUPPRESS)
    parser.add_argument("--_case-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _compile_kwargs(args, *, num_devices=None, prefill_seq_len=None):
    return {
        "prefill_seq_len": args.prefill_seq_len if prefill_seq_len is None else prefill_seq_len,
        "ctx_len": args.ctx_len,
        "batch_size": 1,
        "num_devices": args.num_devices if num_devices is None else num_devices,
        "mdp_num_partitions": args.mdp_num_partitions,
        "num_cores": args.num_cores,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
        "retain_full_kv": True,
        "user_tiled": True,
    }


def _prepare_inputs(tokenizer, prompt, prefill_seq_len):
    import numpy as np

    encoded = tokenizer(prompt, return_tensors="np", padding=True)
    prompt_len = int(encoded["attention_mask"].sum())
    padded_len = -(-prompt_len // prefill_seq_len) * prefill_seq_len
    encoded = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    attention_mask = encoded.pop("attention_mask")
    encoded.pop("token_type_ids", None)
    encoded["position_ids"] = np.where(attention_mask, np.arange(padded_len), -1).astype(np.int64)
    return encoded, prompt_len


def _run_worker(args):
    import gc

    case_root = args._case_root.resolve()
    os.environ["TMPDIR"] = str(case_root / "tmp")
    os.environ["QEFF_HOME"] = str(case_root / "qeff_home")
    os.environ["QEFF_WF_HOME"] = str(case_root / "prepared_checkpoints")
    os.environ["XDG_CACHE_HOME"] = str(case_root / "xdg_cache")
    os.environ["HF_HUB_CACHE"] = args.hf_hub_cache
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    for directory in ("TMPDIR", "QEFF_HOME", "QEFF_WF_HOME", "XDG_CACHE_HOME"):
        Path(os.environ[directory]).mkdir(parents=True, exist_ok=True)

    result_path = case_root / "result.json"
    result = {
        "model": args.model,
        "blocking_mode": args._worker_mode,
        "weight_free": True,
        "configuration": {
            "prefill_seq_len": args.prefill_seq_len,
            "ctx_len": args.ctx_len,
            "generation_len": args.generation_len,
            "num_devices": args.num_devices,
            "mdp_num_partitions": args.mdp_num_partitions,
            "num_cores": args.num_cores,
            "mxfp6_matmul": True,
            "mxint8_kv_cache": True,
            "use_onnx_subfunctions": True,
        },
        "prompt": args.prompt,
        "status": "running",
        "stages": {},
        "artifacts": {"case_root": str(case_root)},
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_json(result_path, result)
    model = None
    torch = None
    current_stage = "model_load"
    try:
        import numpy as np
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from QEfficient import QEFFAutoModelForCausalLM
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        model_load_start = perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = QEFFAutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, weight_free=True)
        model.model.eval()
        result["stages"]["model_load"] = {
            "status": "passed",
            "seconds": round(perf_counter() - model_load_start, 3),
        }
        _write_json(result_path, result)

        mode_config = {
            **BLOCKING_MODES[args._worker_mode],
            "ctx_len": args.ctx_len,
            "moe_config": {"expert_parallel_chunk_size": args.expert_chunk_size},
        }
        prefill_compile_kwargs = _compile_kwargs(args)
        export_dir = case_root / "prefill_export"
        current_stage = "prefill_export"
        export_start = perf_counter()
        model.export(
            str(export_dir),
            prefill_only=True,
            enable_chunking=True,
            qaic_config=mode_config,
            **prefill_compile_kwargs,
        )
        result["stages"]["prefill_export"] = {
            "status": "passed",
            "seconds": round(perf_counter() - export_start, 3),
            "onnx_path": str(model.onnx_path),
        }
        _write_json(result_path, result)

        current_stage = "prefill_compile"
        prefill_compile_start = perf_counter()
        prefill_qpc = model.compile(
            onnx_path=str(model.onnx_path),
            compile_dir=str(case_root / "prefill_compile"),
            prefill_only=True,
            enable_chunking=True,
            qaic_config=mode_config,
            **prefill_compile_kwargs,
        )
        result["stages"]["prefill_compile"] = {
            "status": "passed",
            "seconds": round(perf_counter() - prefill_compile_start, 3),
            "qpc_path": str(prefill_qpc),
            "qpc_bytes": (Path(prefill_qpc) / "programqpc.bin").stat().st_size,
        }
        _write_json(result_path, result)

        current_stage = "decode_compile_including_export"
        decode_compile_start = perf_counter()
        decode_qpc = model.compile(
            compile_dir=str(case_root / "decode_compile"),
            **_compile_kwargs(args, prefill_seq_len=1),
        )
        result["stages"]["decode_compile_including_export"] = {
            "status": "passed",
            "seconds": round(perf_counter() - decode_compile_start, 3),
            "qpc_path": str(decode_qpc),
        }
        _write_json(result_path, result)

        current_stage = "prefill_inference"
        inputs, prompt_len = _prepare_inputs(tokenizer, args.prompt, args.prefill_seq_len)
        if inputs["input_ids"].shape[1] > args.ctx_len:
            raise ValueError(
                f"Padded prompt length {inputs['input_ids'].shape[1]} exceeds context length {args.ctx_len}"
            )
        prefill_session = QAICInferenceSession(str(prefill_qpc))
        decode_session = QAICInferenceSession(str(decode_qpc))
        try:
            prefill_start = perf_counter()
            prefill_output = prefill_session.run(inputs)
            prefill_seconds = perf_counter() - prefill_start
            logits = prefill_output["logits"]
            logit_position = min(prompt_len - 1, logits.shape[1] - 1)
            selected_logits = logits[:, logit_position, :]
            top_k = np.argsort(selected_logits[0])[-5:][::-1]
            first_token = np.argmax(selected_logits, axis=-1, keepdims=True).astype(np.int64)
            generated = [int(first_token[0, 0])]
            logits_path = case_root / "prefill_logits.npy"
            np.save(logits_path, selected_logits)
            result["stages"]["prefill_inference"] = {
                "status": "passed",
                "seconds": round(prefill_seconds, 3),
                "logits_shape": list(logits.shape),
                "selected_logit_position": int(logit_position),
                "logits_path": str(logits_path),
                "top_token_ids": [int(token_id) for token_id in top_k],
                "top_token_logits": [float(selected_logits[0, token_id]) for token_id in top_k],
            }
            result["generated_token_ids"] = generated.copy()
            _write_json(result_path, result)

            decode_inputs = {
                "input_ids": first_token,
                "position_ids": np.array([[prompt_len]], dtype=np.int64),
            }
            for layer_index in range(config.num_hidden_layers):
                decode_inputs[f"past_key.{layer_index}"] = prefill_output[f"past_key.{layer_index}_RetainedState"]
                decode_inputs[f"past_value.{layer_index}"] = prefill_output[f"past_value.{layer_index}_RetainedState"]

            current_stage = "decode_inference"
            decode_start = perf_counter()
            for _ in range(args.generation_len - 1):
                decode_output = decode_session.run(decode_inputs)
                next_token = np.argmax(decode_output["logits"], axis=-1).astype(np.int64)
                generated.append(int(next_token[0, 0]))
                result["generated_token_ids"] = generated.copy()
                _write_json(result_path, result)
                decode_inputs["input_ids"] = next_token
                decode_inputs["position_ids"] += 1
                for layer_index in range(config.num_hidden_layers):
                    decode_inputs[f"past_key.{layer_index}"] = decode_output[f"past_key.{layer_index}_RetainedState"]
                    decode_inputs[f"past_value.{layer_index}"] = decode_output[
                        f"past_value.{layer_index}_RetainedState"
                    ]
            result["stages"]["decode_inference"] = {
                "status": "passed",
                "seconds": round(perf_counter() - decode_start, 3),
            }
        finally:
            prefill_session.deactivate()
            decode_session.deactivate()

        result["generated_text"] = tokenizer.decode(generated)
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["stages"].setdefault(current_stage, {}).update({"status": "failed", "error": result["error"]})
        stage_order = (
            "model_load",
            "prefill_export",
            "prefill_compile",
            "decode_compile_including_export",
            "prefill_inference",
            "decode_inference",
        )
        for stage in stage_order[stage_order.index(current_stage) + 1 :]:
            result["stages"].setdefault(stage, {"status": "not_reached"})
    finally:
        result["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_json(result_path, result)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if result["status"] == "passed" else 1


def _write_report(run_root, results):
    import numpy as np

    cases = results["cases"]
    baseline = cases.get("prefill_q", {}).get("generated_token_ids")
    baseline_logits_path = cases.get("prefill_q", {}).get("stages", {}).get("prefill_inference", {}).get("logits_path")
    baseline_logits = (
        np.load(baseline_logits_path) if baseline_logits_path and Path(baseline_logits_path).is_file() else None
    )
    rows = [
        "# GPT-OSS Blocking-Mode Parity",
        "",
        f"- Model: `{results['model']}`",
        f"- Configuration: `{results['configuration']}`",
        f"- Prompt: {results['prompt']!r}",
        "- Modes run serially with separate export, prefill-QPC, and decode-QPC directories.",
        "",
        "| Mode | Status | Export (s) | Prefill compile (s) | Decode compile incl. export (s) | Prefill (s) | First token | Max logit delta vs prefill_q | Token IDs match | Generated token IDs | Text | Error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for mode in BLOCKING_MODES:
        case = cases.get(mode, {})
        stages = case.get("stages", {})
        token_ids = case.get("generated_token_ids")
        matches = "—" if not baseline or not token_ids else ("yes" if token_ids == baseline else "no")
        first_token = token_ids[0] if token_ids else "—"
        logits_path = stages.get("prefill_inference", {}).get("logits_path")
        if baseline_logits is None or not logits_path or not Path(logits_path).is_file():
            logit_delta = "—"
        else:
            mode_logits = np.load(logits_path)
            logit_delta = f"{float(np.max(np.abs(mode_logits - baseline_logits))):.8g}"
        cells = [
            f"`{mode}`",
            case.get("status", "pending"),
            str(stages.get("prefill_export", {}).get("seconds", "—")),
            str(stages.get("prefill_compile", {}).get("seconds", "—")),
            str(stages.get("decode_compile_including_export", {}).get("seconds", "—")),
            str(stages.get("prefill_inference", {}).get("seconds", "—")),
            str(first_token),
            logit_delta,
            matches,
            str(token_ids or "—").replace("|", "\\|"),
            str(case.get("generated_text", "—")).replace("|", "\\|").replace("\n", " "),
            str(case.get("error", "—")).replace("|", "\\|").replace("\n", " "),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    (run_root / "report.md").write_text("\n".join(rows) + "\n")


def _run_matrix(args):
    run_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else Path(
            f"/home/sabhis/blocking/qeff_artifacts/gpt_oss_blocking_parity_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
        )
    )
    run_root.mkdir(parents=True, exist_ok=False)
    modes = tuple(BLOCKING_MODES) if args.mode == "all" else (args.mode,)
    results = {
        "model": args.model,
        "prompt": args.prompt,
        "configuration": {
            "prefill_seq_len": args.prefill_seq_len,
            "ctx_len": args.ctx_len,
            "generation_len": args.generation_len,
            "num_devices": args.num_devices,
            "mdp_num_partitions": args.mdp_num_partitions,
            "num_cores": args.num_cores,
            "weight_free": True,
            "mxfp6_matmul": True,
            "mxint8_kv_cache": True,
            "use_onnx_subfunctions": True,
        },
        "cases": {},
    }
    summary_path = run_root / "summary.json"
    _write_json(summary_path, results)
    script_path = Path(__file__).resolve()
    for mode in modes:
        case_root = run_root / mode
        case_root.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(script_path),
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--prefill-seq-len",
            str(args.prefill_seq_len),
            "--ctx-len",
            str(args.ctx_len),
            "--generation-len",
            str(args.generation_len),
            "--num-devices",
            str(args.num_devices),
            "--mdp-num-partitions",
            str(args.mdp_num_partitions),
            "--num-cores",
            str(args.num_cores),
            "--expert-chunk-size",
            str(args.expert_chunk_size),
            "--hf-hub-cache",
            args.hf_hub_cache,
            "--_worker-mode",
            mode,
            "--_case-root",
            str(case_root),
        ]
        environment = os.environ.copy()
        environment["HF_HUB_CACHE"] = args.hf_hub_cache
        environment["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        log_path = case_root / "run.log"
        print(f"\n[{mode}] starting sequential GPT-OSS run", flush=True)
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
        result_path = case_root / "result.json"
        if result_path.is_file():
            results["cases"][mode] = json.loads(result_path.read_text())
        else:
            results["cases"][mode] = {
                "blocking_mode": mode,
                "status": "failed",
                "error": f"Worker exited {return_code} without writing {result_path}",
            }
        _write_json(summary_path, results)
        _write_report(run_root, results)
        print(f"[{mode}] finished with exit code {return_code}", flush=True)
    _write_json(summary_path, results)
    _write_report(run_root, results)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Report: {run_root / 'report.md'}", flush=True)
    return 1 if any(results["cases"].get(mode, {}).get("status") != "passed" for mode in modes) else 0


def main():
    args = _parse_args()
    if args._worker_mode:
        return _run_worker(args)
    if args.prefill_seq_len < 1 or args.ctx_len < args.prefill_seq_len or args.generation_len < 1:
        raise ValueError("Require prefill-seq-len >= 1, ctx-len >= prefill-seq-len, and generation-len >= 1")
    if min(args.num_devices, args.mdp_num_partitions, args.num_cores, args.expert_chunk_size) < 1:
        raise ValueError("Device, partition, core, and expert chunk counts must all be positive")
    return _run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
