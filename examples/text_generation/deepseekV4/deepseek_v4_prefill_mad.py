# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compare saved Hugging Face and QAIC DeepSeek V4 prefill logits."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-logits", type=Path, required=True, help="NumPy file containing HF final-token logits.")
    parser.add_argument(
        "--qaic-logits", type=Path, required=True, help="NumPy file containing QAIC final-token logits."
    )
    parser.add_argument("--report", type=Path, help="Optional path to write the JSON report.")
    return parser.parse_args()


def load_logits(path: Path) -> np.ndarray:
    """Load logits as float32 and flatten batch/sequence dimensions."""
    values = np.load(path, allow_pickle=False)
    if values.dtype.kind not in "fiu":
        raise TypeError(f"Expected numeric logits in {path}, got dtype={values.dtype}.")
    values = values.astype(np.float32, copy=False).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Logits file is empty: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Logits contain NaN or infinity: {path}")
    return values


def compare_logits(hf_logits: np.ndarray, qaic_logits: np.ndarray) -> dict[str, object]:
    if hf_logits.shape != qaic_logits.shape:
        raise ValueError(f"Logit sizes differ: HF has {hf_logits.size}, QAIC has {qaic_logits.size} values.")

    absolute_difference = np.abs(hf_logits.astype(np.float64) - qaic_logits.astype(np.float64))
    hf_top1_token_id = int(np.argmax(hf_logits))
    qaic_top1_token_id = int(np.argmax(qaic_logits))
    return {
        "mean_absolute_difference": float(np.mean(absolute_difference, dtype=np.float64)),
        "max_absolute_difference": float(np.max(absolute_difference)),
        "hf_top1_token_id": hf_top1_token_id,
        "qaic_top1_token_id": qaic_top1_token_id,
        "token_match": hf_top1_token_id == qaic_top1_token_id,
        "logit_count": int(hf_logits.size),
    }


def main() -> None:
    args = parse_args()
    hf_logits = load_logits(args.hf_logits)
    qaic_logits = load_logits(args.qaic_logits)
    report = {
        "comparison": "HF vs QAIC final-token prefill logits",
        "hf_logits_path": str(args.hf_logits),
        "qaic_logits_path": str(args.qaic_logits),
        **compare_logits(hf_logits, qaic_logits),
    }
    report_json = json.dumps(report, indent=2)
    print(report_json)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
