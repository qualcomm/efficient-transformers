#!/usr/bin/env python3
"""Create the local four-layer, weight-free Qwen3.8-Flash checkpoint window.

The source snapshot is read shard-by-shard. Host-side PLE embedding tables are
excluded because the four-layer decode path supplies ``ngram_embeddings`` at
runtime instead of loading the table into the checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Local Qwen3.8-Flash-Next snapshot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qwen3.8_flash_experiment/example_repro_20261013/qwen38_flash_4layer_checkpoint_v2"),
        help="Output checkpoint directory",
    )
    parser.add_argument("--layers", type=int, default=4, help="Number of leading language-model layers to retain")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen4_exp_decode_parity import _prune_weight_free_checkpoint

    snapshot = args.snapshot.resolve()
    output = args.output.resolve()
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Qwen3.8 snapshot config not found: {snapshot / 'config.json'}")
    if not (snapshot / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Safetensors index not found: {snapshot / 'model.safetensors.index.json'}")
    if args.layers != 4:
        raise ValueError("The Qwen3.8-Flash QSA example currently supports exactly 4 layers")

    checkpoint = _prune_weight_free_checkpoint(snapshot, args.layers, output)
    print(f"four_layer_checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
