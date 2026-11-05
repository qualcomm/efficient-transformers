#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Export and compile a four-layer GLM-5.3 prefill-only graph for disaggregated serving experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from glm53_four_layer_decode_compile_generate import (
    DEFAULT_HF_CACHE,
    DEFAULT_QEFF_HOME,
    MODEL_ID,
    install_partial_fp8_dequant_patch,
)
from transformers import AutoConfig, AutoModelForCausalLM

PREFILL_ATTENTION_PRESETS = {
    "dense_prefill_parallel": {
        "blocking_mode": "prefill_par",
        "num_kv_blocks": 2,
        "par_num_split": 16,
        "mla_absorption": {"absorption": True, "online": False, "cache_compressed": True},
    },
    "dense_prefill_parallel_online": {
        "blocking_mode": "prefill_par_online",
        "num_kv_blocks": 2,
        "par_num_split": 16,
        "mla_absorption": {"absorption": True, "online": True, "cache_compressed": True},
    },
    "dsa_prefill_cp1": {
        "mla_absorption": {"absorption": True, "online": False, "cache_compressed": True},
        "indexer_dp": 1,
        "indexer_cp": 1,
        "indexer_kvp": 1,
        "attn_dp": 1,
        "attn_cp": 1,
        "attn_kvp": 1,
        "indexer_num_blocks": 1,
        "num_cores_per_device": 16,
        "indexer_ql_chunk": 128,
        "indexer_q_block_size": 32,
        "indexer_topk_blocking": 16,
        "indexer_prefill_parallel": True,
        "sparse_q_block_size": 128,
        "sparse_kv_num_blocks": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--qeff-home", default=str(Path(DEFAULT_QEFF_HOME).with_name("glm53_prefill_only")))
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--compile-dir", default=None)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument(
        "--attention-preset", choices=sorted(PREFILL_ATTENTION_PRESETS), default="dense_prefill_parallel"
    )
    parser.add_argument("--attention-qaic-json", default=None, help="JSON object merged over the selected preset.")
    return parser.parse_args()


def validate_prefill_dimensions(qaic_config: dict[str, Any], *, prefill_seq_len: int, ctx_len: int) -> None:
    if prefill_seq_len <= 1:
        raise ValueError("prefill_seq_len must be greater than 1 for a prefill-only graph.")
    if prefill_seq_len > ctx_len:
        raise ValueError("prefill_seq_len must not exceed ctx_len.")
    num_kv_blocks = int(qaic_config.get("num_kv_blocks", 1))
    par_num_split = int(qaic_config.get("par_num_split", 1))
    if num_kv_blocks <= 0 or par_num_split <= 0:
        raise ValueError("num_kv_blocks and par_num_split must be positive.")
    if num_kv_blocks > ctx_len:
        raise ValueError("num_kv_blocks cannot exceed ctx_len.")
    if par_num_split > max(1, ctx_len // num_kv_blocks):
        raise ValueError("par_num_split cannot exceed the dense MLA KV block width.")


def main() -> None:
    args = parse_args()
    if args.num_layers != 4:
        raise ValueError("This GLM-5.3 prefill example currently supports exactly four layers.")
    ctx_len = args.ctx_len if args.ctx_len is not None else args.prefill_seq_len
    qaic_config = copy.deepcopy(PREFILL_ATTENTION_PRESETS[args.attention_preset])
    if args.attention_qaic_json:
        qaic_config.update(json.loads(args.attention_qaic_json))
    validate_prefill_dimensions(qaic_config, prefill_seq_len=args.prefill_seq_len, ctx_len=ctx_len)

    os.environ.setdefault("HF_HUB_CACHE", args.hf_cache)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("QEFF_HOME", args.qeff_home)

    from QEfficient import QEFFAutoModelForCausalLM

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    config = AutoConfig.from_pretrained(args.model_id, cache_dir=args.hf_cache)
    config.num_hidden_layers = args.num_layers
    if args.attention_preset.startswith("dense"):
        config.layer_types = ["full_attention"] * args.num_layers
    else:
        config.layer_types = list(config.layer_types[: args.num_layers])
        config.indexer_types = list(config.indexer_types[: args.num_layers])
    config.use_cache = True
    config.torch_dtype = torch.float32
    config.dtype = torch.float32

    print(
        json.dumps(
            {
                "event": "config",
                "model_id": args.model_id,
                "num_layers": args.num_layers,
                "dtype": "torch.float32",
                "batch_size": args.batch_size,
                "prefill_seq_len": args.prefill_seq_len,
                "ctx_len": ctx_len,
                "qeff_home": os.environ["QEFF_HOME"],
                "attention_preset": args.attention_preset,
                "qaic_config": qaic_config,
                "weight_free": False,
            }
        ),
        flush=True,
    )

    install_partial_fp8_dequant_patch()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        cache_dir=args.hf_cache,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).eval()
    qeff_model = QEFFAutoModelForCausalLM(
        hf_model,
        pretrained_model_name_or_path=args.model_id,
        qaic_config=qaic_config,
    )
    qeff_model.transform(
        ctx_len=ctx_len,
        seq_len=args.prefill_seq_len,
        bs=args.batch_size,
        num_devices=args.num_devices,
        qaic_config=qaic_config,
        num_cores=args.num_cores,
        prefill_only=True,
    )

    export_dir = Path(args.export_dir) if args.export_dir is not None else Path(args.qeff_home) / "prefill_export"
    onnx_path = qeff_model.export(
        export_dir=str(export_dir),
        prefill_only=True,
        prefill_seq_len=args.prefill_seq_len,
        offload_pt_weights=False,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
    )
    print(json.dumps({"event": "export_done", "onnx_path": str(onnx_path)}), flush=True)
    if args.skip_compile:
        print(json.dumps({"event": "skip_compile"}), flush=True)
        return

    compile_dir = Path(args.compile_dir) if args.compile_dir is not None else Path(args.qeff_home) / "prefill_compile"
    qpc_path = qeff_model.compile(
        compile_dir=str(compile_dir),
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=ctx_len,
        batch_size=args.batch_size,
        num_devices=args.num_devices,
        num_cores=args.num_cores,
        prefill_only=True,
        offload_pt_weights=False,
    )
    print(json.dumps({"event": "compile_done", "qpc_path": str(qpc_path)}), flush=True)


if __name__ == "__main__":
    main()
