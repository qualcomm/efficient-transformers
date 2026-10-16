# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Single-prompt DFlash runner for a VLM text or image prompt."""

import argparse
import os
import sys
import time

import requests
from PIL import Image
from rich.console import Console
from rich.markup import escape
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from utils import (
    MODEL_MAP,
    compile_gemma_vlm_dlm_qpc,
    compile_gemma_vlm_qpcs,
    compile_qwen3vl_vlm_dlm_qpc,
    compile_qwen3vl_vlm_qpcs,
    get_spd_prompt_chunk_size,
    load_spd_sessions,
    resolve_model_name,
    validate_spd_decode_specialization,
)

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.dflash_generation import (
    build_input_ids_gemma4,
    build_inputs_gemma4,
    run_spd_inference_gemma4,
    run_spd_inference_qwen3_vl,
    run_vision_encoder_gemma4,
)
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

console = Console()
IMAGE_PROMPT = "Can you Describe this image in detail?"
IMAGE_URL = "https://wallup.net/wp-content/uploads/2017/03/28/351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"
SYSTEM_PROMPT = "You are a helpful assistant."


def parse_device_list(value):
    return [int(device) for device in value.split(",") if device.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model_name",
        default="gemma-4-31B-it",
        type=resolve_model_name,
        help=f"VLM name or full HF repo path. Supported: {', '.join(MODEL_MAP.keys())}",
    )
    parser.add_argument("--tlm_hf_path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--image_url", default=None)
    parser.add_argument("--image_prompt", default=None)
    parser.add_argument("--tlm_qpc", default=None)
    parser.add_argument("--vision_qpc", default=None)
    parser.add_argument("--dlm_qpc", default=None)
    parser.add_argument("--tlm_devices", type=parse_device_list, default=[0, 1, 2, 3])
    parser.add_argument("--dlm_devices", type=parse_device_list, default=[0, 1, 2, 3])
    parser.add_argument("--vision_devices", type=parse_device_list, default=[0, 1, 2, 3])
    parser.add_argument("--tlm_cores", type=int, default=8)
    parser.add_argument("--dlm_cores", type=int, default=8)
    parser.add_argument("--ctx_len", type=int, default=2048)
    parser.add_argument("--prefill_seq_len", type=int, default=128)
    parser.add_argument("--generation_len", type=int, default=1800)
    parser.add_argument("--iteration", type=int, default=300)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Requested batch size. QEfficient currently supports batch_size=1 only; higher values fall back to 1.",
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--precision", choices=["fp16", "fp6"], default="fp6", help="Model precision")
    return parser.parse_args()


def _resolve_batch_size(batch_size):
    if batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if batch_size > 1:
        logger.warning(
            "QEfficient currently does not support batch sizes greater than 1. "
            "Falling back to batch_size=1; use vLLM for higher batch sizes."
        )
        return 1
    return batch_size


def _get_mask_token_id(config):
    dflash_config = getattr(config, "dflash_config", None) or config.to_dict().get("dflash_config", {})
    return dflash_config["mask_token_id"] if isinstance(dflash_config, dict) else dflash_config.mask_token_id


def _run_gemma(
    args,
    tokenizer,
    processor,
    config,
    dlm_session,
    tlm_session,
    prompt_chunk_size,
    vision_qpc,
    batch_size,
):
    input_ids = None
    mm_token_type_ids = None
    vision_embeds = None
    vision_encode_time = None

    if args.image:
        if processor is None:
            raise RuntimeError("--image requires a Gemma processor, which failed to load.")
        if not vision_qpc:
            raise ValueError("--image requires the Gemma vision-encoder QPC.")

        image_url = args.image_url or IMAGE_URL
        user_prompt = args.image_prompt or IMAGE_PROMPT
        image = Image.open(requests.get(image_url, stream=True, timeout=60).raw).convert("RGB")
        input_ids, mm_token_type_ids, processor_inputs = build_inputs_gemma4(
            processor, tokenizer, user_prompt, image=image
        )
        logger.info(f"Image: {image_url}")
        vision_devices = args.vision_devices if args.vision_devices is not None else args.tlm_devices
        vision_session = QAICInferenceSession(vision_qpc, vision_devices)
        vision_start = time.perf_counter()
        vision_embeds = run_vision_encoder_gemma4(vision_session, processor_inputs)
        vision_encode_time = time.perf_counter() - vision_start
        if vision_embeds is None:
            raise RuntimeError("Vision QPC produced no vision_embeds.")
        if "vision_embeds" not in tlm_session.input_names:
            raise ValueError("The supplied Gemma TLM QPC was compiled without the vision path.")
        prompt_text = ""
        logger.info(f"Prompt: {user_prompt} (prompt_len={input_ids.shape[1]})")
    elif processor is not None:
        input_ids = build_input_ids_gemma4(processor, tokenizer, args.prompt, system_prompt=SYSTEM_PROMPT)
        prompt_text = ""
        logger.info(f"Input: {args.prompt[:120].strip()} (prompt_len={input_ids.shape[1]})")
    else:
        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        logger.info(f"Input: {args.prompt[:120].strip()}")

    metrics = run_spd_inference_gemma4(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        mask_token_id=_get_mask_token_id(config),
        vocab_size=config.vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=args.ctx_len,
        block_size=config.block_size,
        max_iterations=args.iteration,
        hidden_size=config.hidden_size,
        generation_len=args.generation_len,
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        vision_embeds=vision_embeds,
        batch_size=batch_size,
    )
    return metrics, {"vision_encode_time_s": vision_encode_time}


def _run_qwen3_vl(
    args,
    tokenizer,
    processor,
    tlm_config,
    config,
    dlm_session,
    tlm_session,
    prompt_chunk_size,
    vision_qpc,
    batch_size,
):
    if processor is None:
        raise RuntimeError("Qwen3-VL requires a processor, which failed to load.")

    image = None
    vision_session = None
    if args.image:
        if not vision_qpc:
            raise ValueError("--image requires the Qwen3-VL vision-encoder QPC.")
        image_url = args.image_url or IMAGE_URL
        logger.info(f"Image: {image_url}")
        image = Image.open(requests.get(image_url, stream=True, timeout=60).raw).convert("RGB")
        vision_devices = args.vision_devices if args.vision_devices is not None else args.tlm_devices
        vision_session = QAICInferenceSession(vision_qpc, vision_devices)
        prompt_text = args.image_prompt or IMAGE_PROMPT
    else:
        prompt_text = args.prompt
    logger.info(f"Prompt: {prompt_text[:120].strip()}")

    metrics = run_spd_inference_qwen3_vl(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        processor=processor,
        tlm_config=tlm_config,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        mask_token_id=_get_mask_token_id(config),
        vocab_size=config.vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=args.ctx_len,
        block_size=config.block_size,
        max_iterations=args.iteration,
        hidden_size=config.hidden_size,
        generation_len=args.generation_len,
        image=image,
        vision_session=vision_session,
        compiled_height=args.height if args.height is not None else 354,
        compiled_width=args.width if args.width is not None else 536,
        batch_size=batch_size,
    )
    return metrics, {"vision_encode_time_s": metrics.vision_prefill_time if args.image else None}


def main():
    args = parse_args()
    batch_size = _resolve_batch_size(args.batch_size)
    if not args.image and not args.prompt:
        raise ValueError("Provide --prompt for text mode, or --image for image mode.")

    tlm_repo_default, dlm_repo = MODEL_MAP[args.model_name]
    tlm_repo = args.tlm_hf_path or tlm_repo_default
    if tlm_repo is None:
        raise ValueError(f"No default TLM HF path for '{args.model_name}'. Pass --tlm_hf_path.")

    tlm_config = AutoConfig.from_pretrained(tlm_repo, trust_remote_code=True, token=args.hf_token)
    is_qwen3vl = tlm_config.model_type == "qwen3_vl"

    if args.tlm_qpc and args.vision_qpc:
        tlm_qpc, vision_qpc = args.tlm_qpc, args.vision_qpc
    else:
        if args.tlm_qpc or args.vision_qpc:
            logger.info("Both --tlm_qpc and --vision_qpc are required to skip the VLM build; rebuilding both.")
        compile_vlm = compile_qwen3vl_vlm_qpcs if is_qwen3vl else compile_gemma_vlm_qpcs
        compile_kwargs = {
            "prefill_seq_len": args.prefill_seq_len,
            "ctx_len": args.ctx_len,
            "num_cores": args.tlm_cores,
            "num_devices": len(args.tlm_devices),
            "hf_token": args.hf_token,
        }
        if is_qwen3vl:
            compile_kwargs.update({"height": args.height, "width": args.width})
        tlm_qpc, vision_qpc = compile_vlm(tlm_repo, dlm_repo, **compile_kwargs)

    if args.dlm_qpc:
        dlm_qpc = args.dlm_qpc
    else:
        compile_dlm = compile_qwen3vl_vlm_dlm_qpc if is_qwen3vl else compile_gemma_vlm_dlm_qpc
        dlm_qpc = compile_dlm(
            tlm_repo,
            dlm_repo,
            ctx_len=args.ctx_len,
            num_cores=args.dlm_cores,
            num_devices=len(args.dlm_devices),
            hf_token=args.hf_token,
        )

    logger.info(f"TLM lang qpc: {tlm_qpc}")
    logger.info(f"Vision qpc: {vision_qpc}")
    logger.info(f"DLM qpc: {dlm_qpc}")

    tokenizer = AutoTokenizer.from_pretrained(tlm_repo, token=args.hf_token, trust_remote_code=True)
    config = AutoConfig.from_pretrained(dlm_repo, token=args.hf_token, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        processor = AutoProcessor.from_pretrained(tlm_repo, token=args.hf_token, trust_remote_code=True)
    except Exception as error:  # noqa: BLE001
        processor = None
        logger.warning(f"No processor for {tlm_repo} ({error}); using tokenizer template.")

    dlm_session, tlm_session = load_spd_sessions(tlm_qpc, dlm_qpc, args.tlm_devices, args.dlm_devices)
    prompt_chunk_size = get_spd_prompt_chunk_size(tlm_session)
    validate_spd_decode_specialization(tlm_session, config.block_size)

    if is_qwen3vl:
        metrics, output_extra = _run_qwen3_vl(
            args,
            tokenizer,
            processor,
            tlm_config,
            config,
            dlm_session,
            tlm_session,
            prompt_chunk_size,
            vision_qpc,
            batch_size,
        )
    else:
        metrics, output_extra = _run_gemma(
            args,
            tokenizer,
            processor,
            config,
            dlm_session,
            tlm_session,
            prompt_chunk_size,
            vision_qpc,
            batch_size,
        )

    output_parts = ["Output: "]
    for token_id, source in zip(metrics.generated_ids, metrics.generated_sources):
        text = escape(tokenizer.decode([token_id], skip_special_tokens=True))
        output_parts.append(f"[blue]{text}[/blue]" if source == "dlm" else f"[white]{text}[/white]")
    console.print("".join(output_parts))

    width = 46
    print("\n" + "=" * width)
    print("  SPD Inference — Metrics")
    print("=" * width)
    print(f"  {'Acceptance Rate (tok/iter)':<30} {metrics.acceptance_rate():>6.2f}")
    print(f"  {'DLM Throughput  (tok/s)':<30} {metrics.dlm_tok_rate():>6.1f}")
    print(f"  {'TLM Throughput  (tok/s)':<30} {metrics.tlm_tok_rate():>6.1f}")
    print(f"  {'SPD Decode Speed (tok/s)':<30} {metrics.spd_tok_rate():>6.1f}")
    print(f"  {'Generated tokens':<30} {metrics.total_generated_tokens:>6}")
    print(f"  {'Iterations':<30} {metrics.num_total_iters:>6}")
    print(f"  {'Prefill time (s)':<30} {metrics.total_prefill_time:>6.3f}")
    if output_extra["vision_encode_time_s"] is not None:
        print(f"  {'Vision Encode (s)':<30} {output_extra['vision_encode_time_s']:>6.3f}")
    print("=" * width + "\n")


if __name__ == "__main__":
    main()
