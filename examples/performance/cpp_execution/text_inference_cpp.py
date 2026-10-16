# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import QEfficient
from QEfficient.cloud.export import get_onnx_path_and_setup_customIO
from QEfficient.generation.text_generation_inference import fix_prompts, get_compilation_dims, get_input_prompts
from QEfficient.utils import check_and_assign_cache_dir, get_qpc_dir_path, load_hf_tokenizer, qpc_exists
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


script_dir = Path(__file__).resolve().parent
so_folder_path = script_dir / "build"

if so_folder_path.is_dir():
    sys.path.append(str(so_folder_path))
    try:
        import InferenceSetIOBuffer  # noqa: E402
    except ImportError:
        logger.error("Error importing InferenceSetIOBuffer Module")
        raise ImportError(
            "Could not import `InfereceSetIoBuffer` executable, Please refer `examples/cpp_execution/README.md` file for building compiling cpp files."
        )
else:
    raise FileNotFoundError(
        "Please follow `examples/cpp_execution/README.md` instructions to first compile the cpp files"
    )


def main(
    model_name: str,
    num_cores: int,
    device_group: Optional[List[int]] = None,
    prompt: Optional[str] = None,  # type: ignore
    prompts_txt_file_path: Optional[str] = None,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    batch_size: int = 1,
    full_batch_size: Optional[int] = None,
    prompt_len: int = 32,
    ctx_len: int = 128,
    generation_len: Optional[int] = None,
    mxfp6: bool = False,
    mxint8: bool = False,
    local_model_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> None:
    """
    1. Check if compiled qpc for given config already exists, if it does jump to execute, else
    2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
    3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
    4. Download HF model -> transform -> export -> compile -> execute
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :num_cores (int): Number of cores to compile model on.
    ``Optional`` Args:
        :device_group (List[int]): Device Ids to be used for compilation. If ``len(device_group) > 1``, multiple Card setup is enabled. ``Defaults to None.``
        :prompt (str): Sample prompt for the model text generation. ``Defaults to None.``
        :prompts_txt_file_path (str): Path to txt file for multiple input prompts. ``Defaults to None.``
        :aic_enable_depth_first (bool): Enables ``DFS`` with default memory size. ``Defaults to False.``
        :mos (int): Effort level to reduce the on-chip memory. ``Defaults to -1.``
        :batch_size (int): Batch size to compile the model for. ``Defaults to 1.``
        :full_batch_size (int): Set full batch size to enable continuous batching mode. ``Default to None``
        :prompt_len (int): Prompt length for the model to compile. ``Defaults to 32.``
        :ctx_len (int): Maximum context length to compile the model. ``Defaults to 128.``
        :generation_len (int): Number of tokens to be generated. ``Defaults to False.``
        :mxfp6 (bool): Enable compilation for MXFP6 precision. ``Defaults to False.``
        :mxint8 (bool): Compress Present/Past KV to ``MXINT8`` using ``CustomIO`` config. ``Defaults to False.``
        :local_model_dir (str): Path to custom model weights and config files. ``Defaults to None.``
        :cache_dir (str): Cache dir where downloaded HuggingFace files are stored. ``Defaults to None.``
        :hf_token (str): HuggingFace login token to access private repos. ``Defaults to None.``

    .. code-block:: bash

        python -m examples.text_inference_from_cpp OPTIONS

    """
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)
    tokenizer = load_hf_tokenizer(
        pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    if full_batch_size is not None:
        raise RuntimeError("Continuous batching will be supported in future, please rerun without continuous batching.")

    qpc_dir_path = get_qpc_dir_path(
        model_name, num_cores, mos, batch_size, prompt_len, ctx_len, mxfp6, mxint8, device_group, full_batch_size
    )
    if qpc_exists(qpc_dir_path):
        logger.info(f"Pre-compiled qpc found at {qpc_dir_path}! Executing with given prompt")
    else:
        # Handle onnx model generation
        onnx_model_path = get_onnx_path_and_setup_customIO(
            model_name, cache_dir, tokenizer, hf_token, local_model_dir, full_batch_size
        )
        _ = QEfficient.compile(
            onnx_path=onnx_model_path,
            qpc_path=os.path.dirname(
                qpc_dir_path
            ),  # We need to pass parent directory of qpc_dir_path, as the compile function handles the qpcs directory creation
            num_cores=num_cores,
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            mxfp6=mxfp6,
            mxint8=mxint8,
            aic_enable_depth_first=aic_enable_depth_first,
            mos=mos,
            device_group=device_group,
            full_batch_size=full_batch_size,
        )

    #########
    # Execute
    #########
    cloud_ai_100_exec_kv_cpp(
        tokenizer=tokenizer,
        qpc_path=qpc_dir_path,
        prompt_len=prompt_len,
        prompt=prompt,
        device_ids=device_group,
        prompts_txt_file_path=prompts_txt_file_path,
        generation_len=generation_len,
        full_batch_size=full_batch_size,
    )


def cloud_ai_100_exec_kv_cpp(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc_path: str,
    prompt_len: int,
    prompt: Optional[List[str]] = None,
    prompts_txt_file_path: Optional[str] = None,
    device_ids: Optional[List[int]] = None,
    generation_len: Optional[int] = None,
    enable_debug_logs: bool = False,
    stream: bool = True,
    full_batch_size: Optional[int] = None,
):
    batch_size, ctx_len, _, _ = get_compilation_dims(qpc_path)
    prompt: List[str] = get_input_prompts(prompt, prompts_txt_file_path)
    prompt = fix_prompts(prompt, batch_size, full_batch_size)

    # ********* CPP Calling ********
    InferenceSetIOBuffer.generatePrompt(
        tokenizer, qpc_path, prompt_len, ctx_len, batch_size, prompt, generation_len, device_ids
    )


def tokenize_for_prefill(prompt, tokenizer):
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    return inputs


def tokenize_for_prefill_with_padded_len(prompt, tokenizer, padded_len):
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    return inputs


# Generating text from generated_ids received from Python
def tokenize_decode_output(tokenizer, generated_ids, prompt):
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for g, p in zip(generated_texts, prompt):
        print("Prompt: ", p)
        print("Generated Text: ", g)
        print()
    return generated_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference command, the model will be downloaded from HF, optimized, compiled, executed on Cloud AI 100"
    )
    parser.add_argument("--model-name", "--model_name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--local-model-dir", "--local_model_dir", required=False, help="Path to custom model weights and config files"
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=None,
        required=False,
        help="Cache dir to store HF Downloads",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1, help="Batch size for text generation")
    parser.add_argument(
        "--prompt-len", "--prompt_len", default=32, type=int, help="Sequence length for text generation."
    )
    parser.add_argument("--ctx-len", "--ctx_len", default=128, type=int, help="Context length for text generation.")
    parser.add_argument(
        "--mxfp6", action="store_true", help="Compress constant MatMul weights to MXFP6 E2M3, default is no compression"
    )
    parser.add_argument(
        "--mxint8",
        action="store_true",
        help="Compress Present/Past KV to MXINT8 using CustomIO config, default is False",
    )
    parser.add_argument(
        "--num_cores", "--num-cores", type=int, required=True, help="Number of cores to compile on Cloud AI 100"
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1]  ",
    )
    parser.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        help="Input prompt, if executing for batch size>1, pass input prompts in single string but separate with pipe (|) symbol",
    )
    parser.add_argument(
        "--prompts_txt_file_path",
        "--prompts-txt-file-path",
        type=str,
        help="File path for taking input prompts from txt file, sample prompts.txt file present in examples/sample_prompts folder",
    )
    parser.add_argument("--generation_len", "--generation-len", type=int, help="Number of tokens to generate")
    parser.add_argument(
        "--aic_enable_depth_first",
        "--aic-enable-depth-first",
        action="store_true",
        help="If passed, this option will be enabled during compilation, disabled by default",
    )
    parser.add_argument(
        "--mos",
        type=int,
        default=-1,
        help="Effort level to reduce the on-chip memory",
    )
    # FIXME: Add verbose feature
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="pass to print info logs",
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
    del args.verbose  # type: ignore
    main(**args.__dict__)
