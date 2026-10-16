# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import logging
import sys
import warnings
from typing import List, Optional

import requests
from PIL import Image
from transformers import PreTrainedModel, TextStreamer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from QEfficient.base.common import QEFFCommonLoader
from QEfficient.cloud import warn_deprecated_cloud_api
from QEfficient.utils import check_and_assign_cache_dir, load_hf_processor, load_hf_tokenizer
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


# TODO: Remove after adding support for VLM's compile and execute
def execute_vlm_model(
    qeff_model: PreTrainedModel,
    model_name: str,
    image_url: str,
    image_path: str,
    prompt: Optional[str] = None,  # type: ignore
    device_group: Optional[List[int]] = None,
    local_model_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    generation_len: Optional[int] = None,
):
    """
    Generate output from a compiled Vision-Language Model (VLM) on Cloud AI 100 hardware.

    This method takes a QEfficient VLM model, processes image and text inputs, and generates
    text outputs using the compiled QPC.

    Parameters
    ----------
    qeff_model : PreTrainedModel
        QEfficient model object, expected to be an instance capable of VLM operations.
    model_name : str
        Hugging Face Model Card name (e.g., ``llava-hf/llava-1.5-7b-hf``) used for loading processor.
    image_url : str
        URL of the image to be used for inference.
    image_path : str
        Local file path to the image to be used for inference.

    Other Parameters
    ----------------
    prompt : str, optional
        Sample prompt for the model text generation. Default is None.
    device_group : List[int], optional
        List of device IDs to be used for inference. If ``len(device_group) > 1``,
        multiple card setup is enabled. Default is None.
    local_model_dir : str, optional
        Path to custom model weights and config files, used if not loading from Hugging Face Hub. Default is None.
    cache_dir : str, optional
        Cache directory where downloaded HuggingFace files are stored. Default is None.
    hf_token : str, optional
        HuggingFace login token to access private repositories. Default is None.
    generation_len : int, optional
        Maximum number of tokens to be generated. Default is None.

    Returns
    -------
    dict
        Output from the ``AI_100`` runtime, typically containing generated text and performance metrics.

    Raises
    ------
    ValueError
        If neither ``image_url`` nor ``image_path`` is provided.
    """
    warn_deprecated_cloud_api("infer")
    if device_group is not None:
        warnings.warn(
            "device_group is deprecated and will be renamed to device_ids in the next release.",
            FutureWarning,
            stacklevel=2,
        )
    if not (image_url or image_path):
        raise ValueError('Neither Image URL nor Image Path is found, either provide "image_url" or "image_path"')
    raw_image = Image.open(requests.get(image_url, stream=True).raw) if image_url else Image.open(image_path)

    processor = load_hf_processor(
        pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    # Added for QEff version 1.20 supported VLM models (mllama and llava)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt[0]},
            ],
        }
    ]

    # Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token ids.
    input_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    split_inputs = processor(
        text=input_text,
        images=raw_image,
        return_tensors="pt",
        add_special_tokens=False,
    )
    streamer = TextStreamer(processor.tokenizer)
    output = qeff_model.generate(
        inputs=split_inputs,
        streamer=streamer,
        device_ids=device_group,
        generation_len=generation_len,
    )
    return output


def main(
    model_name: str,
    num_cores: int,
    device_group: Optional[List[int]] = None,
    prompt: Optional[str] = None,  # type: ignore
    prompts_txt_file_path: Optional[str] = None,
    aic_enable_depth_first: bool = False,
    mos: Optional[int] = 1,
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
    allow_mxint8_mdp_io: bool = False,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    trust_remote_code: Optional[bool] = False,
    ccl_enabled: Optional[bool] = False,
    use_onnx_subfunctions: bool = False,
    dynamo: bool = False,
    **kwargs,
) -> None:
    """
    Main entry point for the QEfficient inference script.

    This function handles the end-to-end process of downloading, optimizing,
    compiling, and executing a HuggingFace model on Cloud AI 100 hardware.
    The process follows these steps:
    
    1. Checks for an existing compiled QPC package. If found, it jumps directly to execution.
    2. Checks for an existing exported ONNX file. If true, it proceeds to compilation then execution.
    3. Checks if the HuggingFace model exists in the cache. If true, it performs model transformation, ONNX export, compilation, and then execution.
    4. If none of the above, it downloads the HuggingFace model, then performs transformation, ONNX export, compilation, and execution.

    Parameters
    ----------
    model_name : str
        Hugging Face Model Card name (e.g., ``gpt2``) or path to a local model.
    num_cores : int
        Number of cores to compile the model on.

    Other Parameters
    ----------------
    device_group : List[int], optional
        List of device IDs to be used for compilation and inference. If ``len(device_group) > 1``,
        a multiple card setup is enabled. Default is None.
    prompt : str, optional
        Sample prompt(s) for the model text generation. For batch size > 1,
        pass multiple prompts separated by a pipe (``|``) symbol. Default is None.
    prompts_txt_file_path : str, optional
        Path to a text file containing multiple input prompts, one per line. Default is None.
    aic_enable_depth_first : bool, optional
        Enables Depth-First Search (DFS) with default memory size during compilation. Default is False.
    mos : int, optional
        Effort level to reduce on-chip memory. Default is 1.
    batch_size : int, optional
        Batch size to compile the model for. Default is 1.
    full_batch_size : int, optional
        Sets the full batch size to enable continuous batching mode. Default is None.
    prompt_len : int, optional
        Prompt length for the model to compile. Default is 32.
    ctx_len : int, optional
        Maximum context length to compile the model for. Default is 128.
    generation_len : int, optional
        Maximum number of tokens to be generated during inference. Default is None.
    mxfp6 : bool, optional
        Enables compilation for MXFP6 precision for constant MatMul weights. Default is False.
        A warning is issued as ``--mxfp6`` is deprecated; use ``--mxfp6-matmul`` instead.
    mxint8 : bool, optional
        Compresses Present/Past KV to ``MXINT8`` using ``CustomIO`` config. Default is False.
        A warning is issued as ``--mxint8`` is deprecated; use ``--mxint8-kv-cache`` instead.
    local_model_dir : str, optional
        Path to custom model weights and config files. Default is None.
    cache_dir : str, optional
        Cache directory where downloaded HuggingFace files are stored. Default is None.
    hf_token : str, optional
        HuggingFace login token to access private repositories. Default is None.
    allow_mxint8_mdp_io : bool, optional
        Allows MXINT8 compression of MDP IO traffic during compilation. Default is False.
    enable_qnn : bool or str, optional
        Enables QNN compilation. Can be passed as a flag (True) or with a configuration file path (str).
        If a string path is provided, it's treated as ``qnn_config``. Default is False.
    qnn_config : str, optional
        Path of the QNN Config parameters file. Default is None.
    trust_remote_code : bool, optional
        If True, trusts remote code when loading models from HuggingFace. Default is False.
    use_onnx_subfunctions : bool, optional
        Enables ONNX subfunctions during export and compile. Default is False.
    dynamo : bool, optional
        Use the dynamo-based ONNX exporter instead of the legacy TorchScript exporter. Default is False.
    **kwargs :
        Additional compiler options passed directly to `qaic-compile`. Any flag supported by
        `qaic-compile` can be passed. Parameters are converted to flags as follows:

        - ``-allocator_dealloc_delay=1`` -> ``-allocator-dealloc-delay=1``
        - ``-qpc_crc=True`` -> ``-qpc-crc``

    Example
    -------
    To run inference from the command line:

    .. code-block:: bash

        python -m QEfficient.cloud.infer --model-name gpt2 --num-cores 16 --prompt "Hello world"

    For advanced compilation options:

    .. code-block:: bash

        python -m QEfficient.cloud.infer --model-name meta-llama/Llama-3.2-11B-Vision-Instruct \\
            --num-cores 16 --prompt "Describe this image." --image-url "https://example.com/image.jpg" \\
            --ctx-len 512 --img-size 560 --mxfp6-matmul

    """
    warn_deprecated_cloud_api("infer")
    if device_group is not None:
        warnings.warn(
            "device_group is deprecated and will be renamed to device_ids in the next release.",
            FutureWarning,
            stacklevel=2,
        )
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)

    if "--mxfp6" in sys.argv and mxfp6:
        logger.warning("mxfp6 is going to be deprecated in a future release, use -mxfp6_matmul instead.")
    if "--mxint8" in sys.argv and mxint8:
        logger.warning("mxint8 is going to be deprecated in a future release, use -mxint8_kv_cache instead.")

    qaic_config = {"ccl_enabled": True} if ccl_enabled else None

    qeff_model = QEFFCommonLoader.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir,
        hf_token=hf_token,
        full_batch_size=full_batch_size,
        local_model_dir=local_model_dir,
        trust_remote_code=trust_remote_code,
        qaic_config=qaic_config,
    )

    image_path = kwargs.pop("image_path", None)
    image_url = kwargs.pop("image_url", None)
    iteration = kwargs.pop("iteration", 1)
    automation = kwargs.pop("automation", False)

    config = qeff_model.model.config
    architecture = config.architectures[0] if config.architectures else None

    if architecture not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values() and (
        kwargs.pop("img_size", None) or image_path or image_url
    ):
        logger.warning(f"Skipping image arguments as they are not valid for {architecture}")

    #########
    # Compile
    #########
    _ = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        mxfp6_matmul=mxfp6,
        aic_enable_depth_first=aic_enable_depth_first,
        batch_size=batch_size,
        mos=mos,
        mxint8_kv_cache=mxint8,
        num_devices=(0 if device_group is None else len(device_group)),
        full_batch_size=full_batch_size,
        allow_mxint8_mdp_io=allow_mxint8_mdp_io,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
        use_onnx_subfunctions=use_onnx_subfunctions,
        dynamo=dynamo,
        **kwargs,
    )

    #  If the io-encrypt flag is passed we will exit after QPC generation.
    if kwargs.get("io_encrypt", None):
        exit()

    #########
    # Execute
    #########
    if architecture in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values():
        exec_info = execute_vlm_model(
            qeff_model=qeff_model,
            model_name=model_name,
            prompt=prompt,
            image_url=image_url,
            image_path=image_path,
            device_group=device_group,
            local_model_dir=local_model_dir,
            cache_dir=cache_dir,
            hf_token=hf_token,
            generation_len=generation_len,
        )
        print(exec_info)
    else:
        tokenizer = load_hf_tokenizer(
            pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
            cache_dir=cache_dir,
            hf_token=hf_token,
        )
        _ = qeff_model.generate(
            tokenizer,
            prompts=prompt,
            device_ids=device_group,
            prompts_txt_file_path=prompts_txt_file_path,
            generation_len=generation_len,
            iteration=iteration,
            automation=automation,
        )


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
        "--comp-ctx-lengths-prefill",
        type=lambda comp_ctx_lengths_prefill: [int(x) for x in comp_ctx_lengths_prefill.split(",")],
        default=None,
        help="Define ccl list in csv format (e.g.,--comp-ctx-lengths 512,1024,2048).",
    )
    parser.add_argument(
        "--comp-ctx-lengths-decode",
        type=lambda comp_ctx_lengths_decode: [int(x) for x in comp_ctx_lengths_decode.split(",")],
        default=None,
        help="Define ccl list in csv format (e.g.,--comp-ctx-lengths 512,1024,2048).",
    )
    parser.add_argument(
        "--ccl_enabled",
        "--ccl-enabled",
        action="store_true",
        help="If passed, ccl feature will be activated",
    )
    parser.add_argument(
        "--mxfp6",
        "--mxfp6_matmul",
        "--mxfp6-matmul",
        action="store_true",
        help="Compress constant MatMul weights to MXFP6 E2M3, default is no compression",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Enable trusting remote code when loading models. Default is False; set to True by passing this flag.",
    )
    parser.add_argument(
        "--mxint8",
        "--mxint8_kv_cache",
        "--mxint8-kv-cache",
        action="store_true",
        help="Compress Present/Past KV to MXINT8 using CustomIO config, default is False",
    )
    parser.add_argument(
        "--use-onnx-subfunctions",
        "--use_onnx_subfunctions",
        dest="use_onnx_subfunctions",
        action="store_true",
        default=False,
        help="Enable ONNX subfunctions during export/compile.",
    )
    parser.add_argument(
        "--dynamo",
        dest="dynamo",
        action="store_true",
        default=False,
        help="Use the dynamo-based ONNX exporter instead of the legacy TorchScript exporter.",
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
        default=1,
        help="Effort level to reduce the on-chip memory",
    )
    # FIXME: Add verbose feature
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="pass to print info logs",
    )
    parser.add_argument(
        "--full_batch_size",
        "--full-batch-size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )
    parser.add_argument(
        "--allow-mxint8-mdp-io",
        "--allow_mxint8_mdp_io",
        action="store_true",
        help="If passed, this option allows MXINT8 compression of MDP IO traffic",
    )
    parser.add_argument(
        "--enable_qnn",
        "--enable-qnn",
        nargs="?",
        const=True,
        type=str,
        default=False,
        help="Enables QNN. Optionally, a configuration file can be provided with [--enable_qnn CONFIG_FILE].\
             If not provided, the default configuration will be used.\
             Sample Config: QEfficient/compile/qnn_config.json",
    )

    args, compiler_options = parser.parse_known_args()

    if isinstance(args.enable_qnn, str):
        args.qnn_config = args.enable_qnn
        args.enable_qnn = True

    compiler_options_dict = {}
    for i in range(0, len(compiler_options)):
        if compiler_options[i].startswith("--"):
            key = compiler_options[i].lstrip("-").replace("-", "_")
            value = (
                compiler_options[i + 1]
                if i + 1 < len(compiler_options) and not compiler_options[i + 1].startswith("-")
                else True
            )
            compiler_options_dict[key] = value
    if args.verbose:
        logger.setLevel(logging.INFO)
    del args.verbose  # type: ignore
    main(**args.__dict__, **compiler_options_dict)
