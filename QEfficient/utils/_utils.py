# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import inspect
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
import yaml
from huggingface_hub import login, snapshot_download
from requests.exceptions import HTTPError
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from QEfficient.utils.constants import KWARGS_INCLUSION_LIST, QEFF_MODELS_DIR, Constants, QnnConstants
from QEfficient.utils.hash_utils import json_serializable
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")

# Retained-state buffer name stems that correspond to per-sequence decoder state vLLM transfers
# between disaggregated prefill and decode workers: classical attention KV (past_key/past_value),
# DeepSeek-V3 MLA (compressed_kv/k_pe), and hybrid linear-attention state (conv_state/recurrent_state
# on Qwen3.5/Qwen3.5-MoE). Vision/multimodal retained buffers (vision_embeds, pixel_values, image_idx,
# deepstack_features, ...) are intentionally excluded — they are not part of the cross-worker payload.
_KV_RETAINED_STEMS = (
    "past_key.",
    "past_value.",
    "compressed_kv.",
    "k_pe.",
    "conv_state.",
    "recurrent_state.",
)
_RETAINED_STATE_SUFFIX = "_RetainedState"
_INTERNAL_RETAINED_STATE_SUFFIX = "_InternalRetainedState"


def validate_kv_cache_prefix(kv_cache_prefix: Optional[str]) -> Optional[str]:
    """
    Validate the optional KV-cache buffer-name prefix.

    The prefix is injected as an infix token into KV retained-state names
    (``past_key.0_RetainedState`` -> ``past_key.0_<prefix>_RetainedState``), so it must be a plain
    alphanumeric token. Disallowing ``.`` and ``_`` keeps the ``past_key.{layer}_{prefix}`` structure
    unambiguous for downstream regex matching.

    Returns the prefix unchanged when valid, or ``None`` when not provided.
    """
    if kv_cache_prefix is None:
        return None
    if not isinstance(kv_cache_prefix, str) or not kv_cache_prefix.isalnum():
        raise ValueError(
            "kv_cache_prefix must be a non-empty alphanumeric string (no '.', '_' or whitespace); "
            f"got {kv_cache_prefix!r}"
        )
    return kv_cache_prefix


def _infix_kv_prefix(name: str, kv_cache_prefix: str) -> str:
    """Insert ``_<prefix>`` before the ``_RetainedState`` suffix for LLM KV-cache buffers only."""
    if not name.endswith(_RETAINED_STATE_SUFFIX):
        return name
    stem = name[: -len(_RETAINED_STATE_SUFFIX)]
    if not any(stem.startswith(kv_stem) for kv_stem in _KV_RETAINED_STEMS):
        return name
    return f"{stem}_{kv_cache_prefix}{_RETAINED_STATE_SUFFIX}"


def apply_kv_cache_prefix(output_names, kv_cache_prefix: Optional[str]):
    """
    Insert an infix token into LLM KV-cache retained-state output names.

    ``past_key.0_RetainedState`` -> ``past_key.0_<prefix>_RetainedState`` (and likewise for
    ``past_value`` / ``compressed_kv`` / ``k_pe``). The matching device input buffer is named by the
    compiler by stripping ``_RetainedState`` (``past_key.0_<prefix>``), so KV retention pairing is
    preserved. Vision/multimodal retained buffers are left untouched.

    Accepts either a flat ``List[str]`` (CausalLM / single-QPC VLM) or the
    ``{"vision": [...], "lang": [...]}`` dict (dual-QPC VLM); for the dict form only the ``lang`` list
    is rewritten. No-op when ``kv_cache_prefix`` is falsy. The input is not mutated in place.
    """
    if not kv_cache_prefix:
        return output_names
    validate_kv_cache_prefix(kv_cache_prefix)

    if isinstance(output_names, dict):
        result = dict(output_names)
        if result.get("lang") is not None:
            result["lang"] = [_infix_kv_prefix(name, kv_cache_prefix) for name in result["lang"]]
        return result
    return [_infix_kv_prefix(name, kv_cache_prefix) for name in output_names]


def align_kv_input_names_to_retained_outputs(input_names, output_names):
    """
    Rename KV-cache *input* buffers so each pairs with its retained-state *output*.

    The AIC compiler retains a KV buffer by matching an output ``X_RetainedState`` to the input named
    ``X`` (suffix stripped). When the retained outputs carry an injected prefix
    (``past_key.0_<prefix>_RetainedState``), the corresponding input must be renamed from
    ``past_key.0`` to ``past_key.0_<prefix>`` for the pairing to hold.

    This derives the rename purely from ``output_names`` (which already carry any prefix), so callers
    that build prefixed outputs do not need to thread the prefix separately. It is a no-op for inputs
    that already match a retained output exactly, and for non-KV inputs. ``input_names`` is not mutated.
    """
    # Stripped target names from retained KV outputs, e.g. {"past_key.0_VLLM", "past_value.0_VLLM"}.
    retained_targets = []
    for name in output_names:
        stem = None
        if name.endswith(_RETAINED_STATE_SUFFIX):
            stem = name[: -len(_RETAINED_STATE_SUFFIX)]
        elif name.endswith(_INTERNAL_RETAINED_STATE_SUFFIX):
            stem = name[: -len(_INTERNAL_RETAINED_STATE_SUFFIX)]
        if stem is None:
            continue
        if any(stem.startswith(kv_stem) for kv_stem in _KV_RETAINED_STEMS):
            retained_targets.append(stem)
    retained_set = set(retained_targets)

    aligned = []
    for name in input_names:
        if not any(name.startswith(stem) for stem in _KV_RETAINED_STEMS) or name in retained_set:
            aligned.append(name)
            continue
        # Find a retained target that is this input with an extra "_<prefix>" infix.
        match = next((t for t in retained_targets if t == name or t.startswith(name + "_")), None)
        aligned.append(match if match is not None else name)
    return aligned


class LRUCache:
    """Simple LRU cache with size limit for vision outputs"""

    def __init__(self, max_size=100):
        self._cache = {}
        self._access_order = []
        self._max_size = max_size

    def get(self, key):
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = value
        self._access_order.append(key)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()


class DownloadRetryLimitExceeded(Exception):
    """
    Used for raising error when hf_download fails to download the model after given max_retries.
    """


def login_and_download_hf_lm(model_name, *args, **kwargs):
    logger.info(f"loading HuggingFace model for {model_name}")
    hf_token = kwargs.pop("hf_token", None)
    cache_dir = kwargs.pop("cache_dir", None)
    if hf_token is not None:
        login(hf_token)
    model_path = hf_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.msgpack", "*.h5", "*.pth"],
    )
    return model_path


def hf_download(
    repo_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_retries: Optional[int] = Constants.MAX_RETRIES,
):
    # Setup cache_dir
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    retry_count = 0
    while retry_count < max_retries:
        try:
            model_path = snapshot_download(
                repo_id,
                cache_dir=cache_dir,
                revision="main",
                resume_download=True,
                token=hf_token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            break
        except requests.ReadTimeout as e:
            logger.info(f"Read timeout: {e}")
            retry_count += 1
        except HTTPError as e:
            if e.response.status_code == 401:
                logger.info("You need to pass a valid `--hf_token=...` to download private checkpoints.")
            raise e
        except OSError as e:
            if "Consistency check failed" in str(e):
                logger.info(
                    "Consistency check failed during model download. The file appears to be incomplete. Resuming the download..."
                )
                retry_count += 1
            else:
                raise e

    if retry_count >= max_retries:
        raise DownloadRetryLimitExceeded(
            f"Unable to download full model after {max_retries} tries. If the model fileS are huge in size, please try again."
        )
    return model_path


def qpc_exists(qpc_dir_path: str) -> bool:
    """
    Checks if qpc dir exists.
    Returns
    1. Boolean variable indicating if qpc files exist
    2. Path of the qpc dir if found.
    ---------

    :model_name: `str` - HF Model card name.
    :dir_path: `str` - Path of qpc directory.

    Return:
        qpc_exists and path to qpc directory
    """

    # Compute the boolean indicating if the QPC exists
    qpc_exists_bool = os.path.isdir(qpc_dir_path) and os.path.isfile(os.path.join(qpc_dir_path, "programqpc.bin"))

    return qpc_exists_bool


def get_onnx_dir_name(model_name, has_fbs):
    # Create a unique directory name for the ONNX model
    # Clearly indicate whether it's with or without FBS
    # Replace all hyphens with underscores
    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    if has_fbs:
        return f"onnx_{model_name_safe}_with_fbs"
    else:
        return f"onnx_{model_name_safe}_without_fbs"


def onnx_exists(model_name: str, full_batch_size: int) -> Tuple[bool, str, str]:
    """
    Checks if qpc files already exists, removes the directory if files have been manipulated.
    ---------

    :model_name: `str`- HF Model card name.

    Return:
        onnx_exists and path to onnx file and directory
    """
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
    os.makedirs(model_card_dir, exist_ok=True)

    # Determine if we're using full_batch_size
    has_fbs = full_batch_size is not None

    # ONNX handling
    onnx_dir_name = get_onnx_dir_name(model_name, has_fbs)
    onnx_dir_path = os.path.join(model_card_dir, onnx_dir_name)
    os.makedirs(onnx_dir_path, exist_ok=True)
    clipped_onnx_model_path = os.path.join(onnx_dir_path, model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")
    unclipped_onnx_model_path = clipped_onnx_model_path.replace("_clipped_fp16.onnx", ".onnx")

    # Compute the boolean indicating if the ONNX model exists
    onnx_exists_bool = False
    onnx_model_path = None
    if os.path.isfile(os.path.join(onnx_dir_path, "custom_io_fp16.yaml")):
        if os.path.isfile(clipped_onnx_model_path):
            onnx_exists_bool = True
            onnx_model_path = clipped_onnx_model_path
        elif os.path.isfile(unclipped_onnx_model_path):
            onnx_exists_bool = True
            onnx_model_path = unclipped_onnx_model_path

    # Return the boolean, onnx_dir_path, and onnx_model_path
    return onnx_exists_bool, onnx_dir_path, onnx_model_path


def load_hf_tokenizer(
    pretrained_model_name_or_path: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    padding_side: str = "right",
    **kwargs,
) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
    # FIXME: Fix kwargs to take token, cache_dir and pass via kwargs only on line 129
    logger.info("Loading Tokenizer")
    if hf_token is not None:
        login(hf_token)
    # Download tokenizer along with model if it doesn't exist
    model_hf_path = (
        pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path)
        else hf_download(
            repo_id=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.py", "*token*", "*.txt"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_hf_path, padding_side=padding_side, trust_remote_code=True, **kwargs
    )
    padding_check_and_fix(tokenizer)  # Check and fix tokenizer viability

    return tokenizer


def load_hf_processor(
    pretrained_model_name_or_path: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
    logger.info("Loading Processor")
    if hf_token is not None:
        login(hf_token)
    # Download tokenizer along with model if it doesn't exist
    model_hf_path = (
        pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path)
        else hf_download(
            repo_id=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.py", "*token*", "*.txt"],
        )
    )
    processor = AutoProcessor.from_pretrained(model_hf_path, trust_remote_code=True, **kwargs)

    return processor


def get_qpc_dir_path(
    model_card_name,
    num_cores,
    mos,
    batch_size,
    prompt_len,
    ctx_len,
    mxfp6,
    mxint8,
    device_group,
    full_batch_size,
    num_speculative_tokens: Optional[int] = None,
    enable_qnn: Optional[bool] = False,
):
    # Create a unique directory name for the QPC model based on all parameters
    qpc_base_dir_name = (
        "qpc"
        + f"{'_qnn_' if enable_qnn else '_'}"
        + f"{num_cores}cores_{batch_size}bs_{prompt_len}pl_{ctx_len}cl_{mos}mos"
        + f"{f'_{full_batch_size}fbs_' if full_batch_size is not None else '_'}"
        + f"{f'_{num_speculative_tokens}nst_' if num_speculative_tokens is not None else ''}"
        + f"{len(device_group) if device_group is not None else 1}"
        + "devices"
        + ("_mxfp6_mxint8" if (mxfp6 and mxint8) else "_mxfp6" if mxfp6 else "_fp16_mxint8" if mxint8 else "_fp16")
    )
    model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_card_name))
    os.makedirs(model_card_dir, exist_ok=True)

    qpc_dir_path = os.path.join(model_card_dir, qpc_base_dir_name, "qpcs")
    return qpc_dir_path


def check_and_assign_cache_dir(local_model_dir, cache_dir):
    if local_model_dir is not None:
        if cache_dir is not None:
            logger.warning(
                f"Both local_model_dir ({local_model_dir}) and cache_dir ({cache_dir}) given. Using local_model_dir."
            )
        return None
    return cache_dir if cache_dir else None


def padding_check_and_fix(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
    """
    Checks and fixes tokenizer padding side and pad_token_id viability.
    --------

    tokenizer: `Union[PreTrainedTokenizer, PreTrainedTokenizerFast]` - Pass model tokenizer to check and fix.
    """
    if tokenizer.padding_side != "right":
        logger.warning(f"Setting tokenizer padding_side to 'right', got {tokenizer.padding_side}")
        tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        if not isinstance(tokenizer.eos_token_id, int):
            raise TypeError("Found tokenizer.eos_token_id to be None, expected int")
        # If Pad token is out of range of vocab size
        if tokenizer.eos_token_id < tokenizer.vocab_size:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = tokenizer.vocab_size - 1


def get_sliding_window_layers(config):
    if hasattr(config, "layer_types") and config.layer_types is not None:
        return torch.tensor([layer_type == "sliding_attention" for layer_type in config.layer_types], dtype=torch.bool)

    pattern = getattr(config, "sliding_window_pattern", 4)
    return torch.tensor([bool((i + 1) % pattern) for i in range(config.num_hidden_layers)], dtype=torch.bool)


def get_sliding_window_shapes(config, batch_size, seq_len):
    """
    Gets padding dims from model config - number of kv heads and d_head
    and returns padding shape - (batch_size, number of kv heads, seq_len, hidden size)
    required for initialization of past_key_values
    --------

    :config: AutoConfig from pretrained model.
    :batch_size: int. number of input prompts used to create inputs
    :seq_len: int. sequence length to run the model for.

    Return:
        List[int, int, int, int]
    """

    if hasattr(config, "n_head"):  # Assuming n_head is a key in the config (GPTs/CodeGen)
        n_heads = config.n_head
        d_head = config.n_embd // config.n_head
    elif hasattr(config, "num_key_value_heads") and hasattr(
        config, "num_attention_heads"
    ):  # Check for num_key_value_heads (Llama/Mistral)
        n_heads = config.num_key_value_heads

        if hasattr(config, "head_dim"):
            d_head = config.head_dim
        else:
            d_head = config.hidden_size // config.num_attention_heads

    elif hasattr(config, "n_heads"):  # Check for n_heads and d_model in the config (MPT Model)
        n_heads = config.n_heads
        d_head = config.d_model // config.n_heads
    elif hasattr(config, "new_decoder_architecture"):  # Check for Falcon
        new_decoder_architecture = getattr(config, "new_decoder_architecture")
        if new_decoder_architecture:  # multi_query is ignored when new_decoder_architecture is True
            n_heads = config.num_attention_heads
        else:
            if hasattr(config, "multi_query"):
                multi_query_value = getattr(config, "multi_query")
                if multi_query_value:
                    n_heads = 1  # MQA , multi query is true
                else:
                    n_heads = config.num_attention_heads
        d_head = config.hidden_size // config.num_attention_heads
    else:
        raise ValueError("Invalid model configuration: n_head/d_heads or num_key_value_heads not found.")

    # TODO needs to fetch the head, d_head and batch size from padding shape
    global_cache_shape = [batch_size, n_heads, seq_len, d_head]
    chunk_seq_len = None
    if hasattr(config, "attention_chunk_size"):
        chunk_seq_len = config.attention_chunk_size
    elif hasattr(config, "sliding_window"):
        chunk_seq_len = config.sliding_window

    # Added the check because in case of mistralai/Mixtral-8x7B-Instruct-v0.1 the sliding window value is set to Null
    if chunk_seq_len is None:
        chunk_seq_len = seq_len

    chunked_cache_shape = [
        batch_size,
        n_heads,
        seq_len if seq_len < chunk_seq_len else chunk_seq_len,
        d_head,
    ]

    return global_cache_shape, chunked_cache_shape


def get_padding_shape_from_config(config, batch_size, seq_len):
    """
    Gets padding dims from model config - number of kv heads and d_head
    and returns padding shape - (batch_size, number of kv heads, seq_len, hidden size)
    required for initialization of past_key_values
    --------

    :config: AutoConfig from pretrained model.
    :batch_size: int. number of input prompts used to create inputs
    :seq_len: int. sequence length to run the model for.

    Return:
        List[int, int, int, int]
    """

    if hasattr(config, "n_head"):  # Assuming n_head is a key in the config (GPTs/CodeGen)
        n_heads = config.n_head
        d_head = config.n_embd // config.n_head
    elif hasattr(config, "num_key_value_heads") and hasattr(
        config, "num_attention_heads"
    ):  # Check for num_key_value_heads (Llama/Mistral)
        n_heads = config.num_key_value_heads

        if hasattr(config, "head_dim") and config.head_dim is not None:
            d_head = config.head_dim
        else:
            d_head = config.hidden_size // config.num_attention_heads

    elif hasattr(config, "n_heads"):  # Check for n_heads and d_model in the config (MPT Model)
        n_heads = config.n_heads
        d_head = config.d_model // config.n_heads
    elif hasattr(config, "new_decoder_architecture"):  # Check for Falcon
        new_decoder_architecture = getattr(config, "new_decoder_architecture")
        if new_decoder_architecture:  # multi_query is ignored when new_decoder_architecture is True
            n_heads = config.num_attention_heads
        else:
            if hasattr(config, "multi_query"):
                multi_query_value = getattr(config, "multi_query")
                if multi_query_value:
                    n_heads = 1  # MQA , multi query is true
                else:
                    n_heads = config.num_attention_heads
        d_head = config.hidden_size // config.num_attention_heads
    else:
        raise ValueError("Invalid model configuration: n_head/d_heads or num_key_value_heads not found.")

    if hasattr(config, "architectures") and config.architectures is not None:  # Check for Starcoder1 - 3D layout
        if "GPTBigCodeForCausalLM" in config.architectures:
            if hasattr(config, "multi_query"):
                multi_query_value = getattr(config, "multi_query")
                if multi_query_value:
                    n_heads = 1  # MQA , multi query is true
                else:
                    n_heads = config.n_head
    padding_shape = [batch_size, n_heads, seq_len, d_head]
    return padding_shape


def get_num_layers_from_config(config):
    """
    Gets number of layers from model config
    --------

    :config: AutoConfig from pretrained model.

    Return:
        number of layers
    """

    if hasattr(config, "n_layer"):  # Assuming n_layer is a key in the config (GPTs/CodeGen)
        n_layer = config.n_layer
    elif hasattr(config, "num_hidden_layers"):  # llama/Mistral/Falcon
        n_layer = config.num_hidden_layers
    elif hasattr(config, "n_layers"):  # Check for n_layers in the config (MPT Model)
        n_layer = config.n_layers
    else:
        raise ValueError("Invalid model configuration: n_layer/n_layers or num_hidden_layers not found.")

    return n_layer


def get_num_layers_vlm(config):
    """
    Gets number of layers from model config of VLM
    --------

    :config: AutoConfig from pretrained model.

    Return:
        number of layers of text and vision part
    """

    if hasattr(config, "llm_config") and hasattr(config, "vision_config"):  # Intern
        n_layers_text = config.llm_config.num_hidden_layers
        n_layers_vision = config.vision_config.num_hidden_layers
    elif hasattr(config, "text_config") and hasattr(config, "vision_config"):  # Llava, Mllama
        n_layers_text = config.text_config.num_hidden_layers
        n_layers_vision = config.vision_config.num_hidden_layers

    return (n_layers_text, n_layers_vision)


def get_padding_shape_vlm(config, ctx_len, batch_size=1):
    """
    Gets padding dims for VLM models- number of kv heads and d_head
    and returns padding shape - (batch_size, number of kv heads, seq_len, hidden size)
    required for initialization of past_key_values
    --------

    :config: AutoConfig from pretrained model.
    :batch_size: int. number of input prompts used to create inputs
    :seq_len: int. sequence length to run the model for.

    Return:
        List[int, int, int, int]
    """
    if hasattr(config, "text_config"):
        n_heads = config.text_config.num_key_value_heads
        d_head = config.text_config.hidden_size // config.text_config.num_attention_heads
        padding_shape = [batch_size, n_heads, ctx_len, d_head]
    elif hasattr(config, "llm_config"):
        n_heads = config.llm_config.num_key_value_heads
        d_head = config.llm_config.hidden_size // config.llm_config.num_attention_heads
        padding_shape = [batch_size, n_heads, ctx_len, d_head]
    return padding_shape


def create_model_params(qeff_model, **kwargs) -> Dict:
    """
    Constructs a dictionary of model parameters.

    Includes core model config, PEFT config (if present), and applied
    transform names, merged with any provided `kwargs`.

    Args:
        qeff_model: The qeff_model instance containing the model and its parameters.
        **kwargs: Arbitrary parameters to include or override.

    Returns:
        Dict: A dictionary containing comprehensive model parameters.
    """
    model_params = copy.deepcopy(kwargs)
    model_params = {k: v for k, v in model_params.items() if k in KWARGS_INCLUSION_LIST}
    model_params["peft_config"] = getattr(qeff_model.model, "active_peft_config", None)
    model_params["applied_transform_names"] = qeff_model._transform_names()
    return model_params


def execute_command(process: str, command: str, output_file_path: Optional[str] = None):
    """
    Executes the give command using subprocess.

    ``Mandatory`` Args:
        :process (str): Process name for which command is executed.
        :command (str): Command to be executed on shell.
    ``Optional`` Args:
        :output_file_path (str): If provided stdout & stderr for the executed command will be dumped to a file. ``Defaults to None.``

    """
    print(f"Running {process} command : \n {command}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
    except Exception as e:
        print("Execution failed: %s", e)

    if result.returncode != 0:
        raise RuntimeError(f"{process} failed Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")
    else:
        if output_file_path:
            stdout_path = os.path.join(output_file_path, f"{process}_stdout.txt")
            stderr_path = os.path.join(output_file_path, f"{process}_stderr.txt")
            # Write the output to a file
            try:
                with open(stdout_path, "w") as file:
                    file.write(result.stdout)
            except Exception as e:
                print(f"Failed to create {stdout_path}: {e}")
            try:
                with open(stderr_path, "w") as file:
                    file.write(result.stderr)
            except Exception as e:
                print(f"Failed to create {stderr_path}: {e}")


def load_yaml(file_path: str) -> Dict[Any, Any]:
    """
    Opens the given YAML file, load and return the Dict.

    ``Mandatory`` Args:
        :file_path (str): YAML File to be opened.

    Return:
        Dict Object from the given file.

    """
    try:
        # Load the YAML config file
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Failed to load YAML object from {file_path}: {e}")
    return config_data


def load_json(file_path: str) -> Dict[Any, Any]:
    """
    Opens the given JSON file, load and return the JSON object.

    ``Mandatory`` Args:
        :file_path (str): JSON File to be opened.

    Return:
        JSON Object from the given file.

    """
    try:
        # Load the JSON config file
        with open(file_path, "r") as file:
            config_data = json.load(file)
    except Exception as e:
        raise ValueError(f"Failed to load json object from {file_path}: {e}")
    return config_data


def create_json(file_path: str, json_data: object):
    """
    Creates a JSON file with provided JSON data.

    ``Mandatory`` Args:
        :file_path (str): JSON File to be created.
        :json_data (object): JSON Data Object to be populated inside the created file.

    """
    try:
        with open(file_path, "w") as file:
            json.dump(
                json_data,
                file,
                skipkeys=False,
                ensure_ascii=True,
                check_circular=True,
                allow_nan=False,
                indent=4,
                separators=(",", ":"),
                default=json_serializable,
                sort_keys=True,
            )
    except Exception as e:
        print(f"Failed to create JSON File {file_path}: {e}")


def model_swap(func):
    def wrapper(*args, **kwargs):
        if "model" in kwargs and kwargs["model"] is not None:
            original_model = args[0].model
            args[0].model = kwargs["model"]
            onnx_path = func(*args, **kwargs)
            args[0].model = original_model
            return onnx_path

    return wrapper


# Ensure input obj is JSON serializable
def make_serializable(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        return make_serializable(vars(obj))
    return str(obj)


@dataclass
class IOInfo:
    name: str
    datatype: torch.dtype
    shape: Tuple[Union[int, str], ...]

    def __repr__(self):
        return f"input_name:{self.name}\tdatatype:{self.datatype}\tshape:{self.shape}"


def dump_qconfig(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        # Skip qconfig dumping when no QPC was actually produced (e.g. the
        # layer-wise export short-circuits compile to return an ONNX path).
        # Without this guard we'd hit a TypeError inside create_and_dump_qconfigs
        # and surface a confusing user-facing message.
        qpc_path = getattr(self, "qpc_path", None)
        if qpc_path is None:
            return result
        try:
            create_and_dump_qconfigs(
                qpc_path,
                self.onnx_path,
                self.get_model_config,
                [cls.__name__ for cls in self._pytorch_transforms],
                [cls.__name__ for cls in self._onnx_transforms],
                kwargs.get("specializations"),
                kwargs.get("mdp_ts_num_devices", 1),
                kwargs.get("num_speculative_tokens"),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in ["specializations", "mdp_ts_num_devices", "num_speculative_tokens", "custom_io", "onnx_path"]
                },
            )
        except Exception as e:
            logger.debug("Skipping qconfig dump: %s", e)
        return result

    return wrapper


def get_qaic_sdk_version(qaic_sdk_xml_path: str) -> Optional[str]:
    """
    Extracts the QAIC SDK version from the given SDK XML file.

    Args:
        qaic_sdk_xml_path (str): Path to the SDK XML file.
    Returns:
        The SDK version as a string if found, otherwise None.
    """
    qaic_sdk_version = None

    # Check and extract version from the given SDK XML file
    if os.path.exists(qaic_sdk_xml_path):
        try:
            tree = ET.parse(qaic_sdk_xml_path)
            root = tree.getroot()
            base_version_element = root.find(".//base_version")
            if base_version_element is not None:
                qaic_sdk_version = base_version_element.text
        except ET.ParseError as e:
            print(f"Error parsing XML file {qaic_sdk_xml_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {qaic_sdk_xml_path}: {e}")

    return qaic_sdk_version


def create_and_dump_qconfigs(
    qpc_path,
    onnx_path,
    huggingface_config,
    pytorch_transforms,
    onnx_transforms,
    specializations,
    mdp_ts_num_devices,
    num_speculative_tokens,
    **compiler_options,
):
    """
    This Method creates a JSON file which contains all the configs for a model.
    Such as huggingface configs, QEff transforms, QAIC sdk version, QNN sdk, compilation dir, qpc dir and
    many other compilation options.
    """
    enable_qnn = compiler_options.get("enable_qnn", False)
    qnn_config_path = compiler_options.get("qnn_config", None)
    qconfig_file_path = os.path.join(os.path.dirname(qpc_path), "qconfig.json")
    onnx_path = str(onnx_path)
    specializations_file_path = str(os.path.join(os.path.dirname(qpc_path), "specializations.json"))
    compile_dir = str(os.path.dirname(qpc_path))

    qconfigs = {
        "huggingface_config": make_serializable(huggingface_config),
        "qpc_config": {
            "QEff_config": {
                "pytorch_transforms": make_serializable(pytorch_transforms),
                "onnx_transforms": make_serializable(onnx_transforms),
                "onnx_path": onnx_path,
            },
            "compiler_config": {
                "enable_qnn": enable_qnn,
                "compile_dir": compile_dir,
                "specializations_file_path": specializations_file_path,
                "specializations": make_serializable(specializations),
                "mdp_ts_num_devices": mdp_ts_num_devices,
                "num_speculative_tokens": num_speculative_tokens,
                **compiler_options,
            },
            "aic_sdk_config": {
                "qaic_apps_version": get_qaic_sdk_version(Constants.SDK_APPS_XML),
                "qaic_platform_version": get_qaic_sdk_version(Constants.SDK_PLATFORM_XML),
            },
        },
    }

    if enable_qnn:
        qnn_sdk_path = os.getenv(QnnConstants.QNN_SDK_PATH_ENV_VAR_NAME)
        if not qnn_sdk_path:
            raise EnvironmentError(
                f"QNN_SDK_PATH {qnn_sdk_path} is not set. Please set {QnnConstants.QNN_SDK_PATH_ENV_VAR_NAME}"
            )
        qnn_sdk_yaml_path = os.path.join(qnn_sdk_path, QnnConstants.QNN_SDK_YAML)
        qnn_sdk_details = load_yaml(
            qnn_sdk_yaml_path
        )  # Extract QNN SDK details from YAML file if the environment variable is set
        qnn_config = {
            "qnn_config_path": qnn_config_path,
        }
        qconfigs["qpc_config"]["qnn_config"] = qnn_config
        if qnn_sdk_details:
            qconfigs["qpc_config"]["qnn_config"].update(qnn_sdk_details)

    create_json(qconfig_file_path, qconfigs)


def filter_kwargs(func, kwargs):
    """
    Filter a dictionary of keyword arguments to only include the valid arguments of a function.
    Args:
        func: The function to check the arguments for.
        kwargs: The dictionary of keyword arguments to filter.
    Returns:
        A new dictionary containing only the valid keyword arguments.
    """
    valid_args = inspect.signature(func).parameters
    return {key: value for key, value in kwargs.items() if key in valid_args}


def custom_format_warning(msg, category, *args, **kwargs):
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    return f"{YELLOW}[Warning]: {msg}{RESET}\n"


def _infer_specialization_name(spec: Dict, index: int, module_name: Optional[str] = None) -> str:
    """
    Infer a human-readable name for a specialization entry.

    Priority order:
    1. ``_graph_name`` key inside ``spec`` — set at the point of creation by
       ``build_prefill_specialization``, ``build_decode_specialization``,
       VLM ``get_specializations``, Whisper ``get_specializations``, and
       ``pipeline_utils`` compile helpers.  This is the authoritative source
       and requires no heuristics.
    2. ``module_name`` argument — used by diffusers pipeline modules where the
       module name itself is the graph name (e.g. ``"text_encoder"``).
    3. ``seq_len`` heuristic — reliable fallback for plain causal LM specs that
       do not carry ``_graph_name`` (e.g. user-supplied raw dicts, legacy paths).
       ``seq_len != 1`` → ``"Prefill"``, ``seq_len == 1`` → ``"Decode"``.
    4. ``encoder_ctx_len`` with no ``seq_len`` → ``"Encoder"`` (simplified
       Whisper-like spec without ``feature_len``).
    5. Legacy ``sequence_length`` with no ``seq_len`` → ``"Embedding"``
       (older BERT-like raw dicts). Current embedding compile paths set
       ``_graph_name="Embedding"`` and still use ``seq_len``.
    6. Generic fallback ``f"Graph_{index}"``.

    Parameters
    ----------
    spec : Dict
        A single flat specialization dictionary (key → value).  May contain
        the reserved ``_graph_name`` key which is consumed here and never
        written to ``symbols``.
    index : int
        Zero-based position in the specializations list, used only for the
        generic fallback name.
    module_name : str, optional
        Explicit graph name hint for diffusers pipeline modules.

    Returns
    -------
    str
        The inferred graph name.
    """
    # 1. Authoritative tag set at creation time — no heuristics needed.
    if "_graph_name" in spec:
        return spec["_graph_name"]

    # 2. Explicit module name (diffusers pipeline path).
    if module_name is not None:
        if "model_type" in spec:
            return f"{module_name}_model_type_{spec['model_type']}"
        return module_name

    # 3-6. Heuristic fallback — only reached for raw/legacy dicts.
    if "seq_len" not in spec:
        if "encoder_ctx_len" in spec:
            return "Encoder"
        # Legacy fallback for older BERT-style raw dicts. Current embedding
        # compile paths set _graph_name="Embedding" and use seq_len.
        if "sequence_length" in spec:
            return "Embedding"
        return f"Graph_{index}"
    seq_len = spec["seq_len"]
    if str(seq_len) == "1":
        return "Decode"
    return "Prefill"


def to_named_specializations(specializations: List[Dict], module_name: Optional[str] = None) -> List[Dict]:
    """
    Convert flat specialization dicts to the nested ``{name, symbols}`` format
    expected by the backend compiler.

    Example output (one entry)::

        {"name": "Prefill", "symbols": {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"}}

    For diffusers pipeline modules pass ``module_name`` so the graph name reflects
    the module (e.g. ``"text_encoder"``, ``"vae_decoder"``, ``"transformer_model_type_1"``).

    Parameters
    ----------
    specializations : List[Dict]
        List of flat specialization dicts (values may be int or str).
    module_name : str, optional
        Pipeline module name forwarded to ``_infer_specialization_name``.

    Returns
    -------
    List[Dict]
        List of ``{"name": str, "symbols": Dict[str, str]}`` dicts.
    """
    result = []
    for index, spec in enumerate(specializations):
        # Idempotent: already in named format, pass through unchanged.
        if set(spec.keys()) == {"name", "symbols"}:
            result.append(spec)
            continue
        name = _infer_specialization_name(spec, index, module_name=module_name)
        # Strip the internal _graph_name tag — it must not appear in symbols.
        symbols = {k: str(v) for k, v in spec.items() if k != "_graph_name"}
        result.append({"name": name, "symbols": symbols})

    # Deduplicate names: if two entries share the same inferred name (e.g. both
    # have seq_len=1 in a raw-dict override path), append a positional suffix so
    # the compiler never sees duplicate graph names.
    seen: Dict[str, int] = {}
    for entry in result:
        name = entry["name"]
        if name in seen:
            # Rename the first occurrence retroactively on its second encounter
            if seen[name] == 1:
                first_idx = next(i for i, e in enumerate(result) if e["name"] == name)
                result[first_idx] = {**result[first_idx], "name": f"{name}_0"}
            entry["name"] = f"{name}_{seen[name]}"
        seen[name] = seen.get(name, 0) + 1

    return result


def get_attr_or_key(obj: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def require_value(value: Any, label: str) -> Any:
    if value is None:
        raise ValueError(f"Missing required {label} to compute blocking configuration.")
    return value
