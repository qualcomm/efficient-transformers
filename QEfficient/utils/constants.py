# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from dataclasses import dataclass

from QEfficient.utils.hardware_utils import (
    get_default_aic_hw_version,
    get_default_num_cores,
    get_default_vtcm_size_threshold,
)

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
QEFF_DIR = os.path.dirname(UTILS_DIR)
ROOT_DIR = os.path.dirname(QEFF_DIR)
QEFF_CACHE_DIR_NAME = "qeff_cache"

ONNX_EXPORT_EXAMPLE_BATCH_SIZE = 1
ONNX_EXPORT_EXAMPLE_SEQ_LEN = 32
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
ONNX_EXPORT_EXAMPLE_FBS = 4
ONNX_EXPORT_EXAMPLE_NLK = 2  # Number of Logits to Keep
ONNX_EXPORT_MAX_NUM_IMAGES = 1
ONNX_EXPORT_MAX_IMAGE_TILES = 4
ONNX_EXPORT_IMAGE_WIDTH = 560
ONNX_EXPORT_IMAGE_LENGHT = 560
ONNX_EXPORT_IMAGE_DEPTH = 3
ONNX_EXPORT_CTX_LEN = 1024
DYNAMO_DIM_MAX_BATCH_SIZE = 1024
DYNAMO_DIM_MIN_COMP_CTX_LENGTHS = 4

DEFAULT_AIC_HW_VERSION = get_default_aic_hw_version()

NPI_MAPPING = {
    "google/gemma-3-4b-it": os.path.join(
        QEFF_DIR, "transformers", "models", "gemma3", "configs", "fp32_nodes_gemma3_4b.yaml"
    ),
    "tiny-random/gemma-3": os.path.join(
        QEFF_DIR, "transformers", "models", "gemma3", "configs", "fp32_nodes_gemma3_4b.yaml"
    ),
    "google/gemma-3-27b-it": os.path.join(
        QEFF_DIR, "transformers", "models", "gemma3", "configs", "gemma_updated_npi.yaml"
    ),
}

# Blocking defaults
VTCM_SIZE_THRESHOLD = get_default_vtcm_size_threshold(DEFAULT_AIC_HW_VERSION)

# Compiler defaults
DEFAULT_AIC_NUM_CORES = get_default_num_cores(DEFAULT_AIC_HW_VERSION)
DEFAULT_AIC_MXPF6_MATMUL = False
# Hashing defaults
HASH_HEXDIGEST_STR_LEN = 16
KWARGS_INCLUSION_LIST = [
    "state_dict",
    "revision",
    "key_mapping",
    "commit_hash",
    "adapter_kwargs",
    "adapter_name",
    "gguf_file",
    "pretrained_model_name_or_path",
    "attn_implementation",
    "_attn_implementation",
    "qaic_config",
]

# Minimum value for causal mask
MIN_MASKED_ATTENTION_VALUE = float("-inf")
HEADPAR_MASKED_ATTENTION_VALUE = -3.0e4


# Store the qeff_models inside the ~/.cache directory or over-ride with an env variable.
def get_models_dir():
    """
    Determine the directory for storing QEFF models.
    Priority:
    1. Use $XDG_CACHE_HOME/qeff_models if XDG_CACHE_HOME is set.
    2. Use QEFF_HOME if set in environment.
    3. Default to ~/.cache/qeff_models.
    Sets QEFF_MODELS_DIR environment variable if not already set.
    Returns:
        str: Path to the QEFF models directory.
    """
    qeff_cache_home = os.environ.get("QEFF_HOME")
    # Check if XDG_CACHE_HOME is set
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if qeff_cache_home:
        qeff_models_dir = os.path.join(qeff_cache_home, QEFF_CACHE_DIR_NAME)
    # Check if QEFF_MODELS_DIR is set
    elif xdg_cache_home:
        qeff_models_dir = os.path.join(xdg_cache_home, QEFF_CACHE_DIR_NAME)
    else:
        # Use ~/.cache/qeff_models as the default
        qeff_models_dir = os.path.join(os.path.expanduser("~"), ".cache", QEFF_CACHE_DIR_NAME)

    # Set QEFF_MODELS_DIR environment variable
    return qeff_models_dir


QEFF_MODELS_DIR = get_models_dir()

ONNX_EXPORT_EXAMPLE_REPETITION_PENALTIES = 0.5
ONNX_EXPORT_EXAMPLE_PRESENCE_PENALTIES = 0.5
ONNX_EXPORT_EXAMPLE_TEMPERATURES = 0.80
ONNX_EXPORT_EXAMPLE_MAX_TOP_K_IDS = 512
ONNX_EXPORT_EXAMPLE_TOP_PS = 0.80
ONNX_EXPORT_EXAMPLE_MIN_PS = 0.99
ONNX_LEGACY_EXPORT_OPSET = 17
ONNX_DYNAMO_EXPORT_OPSET = 18
ONNX_EXPORT_OPSET = ONNX_LEGACY_EXPORT_OPSET
FILE_CHUNK_SIZE_DEFAULT = 10 * 2**30  # 10 GB
SIZE_THRESHOLD_DEFAULT = 1024


def get_onnx_export_opset(dynamo: bool = False) -> int:
    return ONNX_DYNAMO_EXPORT_OPSET if dynamo else ONNX_LEGACY_EXPORT_OPSET


COMPILER = ["/opt/qti-aic/exec/qaic-compile", "-aic-hw"]

ONNX_TRANSFORM_MEMORY_CLEANUP_INTERVAL = 100

# Generic config key aliases used across model families.
ATTENTION_HEAD_CONFIG_KEYS = ("num_attention_heads", "n_head", "n_heads", "num_heads")
KV_HEAD_CONFIG_KEYS = ("num_key_value_heads", "n_kv_heads", "num_kv_heads", "effective_n_kv_heads")
HIDDEN_SIZE_CONFIG_KEYS = ("hidden_size", "n_embd", "d_model")

# InternVL constants
# Fixing the feature size with reference to OpenGVLab/InternVL2_5-1B, OpenGVLab/InternVL2_5-38B and OpenGVLab/InternVL2_5-78B
INTERN_FEATURE_SIZE = 256
INTERN_NUM_PATCHES = 13
INTERN_IMG_SIZE = 448
INTERN_CTX_LEN = 4096
INTERN_PREFILL_SEQ_LEN = INTERN_CTX_LEN - 256  # 4096-256
INTERN_NUM_CHANNELS = 3
INTERN_IMAGE_HEIGHT = 1000
INTERN_IMAGE_WIDTH = 747

INTERN_IMG_CONTEXT_TOKEN = 151667
# Specific to InternVL3_5 series, same token won't work for InternVL2_5 series
INTERN_3_5_IMG_CONTEXT_TOKEN = 151671

# Granite Vision Constants
# Fixing the feature size with reference to ibm-granite/granite-vision-3.2-2b
GRANITEVISION_FEATURE_SIZE = 5239
GRANITEVISION_NUM_PATCHES = 10
GRANITEVISION_IMG_SIZE = 384
GRANITEVISION_IMG_SIZE_HEIGHT = 1109
GRANITEVISION_IMG_SIZE_WIDTH = 1610
GRANITEVISION_PIXEL_VALUE_DIM = 5
GRANITEVISION_PREFIL_SEQ_LEN = GRANITEVISION_SEQ_LEN = 5500
GRANITEVISION_CTX_LEN = 6000
GRANITEVISION_NUM_CHANNELS = 3

VISION_MXFP6_MATMUL = False
VISION_FP16_INPUTS = {"pixel_values", "image_masks"}

# Llama4 Constants
LLAMA4_ATTENTION_CHUNK_SIZE = 8192
LLAMA4_MAX_POSITION_EMBEDDINGS = 65536

# DeepSeek Kimi-k2.5 Constants
KIMI_PATCH_SIZE = 14
KIMI_EXAMPLE_IMAGE_NUM_IMAGE_TOKENS = 600
KIMI_EXAMPLE_IMAGE_NUM_PATCHES_HEIGHT = 30
KIMI_EXAMPLE_IMAGE_NUM_PATCHES_WIDTH = 80
MAX_POSITION_EMBEDDINGS = 32768
FP16_BYTES = 2
DEFAULT_NUM_HEADS = 64
KV_LORA_RANK = 512
ROPE_DIM = 64

# Wav2Vec2 Constant
WAV2VEC2_MAX_SEQ_LEN = 480000  # 30 seconds of audio at 16 kHz sampling rate (16,000 samples/sec × 30 sec)

# Qwen2_5_vl Constants
QWEN2_5_VL_HEIGHT = 354
QWEN2_5_VL_WIDTH = 536
IMAGE_FACTOR_QWEN_2_5 = 28
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384

# Qwen3_vl Constanst
QWEN3_VL_HEIGHT = 354
QWEN3_VL_WIDTH = 536
IMAGE_FACTOR_QWEN_3 = 32

# Modules to cache while clearing the pytorch weights
CACHE_MODULES = ["get_output_names", "get_dummy_inputs", "get_onnx_dynamic_axes", "get_specializations"]

# Mistral3 Constants
MISTRAL3_IMAGE_HEIGHT = 1540
MISTRAL3_IMAGE_WIDTH = 1540

# Molmo Constants
MOLMO_IMAGE_HEIGHT = 536
MOLMO_IMAGE_WIDTH = 354
# Flux Transformer Constants
FLUX_ONNX_EXPORT_SEQ_LENGTH = 256
FLUX_ONNX_EXPORT_COMPRESSED_LATENT_DIM = 4096
FLUX_ADALN_HIDDEN_DIM = 3072
FLUX_ADALN_DUAL_BLOCK_CHUNKS = 12  # 6 chunks for norm1 + 6 chunks for norm1_context
FLUX_ADALN_SINGLE_BLOCK_CHUNKS = 3
FLUX_ADALN_OUTPUT_DIM = 6144  # 2 * FLUX_ADALN_HIDDEN_DIM

# Wan Transformer Constants
WAN_TEXT_EMBED_DIM = 5120
WAN_PROJECTION_DIM = 6
WAN_ONNX_EXPORT_BATCH_SIZE = 1
WAN_ONNX_EXPORT_FRAMES = 81
WAN_ONNX_EXPORT_LATENT_FRAMES = 21
WAN_ONNX_EXPORT_SEQ_LEN = 512
WAN_ONNX_EXPORT_ROTARY_DIM = 128
WAN_DIT_OUT_CHANNELS = 64
# Wan dims for 45p
WAN_ONNX_EXPORT_CL_45P = 252
WAN_ONNX_EXPORT_LATENT_HEIGHT_45P = 6
WAN_ONNX_EXPORT_LATENT_WIDTH_45P = 8
WAN_ONNX_EXPORT_HEIGHT_45P = 48
WAN_ONNX_EXPORT_WIDTH_45P = 64

# WAN I2V
WAN_DIT_I2V_IMG_LATENT_CHANNELS = 32

# For the purpose of automatic CCL lists generation, to limit the number of elements in CCL list, the starting point will be calculated based on context length
CCL_START_MAP = {
    32768: (4096, 4000),
    65536: (8192, 8000),
    float("inf"): (16384, 16000),
}
# Limitation in the maximum number of elements in comp_ctx_lengths_decode and comp_ctx_lengths_prefill lists during automatic lists generation process.
CCL_MAX_ELEMENTS_LISTS = 5
CCL_START_CTX_LEN = 4096
CCL_MIN_CTX_LEN = 1024
CCL_UNIQNE_STEP = 32

# constant for scaling down FC layer for dflash
_DFLASH_TARGET_ABSMAX = 128.0

# used for gpt-oss prefill-only model Q-blocking
GPT_OSS_PREFILL_Q_BLOCK_SIZE = 256


class Constants:
    # Export Constants.
    SEQ_LEN = 32
    CTX_LEN = 32
    PROMPT_LEN = 8
    INPUT_STR = ["My name is"]
    GB = 2**30
    MAX_QPC_LIMIT = 30
    MAX_RETRIES = 10  # This constant will be used set the maximum number of retry attempts for downloading a model using huggingface_hub snapshot_download
    NUM_SPECULATIVE_TOKENS = 2
    MAX_TOP_K_IDS = ONNX_EXPORT_EXAMPLE_MAX_TOP_K_IDS
    SAMPLER_OPS = {
        "repetition_penalties",
        "presence_penalties",
        "temperatures",
        "top_ks",
        "top_ps",
        "min_ps",
        "random_numbers",
    }
    SAMPLER_INPUTS = SAMPLER_OPS | {"last_accepted_output_tokens"}
    SDK_APPS_XML = "/opt/qti-aic/versions/apps.xml"  # This xml file is parsed to find out the SDK apps version.
    SDK_PLATFORM_XML = (
        "/opt/qti-aic/versions/platform.xml"  # This xml file is parsed to find out the SDK platform version.
    )


@dataclass
class QnnConstants:
    # QNN PATH to be read from environment variable.
    QNN_SDK_PATH_ENV_VAR_NAME = "QNN_SDK_ROOT"
    QNN_SDK_YAML = "sdk.yaml"

    # QNN Compilation tools
    QAIRT_CONVERTER = "{}/bin/{}/qairt-converter"
    QNN_CONTEXT_BIN = "{}/bin/{}/qnn-context-binary-generator"

    # QNN Libraries required for compilation
    QNN_CONTEXT_LIB_BACKEND = "{}/lib/{}/libQnnAic.so"
    QNN_CONTEXT_LIB_NET_RUN_EXTENSIONS = "{}/lib/{}/libQnnAicNetRunExtensions.so"

    # QNN Compilation target names
    MODEL_NAME = "model"
    QNN_DATA_FORMAT_CONFIG_NAME = "qnn_data_format_config.json"
    CONTEXT_BIN_NAME = "qnngraph.serialized"
    CONTEXT_BIN_QPC_NAME = "programqpc.bin"

    # TARGET System Architecture
    TARGET = "x86_64-linux-clang"  # TODO add support in infer to be override

    # Converter Arguments
    FLOAT_BITWIDTH = 16
    FLOAT_BIAS_BITWIDTH = 32
    CONVERTER_DEFAULT_ARGS = "--preserve_io_datatype --onnx_skip_simplification --target_backend AIC "

    # Context-Binary-Generator Arguments
    LOG_LEVEL = "error"

    # qnn_compilation_backend default Arguments
    COMPILER_COMPILATION_TARGET = "hardware"
    COMPILER_CONVERT_TO_FP16 = True
    COMPILER_DO_DDR_TO_MULTICAST = True
    COMPILER_HARDWARE_VERSION = "2.0"
    COMPILER_PERF_WARNINGS = False
    COMPILER_PRINT_DDR_STATS = False
    COMPILER_PRINT_PERF_METRICS = False
    COMPILER_RETAINED_STATE = True
    COMPILER_STAT_LEVEL = 10
    COMPILER_STATS_BATCH_SIZE = 1
    COMPILER_TIME_PASSES = False
    GRAPH_NAMES = [f"{MODEL_NAME}_configuration_1", f"{MODEL_NAME}_configuration_2"]
    GRAPH_NAMES_PREFILL_ONLY = [f"{MODEL_NAME}"]

    # qnn_config JSON file supported Keys
    CONVERTER_ARGS_EXTENSION_STR = "converter_args_extension"
    CONTEXT_BIN_ARGS_EXTENSION_STR = "context_binary_generator_args_extension"
    QNN_COMPILATION_BACKEND_STR = "qnn_compilation_backend"
    SKIP_QNN_CONVERTER_STEP_STR = "SKIP_QNN_CONVERTER_STEP"

    IMMUTABLE_CONVERTER_ARGS = [
        "--input_network ",
        "--output_path ",
        "--config ",
        "--float_bias_bitwidth ",
        "--float_bitwidth ",
        "--preserve_io_datatype",
        "--onnx_skip_simplification",
    ]

    IMMUTABLE_CONTEXT_BIN_GEN_ARGS = [
        "--binary_file ",
        "--backend_binary ",
        "--output_dir ",
        "--backend ",
        "--model ",
        "--dlc_path ",
        "--config_file ",
    ]

    QNN_SAMPLE_CONFIG = {
        "converter_args_extension": "--onnx_defer_loading",
        "context_binary_generator_args_extension": "--log_level debug",
        "qnn_compilation_backend": {
            "compiler_enable_depth_first": True,
            "compiler_printDDRStats": False,
            "compiler_printPerfMetrics": False,
        },
        "SKIP_QNN_CONVERTER_STEP": False,
    }


_KNOWN_DECODER_LAYER_ATTR_PATHS = (
    "layers",
    "h",
    "model.layers",
    "model.h",
    "decoder.layers",
    "model.decoder.layers",
    "encoder.layer",
    "encoder.layers",
    "model.encoder.layer",
    "model.encoder.layers",
    "transformer.h",
    "transformer.layers",
    "model.transformer.h",
    "model.transformer.layers",
    "language_model.layers",
    "language_model.model.layers",
    "llm.layers",
    "llm.model.layers",
    "vision_model.encoder.layers",
    "vision_model.transformer.layers",
    "model.vision_model.encoder.layers",
    "model.vision_model.transformer.layers",
    "vision_tower.transformer.layers",
    "vision_tower.vision_model.encoder.layers",
    "model.vision_tower.transformer.layers",
    "model.vision_tower.vision_model.encoder.layers",
)

_KNOWN_DECODER_LAYER_SUFFIXES = (
    ".layers",
    ".layer",
    ".h",
    ".blocks",
    ".block",
    ".encoder_layers",
    ".decoder_layers",
)
