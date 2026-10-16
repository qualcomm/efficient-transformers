# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

# ----------------------------------------------------------------------------- #
# For faster downloads via hf_transfer
# This code is put above import statements as this needs to be executed before
# hf_transfer is imported (will happen on line 15 via leading imports)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# DO NOT ADD ANY CODE ABOVE THIS LINE
# Please contact maintainers if you must edit this file above this line.
# ----------------------------------------------------------------------------- #
# Placeholder for all non-transformer models registered in QEfficient
import warnings  # noqa: I001
import transformers
import transformers.utils as transformers_utils
from transformers.utils import import_utils as hf_import_utils

try:
    from transformers import HybridCache as _TransformersHybridCache  # noqa: F401
except ImportError:
    from transformers.cache_utils import DynamicCache

    class HybridCache(DynamicCache):
        pass

    class HybridChunkedCache(HybridCache):
        pass

    transformers.HybridCache = HybridCache
    transformers.HybridChunkedCache = HybridChunkedCache

if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
    transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
import QEfficient.utils.model_registery  # noqa: F401
from QEfficient.base import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForImageTextToText,
    QEFFAutoModelForSequenceClassification,
    QEFFAutoModelForSpeechSeq2Seq,
    QEFFCommonLoader,
)
from QEfficient.compile.compile_helper import compile
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
from QEfficient.peft import QEffAutoPeftModelForCausalLM
from QEfficient.transformers.transform import transform
from QEfficient.utils import custom_format_warning
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


try:
    from QEfficient.diffusers.pipelines.flux.pipeline_flux import QEffFluxPipeline
    from QEfficient.diffusers.pipelines.wan.pipeline_wan import QEffWanPipeline
    from QEfficient.diffusers.pipelines.wan.pipeline_wan_i2v import QEffWanImageToVideoPipeline
except Exception:
    QEffFluxPipeline = None
    QEffWanPipeline = None
    QEffWanImageToVideoPipeline = None

try:
    from QEfficient.peft import QEffAutoPeftModelForCausalLM
except Exception:
    QEffAutoPeftModelForCausalLM = None

# custom warning for the better logging experience
warnings.formatwarning = custom_format_warning


# Users can use QEfficient.export for exporting models to ONNX
export = qualcomm_efficient_converter
__all__ = [
    "transform",
    "export",
    "compile",
    "cloud_ai_100_exec_kv",
    "QEFFAutoModel",
    "QEFFAutoModelForCausalLM",
    "QEFFAutoModelForCTC",
    "QEffAutoPeftModelForCausalLM",
    "QEFFAutoModelForImageTextToText",
    "QEFFAutoModelForSequenceClassification",
    "QEFFAutoModelForSpeechSeq2Seq",
    "QEFFCommonLoader",
]

if QEffFluxPipeline is not None:
    __all__.append("QEffFluxPipeline")
if QEffWanPipeline is not None:
    __all__.append("QEffWanPipeline")
if QEffWanImageToVideoPipeline is not None:
    __all__.append("QEffWanImageToVideoPipeline")


# Conditionally import QAIC-related modules if the SDK is installed
__version__ = "1.23.0.dev0"


def check_qaic_sdk():
    """Check if QAIC SDK is installed"""
    try:
        import platform
        import sys

        sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
        import qaicrt  # noqa: F401

        return True
    except ImportError:
        return False


if not check_qaic_sdk():
    logger.warning("QAIC SDK is not installed, eager mode features won't be available!")


def ensure_torch_fx_import_compatibility():
    if hasattr(hf_import_utils, "is_torch_fx_available"):
        return

    def _is_torch_fx_available() -> bool:
        if not hf_import_utils.is_torch_available():
            return False
        try:
            import torch.fx  # noqa: F401

            return True
        except Exception:
            return False

    hf_import_utils.is_torch_fx_available = _is_torch_fx_available


ensure_torch_fx_import_compatibility()
