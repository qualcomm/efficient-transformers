# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.hifigan import HiFTGenerator
from chatterbox.models.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder
from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.tts.models.chatterbox.modeling_chatterbox import (
    QEffConvRNNF0Predictor,
    QEffConvTranspose1d,
    QEffHiFTGenerator,
    QEffUpsampleConformerEncoder,
)


class S3GenEncoderTransform(ModuleMappingTransform):
    _module_mapping = {
        UpsampleConformerEncoder: QEffUpsampleConformerEncoder,
    }


class HiFTTransform(ModuleMappingTransform):
    # folds weight_norm (in __qeff_init__), so it must run before ConvTranspose1dTransform
    _module_mapping = {
        HiFTGenerator: QEffHiFTGenerator,
        ConvRNNF0Predictor: QEffConvRNNF0Predictor,
    }


class ConvTranspose1dTransform(ModuleMappingTransform):
    _module_mapping = {
        nn.ConvTranspose1d: QEffConvTranspose1d,
    }
