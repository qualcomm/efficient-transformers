# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch
from torch import nn

from QEfficient.customop.onnxscript_utils import qeff_custom_op
from QEfficient.customop.utils import select_interface
from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_LEGACY_EXPORT_OPSET))


@qeff_custom_op("com.qti.aisw.onnx", 1)
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float) -> onnxscript.FLOAT:
    hidden_states_fp32 = ops.Cast(hidden_states, to=1)
    weight = ops.Cast(weight, to=1)
    variance = ops.ReduceMean(ops.Pow(hidden_states_fp32, 2), axes=[-1], keepdims=1)
    epsilon = ops.Expand(epsilon, ops.Shape(variance))
    hidden_states_fp32 = hidden_states_fp32 * ops.Reciprocal(ops.Sqrt(variance + epsilon))
    return ops.CastLike(weight * hidden_states_fp32, hidden_states)


class CustomRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(variance + epsilon)
        return (weight.float() * hidden_states_fp32).to(dtype=hidden_states.dtype)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)


class CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def __init__(self, hidden_size, eps=1e-05):
        super(CustomRMSNormAIC, self).__init__()
        self.variance_epsilon = eps
        self.eps = eps  # Added to support GemmaRMSNorm
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        rms_interface = select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)
        return rms_interface(
            hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )


class GemmaCustomRMSNormAIC(CustomRMSNormAIC):
    """
    HF's GemmaRMSNorm stores weight as zeros and adds 1.0 at forward time.
    Apply the +1.0 offset here at runtime so both normal and weight-free export
    see the correct value without modifying the stored parameter.
    """

    def forward(self, hidden_states):
        rms_interface = select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)
        eps = self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        return rms_interface(hidden_states, self.weight + 1.0, eps)
