# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import torch.nn as nn

from QEfficient.base.pytorch_transforms import ProxyModuleMappingTransform
from QEfficient.proxy import QeffProxyEmbedding, QeffProxyLinear


class QeffProxyModuleTransform(ProxyModuleMappingTransform):
    """
    This transform is used to replace the original modules with QEfficient modules.
    """

    _module_mapping = {
        nn.Embedding: QeffProxyEmbedding,
        nn.Linear: QeffProxyLinear,
    }

    @classmethod
    def apply(cls, model: nn.Module):
        """Keep VLM vision modules intact while preserving legacy traversal elsewhere."""
        get_language_decoder = getattr(model, "get_qeff_language_decoder", None)
        if not callable(get_language_decoder):
            return super().apply(model)

        transformed = False
        for getter_name in ("get_input_embeddings", "get_output_embeddings"):
            getter = getattr(model, getter_name, None)
            try:
                module = getter() if callable(getter) else None
            except NotImplementedError:
                module = None
            for base_type, replacement_type in cls._module_mapping.items():
                if isinstance(module, base_type):
                    module.__class__ = replacement_type
                    transformed = True
                    break
        return model, transformed
