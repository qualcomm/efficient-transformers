# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""QEfficient DFlash-2 draft model.

DFlash-2 has no upstream `transformers` implementation, so unlike the other model packages here
there is no HF-side class to swap out: these classes are written export-ready and are used
directly. Only `Qwen3RMSNorm` (inherited from the Qwen3 layers this builds on) is class-swapped,
by the existing `CustomOpsTransform` mapping.
"""

from .modeling_dflash2_draft import (
    CandidateSelector as QEffDFlash2CandidateSelector,
)
from .modeling_dflash2_draft import (
    DFlash2Attention as QEffDFlash2Attention,
)
from .modeling_dflash2_draft import (
    DFlash2DecoderLayer as QEffDFlash2DecoderLayer,
)
from .modeling_dflash2_draft import (
    DFlash2ForCausalLM as QEffDFlash2ForCausalLM,
)
from .modeling_dflash2_draft import (
    DFlash2Model as QEffDFlash2Model,
)
from .modeling_dflash2_draft import (
    GroupedDynamicCausalConv as QEffGroupedDynamicCausalConv,
)
from .modeling_dflash2_draft import (
    remap_dflash2_state_dict,
)

__all__ = [
    "QEffDFlash2Attention",
    "QEffDFlash2CandidateSelector",
    "QEffDFlash2DecoderLayer",
    "QEffDFlash2ForCausalLM",
    "QEffDFlash2Model",
    "QEffGroupedDynamicCausalConv",
    "remap_dflash2_state_dict",
]
