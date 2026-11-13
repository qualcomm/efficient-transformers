# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np

from QEfficient.generation.cloud_infer import is_retained_state_name


class DFlashTlmHandoff:
    """Host buffers for a single-request prefill-to-verification state transfer."""

    def __init__(self, prefill_session, decode_session):
        self.inputs = {}
        self.outputs = {}
        dtype_mapping = decode_session.aic_to_np_dtype_mapping
        if any(len(getattr(session, "allowed_shapes", ())) > 1 for session in (prefill_session, decode_session)):
            raise ValueError("Split DFlash TLM requires one specialization in each QPC.")
        input_names = {name for name in decode_session.input_names if is_retained_state_name(name)}
        output_names = {
            name.removesuffix("_RetainedState")
            for name in prefill_session.output_names
            if name.endswith("_RetainedState") and is_retained_state_name(name)
        }
        if not input_names or input_names != output_names:
            raise ValueError("DFlash TLM prefill/decode QPCs must expose matching retained-state bindings.")
        for name in sorted(input_names):
            output_name = name + "_RetainedState"
            source = prefill_session.bindings[prefill_session.binding_index_map[output_name]]
            target = decode_session.bindings[decode_session.binding_index_map[name]]
            if tuple(source.dims) != tuple(target.dims) or source.type != target.type or source.size != target.size:
                raise ValueError(f"DFlash TLM state shape or precision mismatch for {name}.")
            if target.dims[0] != 1:
                raise ValueError("Split DFlash TLM currently supports batch size 1 only.")
            buffer = np.empty(tuple(target.dims), dtype=dtype_mapping[target.type])
            self.inputs[name] = buffer
            self.outputs[output_name] = buffer

    def capture(self, outputs):
        """Copy outputs from the standalone session, whose run() returns new arrays."""
        for name, buffer in self.outputs.items():
            np.copyto(buffer, outputs[name])
