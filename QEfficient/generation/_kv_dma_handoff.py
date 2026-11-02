# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
from queue import Queue
from typing import Dict, List, Optional, Tuple
from warnings import warn

import numpy as np

# Slicing-spec DimSpecs templates, chosen per KV binding's attention type.
# The symbolic "batch_index" / "ctx_start" tokens are resolved at handoff time
# from the (name, offset) pairs passed as `slicing_parameters`.
FULL_ATTN_DIMSPEC = [
    {"start": "batch_index"},
    {"start": 0},
    {"start": "ctx_start"},
    {"start": 0},
]
LINEAR_ATTN_DIMSPEC = [
    {"start": "batch_index"},
    {"start": 0},
    {"start": 0},
]

try:
    import qaicrt
except ImportError:
    qaicrt = None


def _is_linear_state_name(name: str) -> bool:
    """Return True for linear/recurrent (3-D) retained-state bindings."""
    return name.startswith(("conv_state.", "recurrent_state."))


def _kv_layer_sort_key(item: Tuple[str, int]) -> Tuple[int, str]:
    """Order KV bindings by layer index, then name.

    Standard attention sorts ``past_key`` before ``past_value``; MLA sorts
    ``compressed_kv`` before ``k_pe``. The layer index is parsed from the first
    dotted segment (``past_key.3`` -> 3); names without a dotted index sort as 0.
    """
    name = item[0]
    part = name.split(".")[1] if "." in name else "0"
    return int(part.split("_")[0]), name


def is_retained_state_name(name: str) -> bool:
    """Return True when an I/O binding participates in retained-state cache flow."""
    return name.startswith(("past_", "conv_state.", "recurrent_state.", "compressed_", "k_pe"))


class KvDmaHandoff:
    """Owns the DMA-based prefill->decode KV handoff state and runtime path for a
    ``QAICInferenceSession`` with ``kv_dma_share=True``.

    Reads the base QPC/IO-descriptor state (``session.bindings``,
    ``session.binding_index_map``, ``session.aic_to_np_dtype_mapping``, ``session.program``,
    etc.) that ``QAICInferenceSession`` has already built; does not duplicate QPC loading.
    """

    def __init__(self, session):
        self.session = session

        self.decode_execObj_idx: Optional[int] = None
        if session.cluster_id == "decode":
            self.prefill_num_execObj = 0
            self.decode_num_execObj = 1
            self.decode_execObj_idx = 0
            self.queue_len = 1
        elif session.cluster_id == "prefill":
            self.prefill_num_execObj = session.stages + 1
            self.decode_num_execObj = 0
            self.queue_len = self.prefill_num_execObj
        else:
            raise ValueError(
                f'cluster_id must be "prefill" or "decode" when kv_dma_share=True, got {session.cluster_id!r}'
            )
        # Prefill exec slots follow the single decode slot (index 0) in the pool.
        self.prefill_available_exec_objs: Queue = Queue()
        for i in range(self.decode_num_execObj, self.decode_num_execObj + self.prefill_num_execObj):
            self.prefill_available_exec_objs.put(i)

    def _build_kv_bindings(self) -> List[Tuple[str, int, int]]:
        """One canonical, validated ordering of KV retained-state bindings.

        Every downstream buffer map is derived from this single list so they can
        never drift out of sync with each other (e.g. from independently sorting
        differently-shaped name sets). Cross-checks that every retained-state
        input has a matching ``_RetainedState`` output and vice versa, failing
        fast instead of silently zipping mismatched bindings together.

        Returns a list of ``(base_name, input_index, output_index)``, ordered by
        ``_kv_layer_sort_key``.
        """
        session = self.session
        suffix = "_RetainedState"
        input_map = {
            name: session.binding_index_map[name] for name in session.input_names if is_retained_state_name(name)
        }
        output_map = {
            name[: -len(suffix)]: session.binding_index_map[name]
            for name in session.output_names
            if name.endswith(suffix) and is_retained_state_name(name[: -len(suffix)])
        }
        if input_map.keys() != output_map.keys():
            missing_output = sorted(input_map.keys() - output_map.keys())
            missing_input = sorted(output_map.keys() - input_map.keys())
            raise ValueError(
                "Retained-state KV binding mismatch between decode inputs and outputs: "
                f"inputs with no matching {suffix!r} output: {missing_output}; "
                f"outputs with no matching input: {missing_input}"
            )
        ordered_names = sorted(input_map, key=lambda name: _kv_layer_sort_key((name, 0)))
        return [(name, input_map[name], output_map[name]) for name in ordered_names]

    def init_buffer_maps(self):
        """Build ordered buffer maps and the compiled KV slicing spec."""
        session = self.session
        self.kv_bindings = self._build_kv_bindings()

        self.decode_buff_map = [(name, input_index) for name, input_index, _ in self.kv_bindings]
        self.decode_rs_kv_only_buff_map = [
            (f"{name}_RetainedState", output_index) for name, _, output_index in self.kv_bindings
        ]
        self.decode_rs_full_buff_map = [(name, output_index) for name, _, output_index in self.kv_bindings]
        self.kv_only_buff_map = [
            (name, output_index)
            for name, output_index in self.decode_rs_full_buff_map
            if not _is_linear_state_name(name)
        ]

        # Per-slot KV geometry, in canonical kv_bindings order.
        self.kv_cache_info: List[Tuple[tuple, np.dtype]] = [
            (
                tuple(session.bindings[input_index].dims),
                session.aic_to_np_dtype_mapping[session.bindings[input_index].type],
            )
            for _, input_index, _ in self.kv_bindings
        ]

        # Hybrid iff more than one distinct (shape, dtype) KV family exists.
        distinct = {(shape, dtype.str) for shape, dtype in self.kv_cache_info}
        self.is_hybrid_kv = len(distinct) > 1

        self.kv_slicing_spec_handle = None
        if self.kv_cache_info:
            spec_json = (
                self._build_full_kv_slicing_json() if self.is_hybrid_kv else self._build_uniform_kv_slicing_json()
            )
            self.kv_slicing_spec_handle = self._create_slicing_spec_handle(spec_json)

        self.readable_output_bindings = [
            (name, session.binding_index_map[name])
            for name in session.output_names
            if not name.endswith("_RetainedState")
        ]
        self.output_buffers: List[Dict[str, np.ndarray]] = [
            {
                name: np.zeros(
                    tuple(session.bindings[index].dims),
                    dtype=session.aic_to_np_dtype_mapping[session.bindings[index].type],
                )
                for name, index in self.readable_output_bindings
            }
            for _ in range(self.queue_len)
        ]

        self.persistent_inputs: Dict[str, np.ndarray] = {}

    def _build_uniform_kv_slicing_json(self) -> str:
        """One BufferSpec per KV name family; every KV shares shape/dtype."""
        elem_size = self.kv_cache_info[0][1].itemsize
        names = sorted({name.split(".")[0] for name, _ in self.kv_only_buff_map})
        buffer_specs = [{"Name": f"{base}.*", "ElemSize": elem_size, "DimSpecs": FULL_ATTN_DIMSPEC} for base in names]
        return json.dumps({"BufferSpecs": buffer_specs})

    def _build_full_kv_slicing_json(self) -> str:
        """One BufferSpec per RetainedState binding; DimSpecs chosen by ndim."""
        session = self.session
        buffer_specs = []
        for binding in session.bindings:
            name = binding.name
            if not (name.endswith("_RetainedState") and is_retained_state_name(name)):
                continue
            base_name = name.replace("_RetainedState", "")
            elem_size = session.aic_to_np_dtype_mapping[binding.type].itemsize
            dim_spec = FULL_ATTN_DIMSPEC if len(binding.dims) == 4 else LINEAR_ATTN_DIMSPEC
            buffer_specs.append({"Name": f"{base_name}(_.*)?", "ElemSize": elem_size, "DimSpecs": dim_spec})
        return json.dumps({"BufferSpecs": buffer_specs})

    def _create_slicing_spec_handle(self, buffer_spec_json: str):
        status, slicing_spec_handle = self.session.program.createSlicingSpecHandle(buffer_spec_json)
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to create SlicingSpecHandle")
        return slicing_spec_handle

    def _tuple_list_from_dict(self, inputs: Dict[str, np.ndarray]) -> List[tuple]:
        """Map a name-keyed input dict to (binding_index, buffer) tuples."""
        session = self.session
        tuple_list = []
        for name, buffer in inputs.items():
            if name not in session.binding_index_map:
                warn(f'Buffer: "{name}" not found')
                continue
            if buffer is None:
                continue
            tuple_list.append((session.binding_index_map[name], buffer))
        return tuple_list

    @staticmethod
    def _make_inputs_contiguous(inputs: Dict[str, np.ndarray]) -> None:
        for name, buffer in inputs.items():
            inputs[name] = np.ascontiguousarray(buffer)

    def set_persistent_inputs(self, buffers: Dict[str, np.ndarray]) -> None:
        session = self.session
        for name, buffer in buffers.items():
            if name not in session.binding_index_map:
                warn(f'Buffer: "{name}" not found')
                continue
            self.persistent_inputs[name] = np.ascontiguousarray(buffer)

    def _tuple_list_with_outputs(self, inputs: Dict[str, np.ndarray], exec_obj_idx: int) -> List[tuple]:
        session = self.session
        tuple_list = self._tuple_list_from_dict(inputs)
        for name, buffer in self.persistent_inputs.items():
            if name in inputs:
                continue
            tuple_list.append((session.binding_index_map[name], buffer))
        for name, index in self.readable_output_bindings:
            tuple_list.append((index, self.output_buffers[exec_obj_idx][name]))
        return tuple_list

    def set_data_for_kv_handoff(self, kv_cache_buffers, slicing_parameters, index=0, buff_map=None):
        """Wire a sliced DMA descriptor so the runtime writes RetainedState
        outputs directly into ``kv_cache_buffers`` at the ``slicing_parameters``
        offsets. ``buff_map`` is a list of ``(name, binding_index)`` whose order
        must match ``kv_cache_buffers``.
        """
        if buff_map is None:
            raise ValueError("set_data_for_kv_handoff requires a buff_map")
        if not (len(kv_cache_buffers) == len(buff_map) or len(kv_cache_buffers) + 1 == len(buff_map)):
            raise ValueError(
                f"KV buffer count mismatch: expected {len(buff_map)} (or {len(buff_map) - 1}), "
                f"got {len(kv_cache_buffers)}"
            )
        slices = []
        for (name, binding_index), buf in zip(buff_map, kv_cache_buffers):
            binding = self.session.bindings[binding_index]
            binding_shape = tuple(binding.dims)
            buf_shape = tuple(buf.shape)
            # Dim 0 (batch) is sliced by "batch_index" in the DimSpec, so a buffer
            # covering a single row (e.g. one prefill slot) legitimately has a
            # smaller dim 0 than the binding's full compiled shape. Every other
            # dim is written in full and must match exactly.
            shape_ok = (
                len(binding_shape) == len(buf_shape)
                and binding_shape[1:] == buf_shape[1:]
                and 1 <= buf_shape[0] <= binding_shape[0]
            )
            if not shape_ok:
                raise ValueError(
                    f"KV buffer shape mismatch for {name!r} (binding {binding_index}): "
                    f"expected {binding_shape} (or a smaller dim-0 slice of it), got {buf_shape}"
                )
            expected_dtype = self.session.aic_to_np_dtype_mapping[binding.type]
            if expected_dtype != buf.dtype:
                raise ValueError(
                    f"KV buffer dtype mismatch for {name!r} (binding {binding_index}): "
                    f"expected {expected_dtype}, got {buf.dtype}"
                )
            slices.append((binding_index, buf))
        status, _ = self.session.execObj[index].setDataWithSlices(
            slices, self.kv_slicing_spec_handle, slicing_parameters
        )
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to setDataWithSlices")
        return kv_cache_buffers

    def np_run(self, inputs: Dict[str, np.ndarray], slicing_parameters=None, is_prefill: bool = True) -> int:
        session = self.session
        if is_prefill:
            exec_obj_idx = self.prefill_available_exec_objs.get()
        else:
            if self.decode_execObj_idx is None:
                raise RuntimeError("No decode execObj configured for this session")
            exec_obj_idx = self.decode_execObj_idx
        self._make_inputs_contiguous(inputs)
        tuple_list = self._tuple_list_with_outputs(inputs, exec_obj_idx)
        if slicing_parameters is None:
            status = session.execObj[exec_obj_idx].setData(tuple_list)
        else:
            status, _ = session.execObj[exec_obj_idx].setDataWithSlices(
                tuple_list, self.kv_slicing_spec_handle, slicing_parameters
            )
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to setData")
        if session.queue.enqueue(session.execObj[exec_obj_idx]) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        return exec_obj_idx

    def np_run_pipeline(
        self, inputs: Dict[str, np.ndarray], slicing_parameters=None, last_chunk: bool = False, kv_cache_buffers=None
    ) -> int:
        session = self.session
        exec_obj_idx = self.prefill_available_exec_objs.get()
        if last_chunk:
            if kv_cache_buffers is None:
                raise ValueError("last_chunk requires kv_cache_buffers to wire the handoff")
            batch_index = int(inputs["batch_index"].item()) if "batch_index" in inputs else 0
            buff_map = self.decode_rs_full_buff_map if self.is_hybrid_kv else self.kv_only_buff_map
            self.set_data_for_kv_handoff(
                kv_cache_buffers,
                [("batch_index", batch_index % session.full_batch_size), ("ctx_start", 0)],
                exec_obj_idx,
                buff_map,
            )
        self._make_inputs_contiguous(inputs)
        tuple_list = self._tuple_list_with_outputs(inputs, exec_obj_idx)
        if slicing_parameters is None:
            status = session.execObj[exec_obj_idx].setData(tuple_list)
        else:
            status, _ = session.execObj[exec_obj_idx].setDataWithSlices(
                tuple_list, self.kv_slicing_spec_handle, slicing_parameters
            )
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to setData")
        if session.queue.enqueue(session.execObj[exec_obj_idx]) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        return exec_obj_idx

    def complete_inf(self, index: int, is_prefill: bool) -> None:
        """Block until execObj ``index`` finishes; release prefill slots back to
        the pool.
        """
        session = self.session
        if session.execObj[index].waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            raise ValueError(session._shape_mismatch_message(session.buf_dims[index]))
        if is_prefill:
            self.prefill_available_exec_objs.put(index)

    def get_outputs(self, index: int) -> Dict[str, np.ndarray]:
        """Return the readable (non-RetainedState) outputs of execObj ``index``.

        On the pooled path the runtime DMA-writes these outputs into the per-slot
        host arrays wired at enqueue, so we read straight from ``output_buffers``
        (``getData`` returns empty for tuple-list enqueues). RetainedState KV goes
        directly to the caller's shared arrays via the slicing spec and is not
        surfaced here.
        """
        outputs: Dict[str, np.ndarray] = {}
        for name, _ in self.readable_output_bindings:
            output = self.output_buffers[index][name]
            outputs[name] = output
            outputs.setdefault(name.rsplit("/", 1)[-1], output)
        return outputs
