# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import platform
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np


def _bfloat16_to_float32(values: np.ndarray) -> np.ndarray:
    """Decode a numpy float16-sized array containing raw bfloat16 bit patterns."""
    return (values.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def _public_retained_state_name(output_name: str) -> Optional[str]:
    """Map internal subfunction retained-state outputs to public runtime names."""
    suffix = "_InternalRetainedState"
    if output_name.endswith(suffix):
        return output_name[: -len(suffix)] + "_RetainedState"
    return None


def _add_basename_binding_aliases(binding_index_map: Dict[str, int], bindings) -> None:
    """Allow callers to use unprefixed I/O names for prefixed ONNX graphs."""
    for binding in bindings:
        binding_index_map.setdefault(binding.name.rsplit("/", 1)[-1], binding.index)


try:
    import qaicrt

    is_qaicrt_imported = True
except ImportError:
    try:
        sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
        import qaicrt

        is_qaicrt_imported = True
    except ImportError:
        is_qaicrt_imported = False

try:
    import QAicApi_pb2 as aicapi

    is_aicapi_imported = True
except ImportError:
    try:
        sys.path.append("/opt/qti-aic/dev/python")
        import QAicApi_pb2 as aicapi

        is_aicapi_imported = True
    except ImportError:
        is_qaicrt_imported = False

# Imported after the qaicrt/aicapi sys.path fallbacks above, so that
# _kv_dma_handoff's own `import qaicrt` sees the same patched sys.path.
# `is_retained_state_name` is re-exported here for existing external importers
# (vlm_generation.py, text_generation_inference.py, modeling_auto.py).
from QEfficient.generation._kv_dma_handoff import KvDmaHandoff, is_retained_state_name  # noqa: E402,F401


class QAICInferenceSession:
    def __init__(
        self,
        qpc_path: Union[Path, str],
        device_ids: Optional[List[int]] = None,
        activate: bool = True,
        enable_debug_logs: bool = False,
        data_path_timeout_ms: int = 60_000,
        kv_dma_share: bool = False,
        stages: Optional[int] = 1,
        cluster_id: Optional[str] = None,
        full_batch_size: int = 1,
        profiling_type: str | None = None,
        profiling_output_dir: Path | str | None = None,
        profiling_file_prefix: str = "aic-profiling-python",
    ):
        """
        Initialise for QAIC inference Session
        ---------

        :qpc_path: str. Path to the save generated binary file after compilation.
        :device_ids: List[int]. Device Ids to be used for compilation. if devices > 1, it enables multiple card setup.
        :activate: bool. If false, activation will be disabled. Default=True.
        :enable_debug_logs: bool. If True, It will enable debug logs. Default=False.
        :data_path_timeout_ms: int. Host wait timeout (in ms) for a data-path response from the device. Default=60000 (60s).
        :kv_dma_share: bool. If True, enable the DMA-based prefill->decode KV handoff
            path (`np_run` / `np_run_pipeline` / `set_data_for_kv_handoff`). When False
            (default) the session behaves exactly as before: the handoff members are
            inert and only the numpy-copy `run()` path is available.
        :stages: Optional[int]. Prefill pipeline depth; sizes the prefill execObj pool
            (`stages + 1`). Only used when `kv_dma_share=True`. Default=1.
        :cluster_id: Optional[str]. Must be "prefill" or "decode" when `kv_dma_share=True`;
            selects which exec-object pool this session allocates. Unused otherwise.
        :full_batch_size: int. Number of decode slots; `batch_index` offsets wrap
            modulo this value at prefill handoff. Only used when `kv_dma_share=True`.
        :profiling_type: Optional[str]. One of "latency", "trace", "raw_device_stats", "stats". Selects the
            runtime device profiling type (via the `qaicrt.ProfilingHandle` API on this session's `Program`).
            If None (default), profiling support is disabled.
        :profiling_output_dir: Optional[Union[Path, str]]. Directory to write the profiling report to. Defaults to
            a "profiling_output" directory alongside `qpc_path`. Only used when `profiling_type` is set.
        :profiling_file_prefix: str. Filename prefix for the profiling report. Default="aic-profiling-python".
        """
        if not (is_qaicrt_imported and is_aicapi_imported):
            raise ImportError(
                "Unable to import `qaicrt` and/or `QAicApi_pb2` libraries required for executing QPC files on the CLOUD AI platform.\n"
                "Please ensure that the QAIC platform SDK and apps SDK are installed correctly."
            )

        # Build dtype mapping once (depends on aicapi constants)
        self.aic_to_np_dtype_mapping = {
            getattr(aicapi, "BFLOAT16_TYPE", 11): np.dtype(np.float16),
            aicapi.FLOAT_TYPE: np.dtype(np.float32),
            aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
            aicapi.INT8_Q_TYPE: np.dtype(np.int8),
            aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
            aicapi.INT16_Q_TYPE: np.dtype(np.int16),
            aicapi.INT32_Q_TYPE: np.dtype(np.int32),
            aicapi.INT32_I_TYPE: np.dtype(np.int32),
            aicapi.INT64_I_TYPE: np.dtype(np.int64),
            aicapi.INT8_TYPE: np.dtype(np.int8),
        }

        # KV-DMA-share configuration. When disabled, `_kv_dma` stays None and the
        # session keeps a single scalar execObj / qbuffers / buf_dims exactly as
        # before; all handoff state/logic lives in `self._kv_dma`.
        self.kv_dma_share = kv_dma_share
        self.stages = stages if stages is not None else 1
        self.cluster_id = cluster_id
        self.full_batch_size = full_batch_size
        self._kv_dma: Optional[KvDmaHandoff] = KvDmaHandoff(self) if kv_dma_share else None
        if profiling_type is not None:
            profiling_type_map = {
                "latency": "QAIC_PROFILING_INFERENCE_LATENCY_TYPE",
                "trace": "QAIC_PROFILING_INFERENCE_TRACE_TYPE",
                "raw_device_stats": "QAIC_PROFILING_INFERENCE_RAW_DEVICE_STATS_TYPE",
                "stats": "QAIC_PROFILING_INFERENCE_DEV_KPI_TYPE",
            }
            if profiling_type not in profiling_type_map:
                raise ValueError(
                    f"Unsupported profiling_type {profiling_type!r}; expected one of {list(profiling_type_map)}"
                )
            profiling_enum = getattr(
                getattr(qaicrt, "QAicProfilingTypeEnum", None), profiling_type_map[profiling_type], None
            )
            if profiling_enum is None or not hasattr(qaicrt, "ProfilingHandle"):
                raise RuntimeError(
                    f"profiling_type {profiling_type!r} is not supported by the installed QAIC SDK; "
                    "use a supported profiling mode or disable profiling with profiling_type=None."
                )

        # Load QPC
        if device_ids is not None:
            devices = qaicrt.QIDList(device_ids)
            self.context = qaicrt.Context(devices)
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)  # Async API
        if enable_debug_logs:
            if self.context.setLogLevel(qaicrt.QLogLevel.QL_DEBUG) != qaicrt.QStatus.QS_SUCCESS:
                raise RuntimeError("Failed to setLogLevel")
        qpc = qaicrt.Qpc(str(qpc_path))
        # Load IO Descriptor
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to getIoDescriptor")
        iodesc.ParseFromString(bytes(iodesc_data))
        self.allowed_shapes = [
            [(self.aic_to_np_dtype_mapping[x.type].itemsize, list(x.dims)) for x in allowed_shape.shapes]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map = {binding.name: binding.index for binding in self.bindings}
        _add_basename_binding_aliases(self.binding_index_map, self.bindings)
        # Create and load Program
        prog_properties = qaicrt.QAicProgramProperties()
        prog_properties.dataPathTimeoutMs = data_path_timeout_ms
        dev_id_non_mq = None
        if device_ids:
            if len(device_ids) == 1:
                dev_id_non_mq = device_ids[0]
            elif len(device_ids) > 1:
                prog_properties.devMapping = ":".join(map(str, device_ids))
        self.is_active = False
        self.program = qaicrt.Program(self.context, dev_id_non_mq, qpc, prog_properties)
        if self.program.load() != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to load program")
        self.profiling_handle = None
        if profiling_type is not None:
            output_dir = (
                Path(profiling_output_dir)
                if profiling_output_dir
                else (Path(qpc_path) if Path(qpc_path).is_dir() else Path(qpc_path).parent) / "profiling_output"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            self.profiling_handle = qaicrt.ProfilingHandle(
                programs=[self.program],
                type=profiling_enum,
                fileNamePrefix=profiling_file_prefix,
                outputDirectory=str(output_dir),
            )
        if activate:
            self.activate()
        if self._kv_dma is None:
            # Create input qbuffers and buf_dims (single-execObj `run()` path)
            self.qbuffers = [qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings]
            self.buf_dims = qaicrt.BufferDimensionsVecRef(
                [(self.aic_to_np_dtype_mapping[binding.type].itemsize, list(binding.dims)) for binding in self.bindings]
            )
        else:
            # Per-slot qbuffers / buf_dims for the pooled DMA-handoff path.
            self.qbuffers = [
                [qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings] for _ in range(self._queue_len)
            ]
            self.buf_dims = [
                qaicrt.BufferDimensionsVecRef(
                    [
                        (self.aic_to_np_dtype_mapping[binding.type].itemsize, list(binding.dims))
                        for binding in self.bindings
                    ]
                )
                for _ in range(self._queue_len)
            ]
            self._kv_dma.init_buffer_maps()

    @property
    def _queue_len(self) -> int:
        return self._kv_dma.queue_len if self._kv_dma is not None else 1

    @property
    def decode_execObj_idx(self) -> Optional[int]:
        return self._kv_dma.decode_execObj_idx if self._kv_dma is not None else None

    @property
    def kv_cache_info(self):
        return self._kv_dma.kv_cache_info

    @property
    def decode_buff_map(self):
        return self._kv_dma.decode_buff_map

    @property
    def decode_rs_kv_only_buff_map(self):
        return self._kv_dma.decode_rs_kv_only_buff_map

    @property
    def input_names(self) -> List[str]:
        return [binding.name for binding in self.bindings if binding.dir == aicapi.BUFFER_IO_TYPE_INPUT]

    @property
    def output_names(self) -> List[str]:
        return [binding.name for binding in self.bindings if binding.dir == aicapi.BUFFER_IO_TYPE_OUTPUT]

    def binding_is_bfloat16(self, name: str) -> bool:
        """
        True if the named binding's on-device dtype is bfloat16.

        numpy has no native bfloat16 dtype, so `aic_to_np_dtype_mapping` maps
        BFLOAT16_TYPE to `np.float16` purely to get a matching 2-byte itemsize;
        the bytes carried in such a buffer are real bfloat16 bit patterns, not
        numeric float16 values. Callers that build/consume these buffers must
        bit-cast rather than numerically cast, and can use this to decide which
        conversion applies.
        """
        if name not in self.binding_index_map:
            return False
        return self.bindings[self.binding_index_map[name]].type == getattr(aicapi, "BFLOAT16_TYPE", 11)

    def activate(self):
        """Activate qpc"""
        if not self.is_active:
            try:
                self.program.activate()
                self.is_active = True
                if self._kv_dma is not None:
                    self.execObj = [qaicrt.ExecObj(self.context, self.program) for _ in range(self._queue_len)]
                else:
                    self.execObj = qaicrt.ExecObj(self.context, self.program)
            except Exception:
                try:
                    self._deactivate(force=True)
                except Exception as cleanup_error:
                    warn(f"Failed to deactivate QAIC program after activation error: {cleanup_error}", RuntimeWarning)
                raise

    def _deactivate(self, force: bool = False):
        """Deactivate qpc"""
        if force or getattr(self, "is_active", False):
            if hasattr(self, "execObj"):
                del self.execObj
            try:
                self.program.deactivate()
            finally:
                self.is_active = False

    def deactivate(self):
        """Deactivate qpc"""
        self._deactivate()

    def __del__(self):
        """Best-effort release of device resources when the session is discarded."""
        try:
            self.deactivate()
        except Exception:
            pass

    def start_profiling(self):
        """Start capturing a profiling report for this session's program(s)."""
        if self.profiling_handle is None:
            raise RuntimeError("Profiling is not enabled for this session; pass `profiling_type` to the constructor.")

        status = self.profiling_handle.start()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError(f"Failed to start profiling. Status {status}")

    def stop_profiling(self):
        """Stop profiling and flush the report to `profiling_output_dir`."""
        if self.profiling_handle is None:
            raise RuntimeError("Profiling is not enabled for this session; pass `profiling_type` to the constructor.")
        status = self.profiling_handle.stop()

        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError(f"Failed to stop profiling. Status {status}")

    @contextmanager
    def profile(self):
        """Context manager that brackets a block of `run()` calls with start/stop profiling."""
        self.start_profiling()
        try:
            yield
        finally:
            self.stop_profiling()

    def _release_program_after_run_failure(self) -> None:
        """Release device resources while preserving the original execution error."""
        try:
            self.deactivate()
        except Exception as cleanup_error:
            warn(f"Failed to deactivate QAIC program after execution error: {cleanup_error}", RuntimeWarning)

        try:
            if self.program.unload() != qaicrt.QStatus.QS_SUCCESS:
                warn("Failed to unload QAIC program after execution error", RuntimeWarning)
        except Exception as cleanup_error:
            warn(f"Failed to unload QAIC program after execution error: {cleanup_error}", RuntimeWarning)

    def set_buffers(self, buffers: Dict[str, np.ndarray]):
        """
        Provide buffer mapping for input and output

        Args:
            :buffer (Dict[str, np.ndarray]): Parameter for buffer mapping.
        """

        for buffer_name, buffer in buffers.items():
            if buffer_name not in self.binding_index_map:
                warn(f'Buffer: "{buffer_name}" not found')
                continue
            buffer_index = self.binding_index_map[buffer_name]
            self.qbuffers[buffer_index] = qaicrt.QBuffer(buffer.tobytes())
            self.buf_dims[buffer_index] = (
                buffer.itemsize,
                buffer.shape if len(buffer.shape) > 0 else (1,),
            )

    def skip_buffers(self, skipped_buffer_names: List[str]):
        """
        skip buffer mapping for given list of buffer names

        Args:
            :skipped_buffer_name: List[str]. List of buffer name to be skipped.
        """

        self.set_buffers({k: np.array([]) for k in skipped_buffer_names})

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute on cloud AI 100

        Args:
            :inputs (Dict[str, np.ndarray]): Processed numpy inputs for the model.

        Return:
            :Dict[str, np.ndarray]:
        """
        # Set inputs
        self.set_buffers(inputs)
        if self.execObj.setData(self.qbuffers, self.buf_dims) != qaicrt.QStatus.QS_SUCCESS:
            self._release_program_after_run_failure()
            raise MemoryError("Failed to setData")
        # # Run with sync API
        # if self.execObj.run(self.qbuffers) != qaicrt.QStatus.QS_SUCCESS:
        # Run with async API
        if self.queue.enqueue(self.execObj) != qaicrt.QStatus.QS_SUCCESS:
            self._release_program_after_run_failure()
            raise MemoryError("Failed to enqueue")
        if self.execObj.waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            self._release_program_after_run_failure()
            raise ValueError(self._shape_mismatch_message(self.buf_dims))
        # Get output buffers
        status, output_qbuffers = self.execObj.getData()
        if status != qaicrt.QStatus.QS_SUCCESS:
            self._release_program_after_run_failure()
            raise MemoryError("Failed to getData")
        return self._build_outputs(output_qbuffers, self.qbuffers, self.buf_dims)

    def _shape_mismatch_message(self, buf_dims) -> str:
        """Build the "Failed to run" diagnostic listing allowed vs passed shapes."""
        error_message = "Failed to run"
        # Print additional error messages for unmatched dimension error
        if self.allowed_shapes:
            error_message += "\n\n"
            error_message += '(Only if "No matching dimension found" error is present above)'
            error_message += "\nAllowed shapes:"
            for i, allowed_shape in enumerate(self.allowed_shapes):
                error_message += f"\n{i}\n"
                for binding, (elemsize, shape), (_, passed_shape) in zip(self.bindings, allowed_shape, buf_dims):
                    if passed_shape == [0]:
                        if not binding.is_partial_buf_allowed:
                            warn(f"Partial buffer not allowed for: {binding.name}")
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            error_message += "\n\nPassed shapes:\n"
            for binding, (elemsize, shape) in zip(self.bindings, buf_dims):
                if shape == [0]:
                    continue
                error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
        return error_message

    def _build_outputs(self, output_qbuffers, qbuffers, buf_dims) -> Dict[str, np.ndarray]:
        """Decode device output buffers into a name-keyed dict of numpy arrays."""
        outputs = {}
        for output_name in self.output_names:
            buffer_index = self.binding_index_map[output_name]
            # Skip unmapped outputs and DMA-wired RetainedState buffers, whose data
            # goes straight to the caller's host arrays so getData returns empty.
            if qbuffers[buffer_index].size == 0 or output_qbuffers[buffer_index].size == 0:
                continue
            output = np.frombuffer(
                bytes(output_qbuffers[buffer_index]),
                self.aic_to_np_dtype_mapping[self.bindings[buffer_index].type],
            ).reshape(buf_dims[buffer_index][1])
            if output_name.rsplit("/", 1)[-1] == "logits" and self.binding_is_bfloat16(output_name):
                output = _bfloat16_to_float32(output)
            outputs[output_name] = output
            output_basename = output_name.rsplit("/", 1)[-1]
            outputs.setdefault(output_basename, output)
            public_name = _public_retained_state_name(output_name)
            if public_name is not None:
                outputs[public_name] = output
                outputs.setdefault(public_name.rsplit("/", 1)[-1], output)
        return outputs

    # ------------------------------------------------------------------
    # DMA-based KV handoff path (enabled only when kv_dma_share=True);
    # state/logic lives in QEfficient.generation._kv_dma_handoff.KvDmaHandoff.
    # ------------------------------------------------------------------

    def set_persistent_inputs(self, buffers: Dict[str, np.ndarray]) -> None:
        self._kv_dma.set_persistent_inputs(buffers)

    def set_data_for_kv_handoff(self, kv_cache_buffers, slicing_parameters, index=0, buff_map=None):
        return self._kv_dma.set_data_for_kv_handoff(kv_cache_buffers, slicing_parameters, index, buff_map)

    def np_run(self, inputs: Dict[str, np.ndarray], slicing_parameters=None, is_prefill: bool = True) -> int:
        return self._kv_dma.np_run(inputs, slicing_parameters, is_prefill)

    def np_run_pipeline(
        self, inputs: Dict[str, np.ndarray], slicing_parameters=None, last_chunk: bool = False, kv_cache_buffers=None
    ) -> int:
        return self._kv_dma.np_run_pipeline(inputs, slicing_parameters, last_chunk, kv_cache_buffers)

    def complete_inf(self, index: int, is_prefill: bool) -> None:
        self._kv_dma.complete_inf(index, is_prefill)

    def get_outputs(self, index: int) -> Dict[str, np.ndarray]:
        return self._kv_dma.get_outputs(index)
