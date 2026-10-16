# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import hashlib
import json
import os
import shutil
import warnings
from typing import Dict, List, Optional

from QEfficient.utils._utils import create_json, execute_command, load_json, to_named_specializations
from QEfficient.utils.constants import QnnConstants
from QEfficient.utils.generate_qnn_network_specialization_config import (
    generate_data_format_config,
    generate_qnn_specialization,
)
from QEfficient.utils.hash_utils import to_hashable
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


class QNN:
    """
    The QNN class is designed for providing QNN compilation support for exported ONNX models.
    This class enables use of QNN (Qualcomm Neural Network) sdk for compiling and running ml models on target device.

    """

    def __init__(
        self,
        onnx_path: str,
        qpc_base_path: str,
        num_cores: int,
        custom_io_path: str,
        device_group: Optional[List[int]] = None,
        compiler_enable_depth_first: bool = False,
        compiler_max_out_channel_split: int = -1,
        compiler_mxfp6_matmul_weights: bool = True,
        qnn_target: str = QnnConstants.TARGET,
        qnn_config_path: Optional[str] = None,
        qnn_binary_dir: Optional[str] = None,
        mxint8: Optional[bool] = False,
        compiler_mxint8_mdp_io: Optional[bool] = False,
        prefill_only: Optional[bool] = False,
        **kwargs,
    ) -> None:
        self.onnx_path = onnx_path
        self.qpc_base_path = qpc_base_path
        self.num_cores = num_cores
        self.device_group = device_group
        self.compiler_enable_depth_first = compiler_enable_depth_first
        self.compiler_max_out_channel_split = compiler_max_out_channel_split
        self.compiler_mxfp6_matmul_weights = compiler_mxfp6_matmul_weights
        self.qnn_config_path = qnn_config_path
        self.qnn_binary_dir = qnn_binary_dir
        self.mxint8 = mxint8
        self.compiler_mxint8_mdp_io = compiler_mxint8_mdp_io
        self.custom_io_path = custom_io_path
        self.dlc_model_path = os.path.join(qpc_base_path, f"{QnnConstants.MODEL_NAME}.dlc")
        self.qnn_target = qnn_target
        self.prefill_only = prefill_only
        self.qnn_sdk_path = os.getenv(QnnConstants.QNN_SDK_PATH_ENV_VAR_NAME)
        if not self.qnn_sdk_path:
            raise EnvironmentError(
                f"QNN_SDK_PATH {self.qnn_sdk_path} is not set. Please set {QnnConstants.QNN_SDK_PATH_ENV_VAR_NAME}"
            )

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Parse qnn_config file if present.
        self.qnn_config = None
        if self.qnn_config_path:
            self.parse_qnn_config()

    def check_extension_arg(self, ext_arg_key, ext_arg_value, immutable_arg_list):
        """
        Checks if the passed compile stage extension arguments are valid or not.
        Raises an AttributeError if any immutable argument in present in the extension argument value.

        ``Mandatory`` Args:
            :ext_arg_key (str): Extension argument key.
            :ext_arg_value (str): Extension argument value as present in the passed qnn_config.json
            :immutable_arg_list (List): List containing parameters which can not be modified using qnn_config.json

        """

        immutable_param = [param for param in immutable_arg_list if param in ext_arg_value]
        if immutable_param:
            raise AttributeError(
                f"Immutable Parameters {immutable_param} found in {ext_arg_key}. Please remove {immutable_param} from {ext_arg_key}"
            )

    def parse_qnn_config(self):
        """
        Parsed qnn_config.json file passed by the user for QNN Configuration and stores the key, value pair in class object.

        """
        config_data = load_json(self.qnn_config_path)

        self.qnn_config = {}
        # Copy key-value pairs to the class object
        for key, value in config_data.items():
            if key == QnnConstants.CONVERTER_ARGS_EXTENSION_STR:
                self.check_extension_arg(key, value, QnnConstants.IMMUTABLE_CONVERTER_ARGS)
            if key == QnnConstants.CONTEXT_BIN_ARGS_EXTENSION_STR:
                self.check_extension_arg(key, value, QnnConstants.IMMUTABLE_CONTEXT_BIN_GEN_ARGS)
            self.qnn_config[key] = value

    def create_qnn_tensor_slicing_json(self) -> str:
        """
        Creates tensor_slicing.json file if device_group contains more than 1 device.

        Returns:
            :str: Path to tensor_slicing.json file.
        """
        tensor_slicing = {
            "connections": [{"devices": list(range(len(self.device_group))), "type": "p2p"}],
            "partitions": [
                {
                    "name": "Partition0",
                    "devices": [{"deviceId": device} for device in range(len(self.device_group))],
                }
            ],
        }
        tensor_slicing_json_path = os.path.join(self.qpc_base_path, "tensor_slicing.json")
        create_json(tensor_slicing_json_path, tensor_slicing)
        return tensor_slicing_json_path

    def create_qnn_compile_backend_json(self) -> str:
        """
        Creates qnn_compile_backend.json file containing qnn_compilation_backend parameters.
        If qnn_config.json file is passed, default values will be over-written.

        Returns:
            :str: Path to qnn_compile_backend.json file.
        """
        qnn_compile_backend = {
            "compiler_compilation_target": QnnConstants.COMPILER_COMPILATION_TARGET,
            "compiler_hardware_version": QnnConstants.COMPILER_HARDWARE_VERSION,
            "compiler_convert_to_FP16": QnnConstants.COMPILER_CONVERT_TO_FP16,
            "compiler_retained_state": QnnConstants.COMPILER_RETAINED_STATE,
            "graph_names": QnnConstants.GRAPH_NAMES_PREFILL_ONLY if self.prefill_only else QnnConstants.GRAPH_NAMES,
            "compiler_enable_depth_first": self.compiler_enable_depth_first,
            "compiler_mxfp6_matmul_weights": self.compiler_mxfp6_matmul_weights,
            "compiler_num_of_cores": self.num_cores,
            "compiler_do_DDR_to_multicast": QnnConstants.COMPILER_DO_DDR_TO_MULTICAST,
            "compiler_perfWarnings": QnnConstants.COMPILER_PERF_WARNINGS,
            "compiler_printDDRStats": QnnConstants.COMPILER_PRINT_DDR_STATS,
            "compiler_printPerfMetrics": QnnConstants.COMPILER_PRINT_PERF_METRICS,
            "compiler_stat_level": QnnConstants.COMPILER_STAT_LEVEL,
            "compiler_stats_batch_size": QnnConstants.COMPILER_STATS_BATCH_SIZE,
            "compiler_time_passes": QnnConstants.COMPILER_TIME_PASSES,
            "compiler_mxint8_mdp_io": self.compiler_mxint8_mdp_io,
        }
        if self.compiler_max_out_channel_split > 0:
            qnn_compile_backend["compiler_max_out_channel_split"] = str(self.compiler_max_out_channel_split)

        if self.device_group is not None and len(self.device_group) > 1:
            qnn_compile_backend["compiler_mdp_load_partition_config"] = self.create_qnn_tensor_slicing_json()

        if self.qnn_config and QnnConstants.QNN_COMPILATION_BACKEND_STR in self.qnn_config:
            for key, value in self.qnn_config[QnnConstants.QNN_COMPILATION_BACKEND_STR].items():
                qnn_compile_backend[key] = value

        qnn_compile_backend_json_path = os.path.join(self.qpc_base_path, "qnn_compile_backend.json")
        create_json(qnn_compile_backend_json_path, qnn_compile_backend)
        return qnn_compile_backend_json_path

    def create_qnn_compiler_config_json(self) -> str:
        """
        Creates qnn_compiler_config.json file containing path to qnn_compile_backend.json file & shared_library_path.
        Config file is passed to QNN context-binary-generator.

        Returns:
            :str: Path to qnn_compiler_config.json file.
        """
        qnn_compiler_config = {
            "backend_extensions": {
                "config_file_path": self.create_qnn_compile_backend_json(),
                "shared_library_path": QnnConstants.QNN_CONTEXT_LIB_NET_RUN_EXTENSIONS.format(
                    self.qnn_sdk_path, self.qnn_target
                ),
            }
        }
        qnn_compiler_config_json_path = os.path.join(self.qpc_base_path, "qnn_compiler_config.json")
        create_json(qnn_compiler_config_json_path, qnn_compiler_config)
        return qnn_compiler_config_json_path

    def compile(self) -> str:
        """
        Compiles the given ``ONNX`` model during object creation using QNN compiler and saves the compiled ``qpc`` package at ``qnn_binary_dir``.
            - Creates converter command and convert onnx model to model.dlc using qairt-converter
            - command line arguments and qnn_config.json (if provided) are used to create qnn_compiler_config.json for context-binary-generator
            - model.dlc from converter stage is passed into context-binary-generator command to create programqpc.bin.

        Returns:
            :str: Path to compiled ``qpc`` package.
        """
        if not (
            self.qnn_config
            and (QnnConstants.SKIP_QNN_CONVERTER_STEP_STR in self.qnn_config)
            and self.qnn_config[QnnConstants.SKIP_QNN_CONVERTER_STEP_STR]
        ):
            converter_cmd = self.converter()
            execute_command("converter", converter_cmd, self.qpc_base_path)

        if not os.path.isfile(self.dlc_model_path):
            raise FileNotFoundError(
                f"file {self.dlc_model_path} needs to exist in the qpc_base_path{self.qpc_base_path}. Please rerun infer/compile Api"
            )

        if self.qnn_binary_dir is None:
            self.qnn_binary_dir = os.path.join(self.qpc_base_path, "qpcs")
        if os.path.isdir(self.qnn_binary_dir):
            shutil.rmtree(self.qnn_binary_dir)
        os.makedirs(self.qnn_binary_dir)

        ctx_bin_cmd = self.generate_context_binary()
        execute_command("context_binary", ctx_bin_cmd, self.qpc_base_path)

        print("\n===================== Compilation Done! =====================\n")
        return self.qnn_binary_dir

    def converter(self) -> str:
        """
        Creates QNN converter command using provided options.

        IMMUTABLE parameters which can not be overridden by the user using qnn_config.json:
            :input_network (str): Generated ``ONNX`` Model Path.
            :output_path (str): Path to generated DLC file, which is provided qpc_base_path/model.dlc
            :config (str): Path to custom_io_config.yaml file created using GenerateQNNnetworkSpecializationconfig.py
            :float_bias_bitwidth (int): Bitwidth to use for float bias tensor
            :float_bitwidth (int): Converts the graph to the specified float bitwidth, either 32 or 16(Default).
            :preserve_io_datatype(flag): Passed by default.

        CONVERTER_ARGS_EXTENSION passed in qnn_config.json is appended to the command created.

        Returns:
            :str: QNN Converter command.
        """
        converter_tool = QnnConstants.QAIRT_CONVERTER.format(self.qnn_sdk_path, self.qnn_target)

        cmd = (
            f"{converter_tool} --input_network {self.onnx_path} "
            f"--output_path {self.dlc_model_path} "
            f"--config {self.custom_io_path} "
            f"--float_bias_bitwidth {QnnConstants.FLOAT_BIAS_BITWIDTH} "
            f"--float_bitwidth {QnnConstants.FLOAT_BITWIDTH} "
        )
        # Add default arguments.
        cmd += QnnConstants.CONVERTER_DEFAULT_ARGS

        if self.qnn_config and QnnConstants.CONVERTER_ARGS_EXTENSION_STR in self.qnn_config:
            cmd += self.qnn_config[QnnConstants.CONVERTER_ARGS_EXTENSION_STR]

        return cmd

    def generate_context_binary(self) -> str:
        """
        Creates QNN context-binary-generator command using provided options.

        IMMUTABLE parameters which can not be modified by the user using qnn_config.json:
            :binary_file (str): QNN Binary Graph name to be generated (qnngraph.serialized).
            :backend_binary (str): Generated QPC binary file name, which is provided programqpc.bin
            :output_dir (str): Path to store generated Binaries (qnn_binary_dir).
            :model (str): Path to the <qnn_model_name.so> file containing a QNN network.
            :dlc_path (str): Path to DLC file generated by QNN-Converter.
            :config_file(str): Path to created qnn_compiler_config.json containing qnn_compile_backend.json & shared_library_path.

        Configurable parameters:
            :log_level(str): ``Configurable`` Default(error).

        CONTEXT_BIN_ARGS_EXTENSION passed in qnn_config.json is appended to the command created.

        Returns:
            :str: QNN Context Binary Generator command.
        """
        binary_gen_tool = QnnConstants.QNN_CONTEXT_BIN.format(self.qnn_sdk_path, self.qnn_target)
        backend_lib = QnnConstants.QNN_CONTEXT_LIB_BACKEND.format(self.qnn_sdk_path, self.qnn_target)
        config_file_path = self.create_qnn_compiler_config_json()

        cmd = (
            f"{binary_gen_tool} --binary_file {QnnConstants.CONTEXT_BIN_NAME} "
            f"--backend_binary {QnnConstants.CONTEXT_BIN_QPC_NAME} "
            f"--output_dir {self.qnn_binary_dir} "
            f"--backend {backend_lib} "
            f"--dlc_path {self.dlc_model_path} "
            f"--config_file {config_file_path} "
        )

        if self.mxint8:
            data_format_file_path = os.path.join(self.qpc_base_path, QnnConstants.QNN_DATA_FORMAT_CONFIG_NAME)
            generate_data_format_config(
                self.onnx_path, model_dlc_name=QnnConstants.MODEL_NAME, file_path=data_format_file_path
            )
            if not os.path.isfile(data_format_file_path):
                raise FileNotFoundError(
                    f"file {data_format_file_path} needs to exist in the qpc_base_path for mxint8 compilation. Please rerun infer/compile Api"
                )
            cmd += f"--data_format_config {data_format_file_path} "

        if self.qnn_config and QnnConstants.CONTEXT_BIN_ARGS_EXTENSION_STR in self.qnn_config:
            if "--log_level " not in self.qnn_config[QnnConstants.CONTEXT_BIN_ARGS_EXTENSION_STR]:
                cmd += f"--log_level {QnnConstants.LOG_LEVEL} "
            cmd += self.qnn_config[QnnConstants.CONTEXT_BIN_ARGS_EXTENSION_STR]
        else:
            cmd += f"--log_level {QnnConstants.LOG_LEVEL} "

        return cmd

    def quantize(self):
        raise NotImplementedError("QNN Quantization is not supported")

    def execute(self):
        raise NotImplementedError("QNN Execution is not supported")

    def generate_profiling(self):
        raise NotImplementedError("QNN profiling is not supported")


def compile(
    onnx_path: str,
    qpc_base_path: str,
    num_cores: int,
    device_group: Optional[List[int]] = None,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    mxfp6: bool = True,
    mxint8: bool = False,
    allow_mxint8_mdp_io: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    qnn_binary_dir: Optional[str] = None,
    custom_io: Optional[Dict[str, str]] = None,
    specializations: Optional[List[Dict[str, int]]] = None,
    **kwargs,
) -> str:
    """
    Compiles the given ``ONNX`` model using QNN compiler and saves the compiled ``qpc`` package at ``qnn_binary_dir``.
    Generates model.dlc during converter stage, qnn_compile_backend.json for backend parameters of context-binary-generator.
    Generates tensor-slicing configuration if multiple devices are passed in ``device_group``.

    ``Mandatory`` Args:
        :onnx_path (str): Generated ``ONNX`` Model Path.
        :qpc_base_path (str): base directory for QNN compilation config & binary file.
        :num_cores (int): Number of cores to compile the model on.
    ``Optional`` Args:
        :device_group (List[int]): Used for finding the number of devices to compile for.
        :aic_enable_depth_first (bool): Enables ``DFS`` with default memory size. ``Defaults to False.``
        :mos (int): Effort level to reduce the on-chip memory. ``Defaults to -1.``
        :mxfp6 (bool): Enable compilation for ``MXFP6`` precision.  ``Defaults to True.``
        :mxint8 (bool): Compress Present/Past KV to ``MXINT8`` using ``CustomIO`` config. ``Defaults to False.``
        :allow_mxint8_mdp_io (bool): Allows MXINT8 compression of MDP IO traffic ``Defaults to False.``
        :qnn_config (str): Path to ``qnn_config.json`` file (formatted as a string). ``Defaults to None.``
        :qnn_binary_dir (str): Path for saving qnn binaries.
        :custom_io (dict): Custom IO to specify the input and outputs in different formats than default
        :specializations (list): List of specializations to compile for

    Returns:
        :str: Path to compiled ``qpc`` package.
    """

    if device_group is not None:
        warnings.warn(
            "device_group is deprecated and will be renamed to device_ids in the next release.",
            FutureWarning,
            stacklevel=2,
        )

    if kwargs:
        logger.warning("Extra arguments to QNN compilation are not supported as of now!")
        raise NotImplementedError("Can't handle extra compilation args now!")

    os.makedirs(qpc_base_path, exist_ok=True)

    # Created custom_io_config.yaml file for QNN-Converter stage.
    # TODO To make custom_io_config.yaml configurable as not all models need it.
    custom_io_file_path = os.path.join(qpc_base_path, "custom_io_config.yaml")

    generate_qnn_specialization(
        onnx_graph_path=onnx_path,
        specializations=specializations,
        custom_io=custom_io,
        file_path=custom_io_file_path,
    )

    if not os.path.isfile(custom_io_file_path):
        raise FileNotFoundError(
            f"file {custom_io_file_path} needs to exist in the qpc_base_path for Compilation. Please rerun infer/compile Api"
        )

    prefill_only = True if len(specializations) == 1 else False

    if qnn_binary_dir is None:
        compile_hash = hashlib.sha256(to_hashable("qnn"))

        if specializations is not None:
            compile_hash.update(to_hashable(specializations))

        if custom_io is not None:
            compile_hash.update(to_hashable(custom_io))

        if qnn_config is not None:
            qnn_config_values = load_json(qnn_config)
            compile_hash.update(to_hashable(qnn_config_values))

        if device_group is not None:
            compile_hash.update(to_hashable({"device_group": device_group}))

        compile_hash.update(to_hashable({"num_cores": num_cores}))
        compile_hash.update(to_hashable({"mxfp6": mxfp6}))
        compile_hash.update(to_hashable({"mxint8": mxint8}))

        # Check if already compiled
        compile_hash = compile_hash.hexdigest()[:16]

        qnn_binary_dir = qpc_base_path / "qpc"
        qnn_binary_dir = qnn_binary_dir.with_name(qnn_binary_dir.name + "-" + compile_hash)
        if qnn_binary_dir.is_dir():
            if (qnn_binary_dir / "programqpc.bin").is_file():
                return qnn_binary_dir
            # Probably compilation failure last time, delete directory to start over
            shutil.rmtree(qnn_binary_dir)

        # Write specializations.json file
        if specializations is not None:
            specializations_json = qpc_base_path / "specializations.json"
            with open(specializations_json, "w") as fp:
                json.dump(
                    {"specializations": to_named_specializations(specializations)},
                    fp,
                    indent=4,
                )

    qnn_obj = QNN(
        onnx_path=onnx_path,
        qpc_base_path=qpc_base_path,
        num_cores=num_cores,
        device_group=device_group,
        qnn_config_path=qnn_config,
        custom_io_path=custom_io_file_path,
        compiler_enable_depth_first=aic_enable_depth_first,
        compiler_max_out_channel_split=mos,
        compiler_mxfp6_matmul_weights=mxfp6,
        qnn_binary_dir=qnn_binary_dir,
        mxint8=mxint8,
        compiler_mxint8_mdp_io=allow_mxint8_mdp_io,
        prefill_only=prefill_only,
    )

    compiled_binary_path = qnn_obj.compile()
    return compiled_binary_path
