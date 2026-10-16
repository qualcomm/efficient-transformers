# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import shutil
import subprocess
import warnings
from typing import List, Optional, Tuple

from QEfficient.compile.qnn_compiler import compile as qnn_compile
from QEfficient.utils import constants
from QEfficient.utils._utils import load_json, load_yaml, to_named_specializations
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


def create_and_dump_specializations(
    batch_size: int, prompt_len: int, ctx_len: int, path: str, full_batch_size: Optional[int] = None
):
    # Build the base specialization entries first, then convert to named format.
    base_specializations = [
        {
            "batch_size": str(batch_size),
            "seq_len": str(prompt_len),
            "ctx_len": str(ctx_len),
        },
        {"batch_size": str(batch_size), "seq_len": "1", "ctx_len": str(ctx_len)},
    ]
    # If continuous batching is enabled by providing full_batch_size we need to add FBS to the specialization file and update the batch size of decoder part to FBS
    if full_batch_size is not None:
        base_specializations[0]["full_batch_size"] = str(full_batch_size)
        base_specializations[1]["full_batch_size"] = str(full_batch_size)
        base_specializations[1]["batch_size"] = str(full_batch_size)

    # To handle repetitive input in specializations when prompt_len is 1
    if prompt_len == 1 and full_batch_size is None:
        base_specializations.pop()

    # Dump
    specializations = {"specializations": to_named_specializations(base_specializations)}
    with open(path, "w") as file:
        json.dump(specializations, file, indent=4)


def compile_kv_model_on_cloud_ai_100(
    onnx_path: str,
    specializations_json: str,
    num_cores: int,
    base_path: str,
    mxfp6: bool,
    custom_io_path: str,
    aic_enable_depth_first: bool,
    allow_mxint8_mdp_io: bool,
    mos: int = -1,
    device_group: Optional[List[int]] = None,
    **kwargs,
) -> Tuple[bool, str]:
    """
    Compiles an ONNX Key-Value (KV) model for Cloud AI 100 hardware using `qaic-compile`.

    This function sets up and executes the Qualcomm AI 100 compiler with various options
    to generate a QPC package.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model file to be compiled.
    specializations_json : str
        Path to the JSON file defining compilation specializations (batch size, sequence length, etc.).
    num_cores : int
        Number of cores to use for compilation on Cloud AI 100.
    base_path : str
        Base directory where QPC binaries will be stored (a `qpcs` subdirectory will be created).
    mxfp6 : bool
        If True, enables MXFP6 precision for MatMul weights.
    custom_io_path : str
        Path to the Custom IO list file (e.g., YAML format) specifying input/output data types.
    aic_enable_depth_first : bool
        If True, enables Depth-First Search (DFS) optimization with default memory size.
    allow_mxint8_mdp_io : bool
        If True, allows MXINT8 compression of MDP IO traffic.

    Other Parameters
    ----------------
    mos : int, optional
        Effort level to reduce on-chip memory. A value greater than 0 applies this effort. Default is -1 (no effort).
    device_group : List[int], optional
        List of device IDs for multi-device compilation (tensor slicing). If `len(device_group) > 1`,
        a multi-device partition configuration is generated. Default is None.
    **kwargs :
        Additional compiler options passed directly to `qaic-compile`. These are formatted as
        `-key=value` or `-key` for boolean flags.

    Returns
    -------
    Tuple[bool, str]
        A tuple containing:
        - bool: True if compilation was successful, False otherwise.
        - str: Path to the generated QPC binary directory.

    Raises
    ------
    FileNotFoundError
        If the `specializations_json` or `custom_io_path` files are not found.
    RuntimeError
        If the `qaic-compile` compilation process fails.

    Warnings
    --------
    DeprecationWarning
        This method will be removed soon; use `QEFFAutoModelForCausalLM.compile` instead.
    """
    warnings.warn(
        "\033[93mUse `QEFFAutoModelForCausalLM.compile` instead, this method will be removed soon.\033[0m",
        DeprecationWarning,
        stacklevel=2,
    )
    aic_binary_dir = os.path.join(base_path, "qpcs")

    if os.path.isdir(aic_binary_dir):
        shutil.rmtree(aic_binary_dir)

    if not os.path.isfile(specializations_json):
        raise FileNotFoundError(f"Please use 'QEfficient.compile', as {specializations_json} file was not found")
    if not os.path.isfile(custom_io_path):
        raise FileNotFoundError(f"{custom_io_path} file was not found!")
    command = [
        "/opt/qti-aic/exec/qaic-compile",
        f"-m={onnx_path}",
        "-aic-hw",
        f"-aic-hw-version={kwargs.pop('aic_hw_version', kwargs.pop('aic-hw-version', constants.DEFAULT_AIC_HW_VERSION))}",
        f"-network-specialization-config={specializations_json}",
        "-convert-to-fp16",
        "-retained-state",
        f"-aic-num-cores={num_cores}",
        f"-custom-IO-list-file={custom_io_path}",
        f"-aic-binary-dir={aic_binary_dir}",
    ]
    if mxfp6:
        command.append("-mxfp6-matmul")
    if mos > 0:
        command.append(f"-mos={mos}")
    if aic_enable_depth_first:
        command.append("-aic-enable-depth-first")
    if allow_mxint8_mdp_io:
        command.append("-allow-mxint8-mdp-io")
    if device_group is not None and len(device_group) > 1:
        mdp_ts_config = {
            "connections": [{"devices": list(range(len(device_group))), "type": "p2p"}],
            "partitions": [
                {
                    "name": "Partition0",
                    "devices": [{"deviceId": device, "numCores": num_cores} for device in range(len(device_group))],
                }
            ],
        }
        mdp_ts_config_path = os.path.join(base_path, "mdp_ts_config.json")
        with open(mdp_ts_config_path, "w") as file:
            json.dump(mdp_ts_config, file, indent=4)
        command.append(f"-mdp-load-partition-config={mdp_ts_config_path}")
    for key, value in kwargs.items():
        option = "-" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
            continue
        command.append(f"{option}={value}")
    print("Running AI 100 compiler:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")

    print("\n===================== Compilation Done! =====================\n")
    return result.returncode == 0, aic_binary_dir


def compile(
    onnx_path: str,
    qpc_path: str,
    num_cores: int,
    device_group: Optional[List[int]] = None,  #  FIXME: use num_devices instead
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    batch_size: int = 1,
    prompt_len: int = 32,
    ctx_len: int = 128,
    mxfp6: bool = True,
    mxint8: bool = False,
    custom_io_file_path: Optional[str] = None,
    full_batch_size: Optional[int] = None,
    allow_mxint8_mdp_io: Optional[bool] = False,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Compiles the given ONNX model using either the Cloud AI 100 platform SDK compiler
    or the QNN compiler, and saves the compiled QPC package.

    This function handles the creation of specialization files, selection of custom IO
    configurations, and execution of the appropriate compiler (QAIC or QNN).
    It supports multi-device compilation for tensor slicing.

    Parameters
    ----------
    onnx_path : str
        Path to the generated ONNX model file.
    qpc_path : str
        Target directory path for saving the compiled QPC binaries.
    num_cores : int
        Number of cores to use for compilation.

    Other Parameters
    ----------------
    device_group : List[int], optional
        List of device IDs. Used to determine the number of devices for multi-device compilation.
        Default is None.
    aic_enable_depth_first : bool, optional
        If True, enables Depth-First Search (DFS) optimization with default memory size during QAIC compilation.
        Default is False.
    mos : int, optional
        Effort level to reduce on-chip memory during QAIC compilation. A value greater than 0 applies this effort.
        Default is -1 (no effort).
    batch_size : int, optional
        Batch size to compile the model for. Default is 1.
    full_batch_size : int, optional
        Sets the full batch size to enable continuous batching mode. If provided, `batch_size` must be 1.
        Default is None.
    prompt_len : int, optional
        Prompt length for the model to compile. Default is 32.
    ctx_len : int, optional
        Maximum context length to compile the model for. Default is 128.
    mxfp6 : bool, optional
        If True, enables MXFP6 precision for MatMul weights during compilation. Default is True.
    mxint8 : bool, optional
        If True, compresses Present/Past KV to MXINT8 using a CustomIO configuration. Default is False.
    custom_io_file_path : str, optional
        Explicit path to a Custom IO file (e.g., YAML format). If None, it's inferred based on `mxint8`.
        Default is None.
    allow_mxint8_mdp_io : bool, optional
        If True, allows MXINT8 compression of MDP IO traffic during QAIC compilation. Default is False.
    enable_qnn : bool, optional
        If True, enables compilation using the QNN compiler instead of QAIC. Default is False.
    qnn_config : str, optional
        Path to the QNN Config parameters file, used if `enable_qnn` is True. Default is None.
    **kwargs :
        Additional compiler options passed directly to the chosen compiler.

    Returns
    -------
    str
        Path to the compiled QPC package directory.

    Raises
    ------
    ValueError
        If both `batch_size` and `full_batch_size` are greater than one (mutually exclusive in some contexts).
    FileNotFoundError
        If required Custom IO files are not found.

    Warnings
    --------
    DeprecationWarning
        This method will be removed soon; use `QEFFAutoModelForCausalLM.compile` instead.

    """

    if device_group is not None:
        warnings.warn(
            "device_group is deprecated and will be renamed to device_ids in the next release.",
            FutureWarning,
            stacklevel=2,
        )

    if full_batch_size and batch_size != 1:
        raise ValueError("Only either batch_size or full_batch_size should be greater than one")

    os.makedirs(qpc_path, exist_ok=True)
    specialization_json_path = os.path.join(qpc_path, "specializations.json")

    create_and_dump_specializations(
        batch_size=batch_size,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        path=specialization_json_path,
        full_batch_size=full_batch_size,
    )

    dtype_suffix = "int8" if mxint8 else "fp16"
    source_path = f"./custom_io_{dtype_suffix}.yaml"
    destination_path = os.path.join(os.path.dirname(qpc_path), f"custom_io_{dtype_suffix}.yaml")

    # Move the custom YAML file to the cache/qeff_model directory
    try:
        shutil.move(source_path, destination_path)
        print(f"Successfully moved '{source_path}' to '{destination_path}'.")
    except Exception as e:
        print(f"Error while moving file '{source_path}': {e}")

    custom_io_file_name = f"custom_io_{dtype_suffix}.yaml"
    if custom_io_file_path is None:
        custom_io_file_path = os.path.join(os.path.dirname(qpc_path), custom_io_file_name)

    if not os.path.isfile(custom_io_file_path):
        raise FileNotFoundError(
            f"Custom IO file {custom_io_file_name} is not present at the expected path {custom_io_file_path}. Please pass the correct file path or rerun infer/export API"
        )

    if enable_qnn:
        qpc_path = qnn_compile(
            onnx_path=onnx_path,
            qpc_base_path=qpc_path,
            qnn_binary_dir=os.path.join(qpc_path, "qpcs"),
            num_cores=num_cores,
            mxfp6=mxfp6,
            mxint8=mxint8,
            allow_mxint8_mdp_io=allow_mxint8_mdp_io,
            aic_enable_depth_first=aic_enable_depth_first,
            mos=mos,
            device_group=device_group,
            qnn_config=qnn_config,
            specializations=(load_json(specialization_json_path))["specializations"],
            custom_io=load_yaml(custom_io_file_path),
        )
        logger.info(f"QNN Compiled QPC files can be found here: {qpc_path}")
    else:
        _, qpc_path = compile_kv_model_on_cloud_ai_100(
            onnx_path=onnx_path,
            specializations_json=specialization_json_path,
            num_cores=num_cores,
            custom_io_path=custom_io_file_path,
            base_path=qpc_path,
            mxfp6=mxfp6,
            aic_enable_depth_first=aic_enable_depth_first,
            allow_mxint8_mdp_io=allow_mxint8_mdp_io,
            mos=mos,
            device_group=device_group,
            **kwargs,
        )
        if kwargs.get("io_encrypt", None):
            logger.warning(
                f"Compilation for IO-Encrypt has been successfully completed at path: {qpc_path}. However, Efficient-Transformers do not support IO-Encrypt execution. Please run the execution separately"
            )
        else:
            logger.info(f"Compiled QPC files can be found here: {qpc_path}")

    return qpc_path
