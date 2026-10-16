# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil
import warnings
from typing import Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.base.common import QEFFCommonLoader
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.exporter.export_utils import export_onnx, fix_onnx_fp16, generate_input_files, run_model_on_ort
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import load_hf_tokenizer
from QEfficient.utils.constants import QEFF_MODELS_DIR, Constants
from QEfficient.utils.generate_inputs import InputHandler
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


def convert_to_cloud_bertstyle(
    model_name: str,
    qeff_model: QEFFAutoModelForCausalLM,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_len: int,
) -> str:
    """
    API to convert model to Bertstyle approach.
    Bertstyle Approach:
            1. No Prefill/Decode separably compiled.
            2. No KV retention logic.
            3. KV is every time computed for all the tokens until EOS/max_length.

    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: `gpt2`.
        :qeff_model (QEFFAutoModelForCausalLM): Transformed KV torch model to be used.
        :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Model tokenizer.
        :onnx_dir_path (str): Path to save exported ONNX file.
        :seq_len (int): The length of the sequence.

    Returns:
         :str: Path of exported ``ONNX`` file.
    """
    if os.path.exists(onnx_dir_path):
        logger.warning(f"Overriding {onnx_dir_path}")
        shutil.rmtree(onnx_dir_path)

    # Decide path for saving exported ONNX files.
    model_name = export_bertstyle_model_to_onnx(model_name, qeff_model.model, tokenizer, onnx_dir_path, seq_len)  # type: ignore

    # return the model path for automation.
    return os.path.join(onnx_dir_path, f"{model_name}.onnx")


def export_bertstyle_model_to_onnx(model_name, model, tokenizer, onnx_dir_path, seq_len) -> str:
    model_base_name = model_name.replace("/", "_") + "_bertstyle"
    os.makedirs(onnx_dir_path, exist_ok=True)

    input_str = Constants.INPUT_STR
    # Preprocess inputs
    if seq_len > 0:
        inputs = tokenizer(
            input_str,
            return_tensors="pt",
            padding="max_length",
            max_length=seq_len,
        )
    else:
        inputs = tokenizer(input_str, return_tensors="pt")
    if model.config.is_encoder_decoder:
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        inputs["decoder_input_ids"] = torch.full((1, 1), model.generation_config.decoder_start_token_id)

    # Run PyTorch inference
    try:
        pt_outputs = model(**inputs)
        output_names = list(pt_outputs.keys())
    except Exception as e:
        print(f"Model {model_name} Execution failed in pytorch:%s", e)

    # Add pkv into output_names
    pkv = tuple([(key.detach(), value.detach()) for key, value in pt_outputs.past_key_values])
    pkv_idx = output_names.index("past_key_values")
    key_value_names = [f"past_{x}.{i}" for i in range(len(pkv)) for x in ["key", "value"]]
    output_names[pkv_idx : pkv_idx + 1] = [x for x in key_value_names]

    pt_outputs = dict(pt_outputs)
    pkv_out = pt_outputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv_out):
        pt_outputs[f"past_key.{i}"] = key
        pt_outputs[f"past_value.{i}"] = value

    # Export the model to Onnx.
    try:
        model_name = export_onnx(
            pt_model=model,
            inputs=inputs,
            output_names=output_names,
            gen_models_path=onnx_dir_path,
            model_base_name=model_base_name,
        )
    except Exception as e:
        print(f"Model {model_name} failed to export in Onnx:%s", e)

    # Run onnxrt inference
    input_names, ort_outputs = run_model_on_ort(
        onnx_path=os.path.join(onnx_dir_path, f"{model_name}.onnx"),
        inputs=inputs,
        output_names=output_names,
        pt_outputs=pt_outputs,
    )

    # Fix onnx for fp16
    # Clip the values to fp16 ranges to avoid over/under flow in AI 100
    model_name = fix_onnx_fp16(
        inputs=inputs,
        output_names=output_names,
        ort_outputs=ort_outputs,
        gen_models_path=onnx_dir_path,
        model_base_name=model_name,
        pt_outputs=pt_outputs,
    )

    # Generate inputFiles
    input_list_file = os.path.join(onnx_dir_path, "input_list.txt")
    generate_input_files(
        input_files_path=os.path.join(onnx_dir_path, "inputFiles"),
        input_names=input_names,
        inputs=inputs,
        input_list_file=input_list_file,
    )

    return model_name


def convert_to_cloud_kvstyle(
    model_name: str,
    qeff_model: QEFFAutoModelForCausalLM,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_len: int,
) -> str:
    """
    API to convert model with kv retention and export to ONNX.
    KV Style Approach-
        1. This architecture is particularly suitable for auto-regressive tasks.
        2. where sequence generation involves processing one token at a time.
        3. And contextual information from earlier tokens is crucial for predicting the next token.
        4. The inclusion of a kV cache enhances the efficiency of the decoding process, making it more computationally efficient.

    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: `gpt2`.
        :qeff_model (QEFFAutoModelForCausalLM): Transformed KV torch model to be used.
        :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Model tokenizer.
        :onnx_dir_path (str): Path to save exported ONNX file.
        :seq_len (int): The length of the sequence.

    Returns:
         :str: Path of exported ``ONNX`` file.
    """
    if os.path.exists(onnx_dir_path):
        logger.warning(f"Overriding {onnx_dir_path}")
        shutil.rmtree(onnx_dir_path)

    if not qeff_model.is_transformed:
        raise Exception(f"please pass the {qeff_model.__class__.__name__} after transform API")

    # Decide path for saving exported ONNX files.
    model_name = export_kvstyle_transformed_model_to_onnx(
        model_name, qeff_model.model, tokenizer, onnx_dir_path, seq_len
    )  # type: ignore

    # return the model path for automation.
    return os.path.join(onnx_dir_path, f"{model_name}.onnx")


def export_kvstyle_transformed_model_to_onnx(
    model_name: str,
    transformed_model: torch.nn.Module,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_len: int,
    full_batch_size: Optional[int] = None,
) -> str:
    # Disabling requires_grad on all parameters
    for _, p in enumerate(transformed_model.parameters()):
        p.requires_grad_(False)

    if seq_len <= 0:
        raise ValueError(f"Need seq_len to be greater than zero, got seq_len={seq_len}")

    # Preprocess inputs
    # Build inputs for prefill
    input_handler = InputHandler(
        batch_size=len(Constants.INPUT_STR),
        tokenizer=tokenizer,
        config=transformed_model.config,
        prompt=Constants.INPUT_STR * (full_batch_size if full_batch_size else 1),
        prompt_len=Constants.PROMPT_LEN,
        ctx_len=seq_len,
        full_batch_size=full_batch_size,
    )

    inputs = input_handler.prepare_pytorch_inputs()
    pt_outputs = transformed_model(**inputs)
    output_names = list(pt_outputs.keys())

    # Raise error if expected outputs are not present
    if "logits" not in output_names:
        raise KeyError("logits not found in output")
    if "past_key_values" not in output_names:
        raise KeyError("past_key_values not found in output")

    # Build inputs for next iteration from outputs
    # Build inputs for decode
    inputs = input_handler.update_pytorch_inputs(inputs, pt_outputs)
    # To avoid issues in onnx export
    inputs["position_ids"] = torch.full((full_batch_size if full_batch_size else 1, 1), seq_len - 1)

    # Run PyTorch inference with past
    pt_outputs = transformed_model(**inputs)
    output_names = list(pt_outputs.keys())

    # Add pkv into output_names
    pkv = inputs["past_key_values"]
    pkv_idx = output_names.index("past_key_values")
    key_value_names = [f"past_{x}.{i}" for i in range(len(pkv)) for x in ["key", "value"]]
    output_names[pkv_idx : pkv_idx + 1] = [x + "_RetainedState" for x in key_value_names]

    # Replace nested past_key_values outputs with separate KV tensors
    pt_outputs = dict(pt_outputs)
    pkv_out = pt_outputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv_out):
        pt_outputs[f"past_key.{i}_RetainedState"] = key
        pt_outputs[f"past_value.{i}_RetainedState"] = value

    model_base_name = model_name.replace("/", "_") + "_kv"
    os.makedirs(onnx_dir_path, exist_ok=True)

    # Export and simplify ONNX model
    model_name = export_onnx(
        pt_model=transformed_model,
        inputs=inputs,
        output_names=output_names,
        gen_models_path=onnx_dir_path,
        model_base_name=model_base_name,
    )

    # Replace nested past_key_values inputs with separate KV tensors
    inputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv):
        inputs[f"past_key.{i}"] = key
        inputs[f"past_value.{i}"] = value

    # Run onnxrt inference
    input_names, ort_outputs = run_model_on_ort(
        onnx_path=os.path.join(onnx_dir_path, f"{model_name}.onnx"),
        inputs=inputs,
        output_names=output_names,
        pt_outputs=pt_outputs,
    )

    model_name = fix_onnx_fp16(
        inputs=inputs,
        output_names=output_names,
        ort_outputs=ort_outputs,
        gen_models_path=onnx_dir_path,
        model_base_name=model_name,
        pt_outputs=pt_outputs,
    )

    # Generate custom-IO files for fp16 and int8 kv
    with open(os.path.join(onnx_dir_path, "custom_io_fp16.yaml"), "w") as fp:
        fp.write("# Model Inputs\n\n")
        for input_name in key_value_names:
            fp.write(f" - IOName: {input_name}\n   Precision: float16\n\n")
            inputs[input_name] = inputs[input_name].to(torch.float16)
        fp.write("# Model Outputs\n\n")
        for output_name in key_value_names:
            fp.write(f" - IOName: {output_name}_RetainedState\n   Precision: float16\n\n")

    with open(os.path.join(onnx_dir_path, "custom_io_int8.yaml"), "w") as fp:
        fp.write("# Model Inputs\n\n")
        for input_name in key_value_names:
            fp.write(f" - IOName: {input_name}\n   Precision: mxint8\n\n")
        fp.write("# Model Outputs\n\n")
        for output_name in key_value_names:
            fp.write(f" - IOName: {output_name}_RetainedState\n   Precision: mxint8\n\n")

    # Generate inputFiles
    input_list_file = os.path.join(onnx_dir_path, "input_list.txt")
    generate_input_files(
        input_files_path=os.path.join(onnx_dir_path, "inputFiles"),
        input_names=input_names,
        inputs=inputs,
        input_list_file=input_list_file,
    )

    return model_name


def export_lm_model_for_cloud(
    model_name: str,
    qeff_model: QEFFAutoModelForCausalLM,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_length: int,
    full_batch_size: Optional[int] = None,
) -> str:
    if os.path.exists(onnx_dir_path):
        logger.warning(f"Overriding {onnx_dir_path}")
        shutil.rmtree(onnx_dir_path)

    model_name = export_kvstyle_transformed_model_to_onnx(
        model_name=model_name,
        transformed_model=qeff_model.model,
        tokenizer=tokenizer,
        onnx_dir_path=onnx_dir_path,
        seq_len=seq_length,
        full_batch_size=full_batch_size,
    )
    return os.path.join(onnx_dir_path, f"{model_name}.onnx")


def qualcomm_efficient_converter(
    model_name: str,
    model_kv: QEFFBaseModel = None,  # type: ignore
    local_model_dir: Optional[str] = None,
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    cache_dir: Optional[str] = None,
    onnx_dir_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    seq_length: int = Constants.SEQ_LEN,
    kv: bool = True,
    form_factor: str = "cloud",
    full_batch_size: Optional[int] = None,
) -> Tuple[str, str]:
    """
    This method is an alias for ``QEfficient.export``.

    Usage 1: This method can be used by passing ``model_name`` and ``local_model_dir`` or ``cache_dir`` if required for loading from local dir.
    This will download the model from ``HuggingFace`` and export it to ``ONNX`` graph and returns generated files path check below.

    Usage 2: You can pass ``model_name`` and ``model_kv`` as an object of ``QEfficient.QEFFAutoModelForCausalLM``, In this case will directly export the ``model_kv.model`` to ``ONNX``

    We will be deprecating this function and it will be replaced by ``QEFFAutoModelForCausalLM.export``.

    ``Mandatory`` Args:
        :model_name (str): The name of the model to be used.
    ``Optional`` Args:
        :model_kv (torch.nn.Module): Transformed ``KV torch model`` to be used. ``Defaults to None``.
        :local_model_dir (str): Path of local model. ``Defaults to None``.
        :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Model tokenizer. ``Defaults to None``.
        :cache_dir (str): Path of the ``cache`` directory. ``Defaults to None``.
        :onnx_dir_path (str): Path to store ``ONNX`` file. ``Defaults to None``.
        :hf_token (str): HuggingFace token to access gated models. ``Defaults is None``.
        :seq_len (int): The length of the sequence. ``Defaults is 128``.
        :kv (bool): If false, it will export to Bert style. ``Defaults is True``.
        :form_factor (str): Form factor of the hardware, currently only ``cloud`` is accepted. ``Defaults to cloud``.

    Returns:
        :Tuple[str, str]: Path to Base ``ONNX`` dir and path to generated ``ONNX`` model

    .. code-block:: python

        import QEfficient
        base_path, onnx_model_path = QEfficient.export(model_name="gpt2")

    """
    warnings.warn(
        "\033[93m`qualcomm_efficient_converter` method will be deprecated soon, use `QEFFAutoModelForCausalLM.export` instead\033[0m",
        DeprecationWarning,
        stacklevel=2,
    )

    # Get model_kv first
    model_kv = (
        model_kv
        if model_kv
        else QEFFCommonLoader.from_pretrained(
            pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
            hf_token=hf_token,
            cache_dir=cache_dir,
            full_batch_size=full_batch_size,
        )
    )

    if onnx_dir_path is None:
        model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
        onnx_dir_path = os.path.join(model_card_dir, "onnx")
        os.makedirs(onnx_dir_path, exist_ok=True)

    # Load tokenizer if not passed
    tokenizer = (
        tokenizer
        if tokenizer
        else load_hf_tokenizer(
            pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
            hf_token=hf_token,
            cache_dir=cache_dir,
        )
    )

    if form_factor == "cloud":
        generated_onnx_model_path = export_lm_model_for_cloud(
            model_name=model_name,
            qeff_model=model_kv,
            tokenizer=tokenizer,
            onnx_dir_path=onnx_dir_path,
            seq_length=seq_length,
            full_batch_size=full_batch_size,
        )
        return onnx_dir_path, generated_onnx_model_path
    else:
        # [TODO]: Apply the class transformation to make changes for the KV models in edge use cases
        # model = QEfficient.transform(model_hf, type="Transformers", form_factor="edge")
        # model.eval()
        raise NotImplementedError("Oops! Reached too far!!")
