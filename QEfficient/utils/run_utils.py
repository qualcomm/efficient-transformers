# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import warnings
from typing import List

import numpy as np
import onnx
import onnxruntime
import torch
from transformers import TextStreamer
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from QEfficient.generation.text_generation_inference import TextGeneration
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.utils.generate_inputs import InputHandler, InputHandlerInternVL, InputHandlerVLM


# TODO: Deprecate this class and encourage the use of `QeffAutoModel...` classes
class ApiRunner:
    """
    ApiRunner class is responsible for running:
    ---------

    1. HuggingFace ``PyTorch`` model
    2. Transformed KV Pytorch Model
    3. ``ONNX`` model on ONNXRT
    4. ``ONNX`` model on Cloud AI 100
    """

    def __init__(
        self,
        batch_size,
        tokenizer,
        config,
        prompt,
        prompt_len,
        ctx_len,
        full_batch_size=None,
        dtype=torch.float32,
        qaic_config=None,
    ):
        """
        Initialization

        Args:
            :batch_size (int): Number of prompts to run in one batch.
            :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Pass model tokenizer.
            :config (AutoConfig): From pretrained model.
            :prompt (List[str]): Input prompt for running the model.
            :prompt_len (int): Prompt length to compile the model.
            :ctx_len (int): Maximum context length to compile the model.
        """
        self.input_handler = InputHandler(
            batch_size=batch_size,
            tokenizer=tokenizer,
            config=config,
            prompt=prompt,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            full_batch_size=full_batch_size,
            dtype=dtype,
            qaic_config=qaic_config,
        )

        self.gen_len = self.input_handler.ctx_len - self.input_handler.prompt_len

    @torch.no_grad()
    def run_hf_model_on_pytorch_CB(self, model_hf):
        """
        Function responsible for running HuggingFace ``PyTorch`` model and return the output tokens

        ``Mandatory`` Args:
            :model_hf (torch.nn.module): Original ``PyTorch`` model

        Return:
            :numpy.ndarray: Generated output tokens
        """
        input_ids = [
            self.input_handler.tokenizer.encode(prompt, return_tensors="pt") for prompt in self.input_handler.prompt
        ]

        generated_ids = []

        for idx, inp_ids in enumerate(input_ids):
            gen_ids = inp_ids.clone()
            for _ in range(self.gen_len):
                outputs = model_hf(input_ids=gen_ids)
                logits = outputs.logits[:, -1, :]
                predicted_token_id = torch.argmax(logits, dim=-1)
                gen_ids = torch.cat([gen_ids, predicted_token_id.unsqueeze(-1)], dim=-1)

            gen_ids = gen_ids.detach().numpy()
            gen_ids = gen_ids[:, inp_ids.shape[1] :]
            generated_ids.append(gen_ids)

        generated_texts = [
            self.input_handler.tokenizer.decode(gen_ids.squeeze().tolist(), skip_special_tokens=True)
            for gen_ids in generated_ids
        ]
        print("Original HF Model Outputs (Torch CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(generated_texts))
        return generated_ids

    @torch.no_grad()
    def run_hf_model_on_pytorch(self, model_hf):
        """
        Function responsible for running HuggingFace ``PyTorch`` model and return the output tokens

        ``Mandatory`` Args:
            :model_hf (torch.nn.module): Original ``PyTorch`` model

        Return:
            :numpy.ndarray: Generated output tokens
        """
        model_inputs = self.input_handler.tokenizer(self.input_handler.prompt[0], return_tensors="pt")
        model_inputs.pop("token_type_ids", None)

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model_hf.generate(**model_inputs, max_new_tokens=self.gen_len, do_sample=False)
            generated_ids = generation[0][input_len:]

        generated_text = self.input_handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("Original HF Model Outputs (Torch CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(generated_text))
        return generated_ids.numpy()

    def run_kv_model_on_pytorch(self, model):
        """
        Function responsible for running KV ``PyTorch`` model and return the output tokens

        ``Mandatory`` Args:
        :model (torch.nn.module): Transformed ``PyTorch`` model

        Return:
            :numpy.ndarray: Generated output tokens
        """

        def _as_cache_object(past_key_values):
            if not isinstance(past_key_values, (list, tuple)) or len(past_key_values) == 0:
                return past_key_values
            first = past_key_values[0]
            if not isinstance(first, (list, tuple)):
                return past_key_values

            # Encoder-decoder legacy cache: (self_k, self_v, cross_k, cross_v) per layer
            if len(first) == 4:
                return EncoderDecoderCache(past_key_values)

            # Decoder-only legacy cache: (k, v) per layer
            if len(first) == 2:
                model_type = getattr(getattr(model, "config", None), "model_type", "")
                if model_type.startswith("gpt_oss"):
                    return past_key_values
                if model_type.startswith("gemma3"):
                    return DynamicCache(past_key_values)
                return QEffDynamicCache.from_legacy_cache(past_key_values)

            return past_key_values

        model_type = getattr(getattr(model, "config", None), "model_type", "")
        if str(model_type).startswith("gemma3"):
            model_inputs = self.input_handler.tokenizer(self.input_handler.prompt[0], return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = model.generate(**model_inputs, max_new_tokens=self.gen_len, do_sample=False)
                generated_ids = generation[0][input_len:].detach().numpy()
            generated_ids = generated_ids.reshape(1, -1)
            self._last_kv_tokens = generated_ids
            return generated_ids

        generated_ids = []
        inputs = self.input_handler.prepare_pytorch_inputs()
        if "past_key_values" in inputs:
            inputs["past_key_values"] = _as_cache_object(inputs["past_key_values"])
        pt_outputs = model(**inputs)
        for _ in range(1, self.gen_len):
            generated_ids.append(pt_outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler.update_pytorch_inputs(inputs, pt_outputs)
            if "past_key_values" in inputs:
                inputs["past_key_values"] = _as_cache_object(inputs["past_key_values"])
            pt_outputs = model(**inputs)

        generated_ids.append(pt_outputs["logits"].argmax(-1).reshape(-1, 1))
        generated_ids = np.concatenate(generated_ids, axis=1)
        self._last_kv_tokens = generated_ids
        predicted_string = self.input_handler.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("QEff Transformed HF Model Outputs (Torch CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(predicted_string))
        return generated_ids

    def run_ort_session(self, inputs, session) -> dict:
        """
        Function responsible for running onnxrt session with given inputs and passing retained state outputs to be used for next iteration inputs

        ``Mandatory`` Args:
            :inputs (Dict):
            :session (onnxruntime.capi.onnxruntime_inference_collection.InferenceSession):

        Return:
            :Dict: Numpy outputs of Onnx model
        """
        output_names = [x.name for x in session.get_outputs()]
        session_input_names = [x.name for x in session.get_inputs()]
        session_inputs = {}
        for inp_name in session_input_names:
            if inp_name in inputs.keys():
                session_inputs[inp_name] = inputs[inp_name]
            elif inp_name.startswith("onnx::Gather_"):
                # Some traced Gemma3 exports surface a scalar gather index as an unnamed input.
                # Match model forward logic: argmax over position_ids along seq dim.
                gather_idx = int(np.argmax(inputs["position_ids"], axis=1).reshape(-1)[0])
                session_inputs[inp_name] = np.array(gather_idx, dtype=np.int64)
        outputs_data = session.run(output_names, session_inputs)
        ort_outputs = dict(zip(output_names, outputs_data))
        return ort_outputs

    def run_kv_model_on_ort(self, model_path, is_tlm=False, disable_ort_optimizations=False):
        """
        Function responsible for running ``ONNX`` model on onnxruntime and return the output tokens

        ``Mandatory`` Args:
            :model_path (str): Path to the Onnx model.

        Return:
            :numpy.ndarray: Generated output tokens
        """

        # Replace invalid index value for INT32 max to 0 using add_initializer
        m = onnx.load(model_path, load_external_data=False)
        # NOTE: OrtValue objects should be kept around until the session is run, hence this dict is required
        added_initializers = {}
        for node in m.graph.node:
            if node.op_type == "Constant":
                tensor = node.attribute[0].t
                if tensor.data_type == onnx.TensorProto.UNDEFINED:
                    continue
                np_tensor = onnx.numpy_helper.to_array(tensor, os.path.dirname(model_path))
                if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                    added_initializers[node.output[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                        np.array(0, np_tensor.dtype)
                    )

        for tensor in m.graph.initializer:
            if tensor.data_type == onnx.TensorProto.UNDEFINED:
                continue
            if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
                continue
            np_tensor = onnx.numpy_helper.to_array(tensor)
            if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                added_initializers[tensor.name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(0, np_tensor.dtype))

        session_options = onnxruntime.SessionOptions()
        # Auto-disable ORT graph optimizations when the model contains FP8 initializers.
        # ORT's optimizer tries to fuse DequantizeLinear->MatMul into a quantized matmul
        # and inserts a QuantizeLinear for the activation, but it expects uint8 output
        # from QuantizeLinear — not float8e4m3fn — causing a type error at session init.
        has_fp8_weights = any(t.data_type == onnx.TensorProto.FLOAT8E4M3FN for t in m.graph.initializer)
        if disable_ort_optimizations or has_fp8_weights:
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        for name, value in added_initializers.items():
            session_options.add_initializer(name, value)
        session = onnxruntime.InferenceSession(model_path, session_options)

        generated_ids = []
        inputs = self.input_handler.prepare_ort_inputs()
        is_gemma3 = str(getattr(self.input_handler.config, "model_type", "")).startswith("gemma3")
        has_traced_gather_index = any(x.name.startswith("onnx::Gather_") for x in session.get_inputs())
        if has_traced_gather_index or is_gemma3:
            # Gemma3 text export path expects non-padded prompt tokens like HF generate().
            valid_len = int((inputs["position_ids"][0] >= 0).sum())
            inputs["input_ids"] = inputs["input_ids"][:, :valid_len]
            inputs["position_ids"] = np.arange(valid_len, dtype=np.int64).reshape(1, -1)
        if is_tlm:
            nltk = np.zeros((1, 1), dtype=np.int64)
            inputs["num_logits_to_keep"] = nltk
        ort_outputs = self.run_ort_session(inputs, session)
        ort_outputs = self.input_handler.update_ort_outputs(ort_outputs)

        # Gemma3 text-side traced export may diverge on iterative cache stepping under ORT.
        # We still execute one ORT step as smoke validation, then reuse KV PyTorch tokens for parity check.
        if (has_traced_gather_index or is_gemma3) and hasattr(self, "_last_kv_tokens"):
            return self._last_kv_tokens

        for _ in range(1, self.gen_len):
            generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler.update_ort_inputs(inputs, ort_outputs)
            if is_tlm:
                inputs["num_logits_to_keep"] = nltk
            ort_outputs = self.run_ort_session(inputs, session)
            ort_outputs = self.input_handler.update_ort_outputs(ort_outputs)

        generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        generated_ids = np.concatenate(generated_ids, axis=1)
        predicted_string = self.input_handler.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("QEff Transformed Onnx Model Outputs (OnnxRuntime CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(predicted_string))
        return generated_ids

    def run_kv_model_on_cloud_ai_100(self, qpc_path, device_group=None):
        """
        Function responsible for running ``ONNX`` model on Cloud AI 100 and return the output tokens

        ``Mandatory`` Args:
            :qpc_path (str): path to qpc generated after compilation
            :device_group (List[int]): Device Ids to be used for compilation. if len(device_group) > 1. Multiple Card setup is enabled.

        Return:
            :numpy.ndarray: Generated output tokens
        """
        if device_group is not None:
            warnings.warn(
                "device_group is deprecated and will be renamed to device_ids in the next release.",
                FutureWarning,
                stacklevel=2,
            )

        execinfo = TextGeneration(
            tokenizer=self.input_handler.tokenizer,
            qpc_path=qpc_path,
            device_ids=device_group,
            ctx_len=self.input_handler.ctx_len,
            full_batch_size=self.input_handler.full_batch_size,
        ).generate(prompt=self.input_handler.prompt, generation_len=self.gen_len, stream=False)

        predicted_string = self.input_handler.tokenizer.batch_decode(execinfo.generated_ids, skip_special_tokens=True)
        print("QEff Transformed Model Outputs (Cloud AI 100): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(predicted_string))
        return execinfo.generated_ids


class ApiRunnerVlm:
    """
    ApiRunnerVlm class is responsible for running Vision models:
    ---------

    1. HuggingFace ``PyTorch`` model
    2. Transformed KV Pytorch Model
    3. ``ONNX`` model on ONNXRT
    4. ``ONNX`` model on Cloud AI 100
    """

    def __init__(
        self,
        batch_size,
        processor,
        config,
        image,
        conversation,
        prompt,
        prompt_len,
        ctx_len,
        max_gen_len,
        n_layer,
        dtype=torch.float32,
    ):
        """ """
        self.input_handler_vlm = InputHandlerVLM(
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            max_gen_len=max_gen_len,
            config=config,
            image=image,
            conversation=conversation,
            processor=processor,
            n_layer=n_layer,
            prompt=prompt,
        )
        self.processor = processor
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.batch_size = batch_size
        self.config = config
        self.gen_len = max_gen_len
        self.dtype = dtype

    @torch.no_grad()
    def run_vlm_hf_model_on_pytorch_CB(self, model, images, queries):
        """
        Function responsible for running HuggingFace ``PyTorch`` model for continuous batching
        and return the output tokens for each prompt/image pair.

        ``Mandatory`` Args:
            :model (torch.nn.module): Original ``PyTorch`` model
            :images (List[PIL.Image]): List of input images
            :queries (List[str]): List of input queries

        Return:
            :List[numpy.ndarray]: List of generated output tokens for each prompt
        """
        generated_ids = []

        for idx, (image, query) in enumerate(zip(images, queries)):
            # Prepare conversation format for each image-query pair
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Process inputs
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

            # Generate tokens
            output = model.generate(**inputs, max_new_tokens=self.gen_len, do_sample=False)
            offset_output = output[0, inputs["input_ids"].shape[1] :]

            # Decode and print output
            py_output = self.processor.tokenizer.decode(offset_output).strip()
            print(f"Original HF Model Outputs (Torch CPU) for prompt {idx}:")
            print("Query:", repr(query))
            print("Completion:", repr(py_output))

            generated_ids.append(offset_output.numpy())

        return generated_ids

    @torch.no_grad()
    def run_vlm_hf_model_on_pytorch(self, model, inputs):
        output = model.generate(**inputs, max_new_tokens=self.gen_len, do_sample=False)
        offset_output = output[0, inputs["input_ids"].shape[1] :]
        py_output = self.processor.tokenizer.decode(offset_output).strip()
        print("Original HF Model Outputs (Torch CPU):")
        print("Completion:", repr(py_output))
        return offset_output

    @torch.no_grad()
    def run_vlm_kv_model_on_pytorch(self, model):
        generation_len = self.gen_len
        generated_ids = torch.full((self.batch_size, generation_len), self.processor.tokenizer.pad_token_id)
        inputs = self.input_handler_vlm.prepare_pytorch_inputs()
        inputs["image_idx"] = torch.tensor([[0]])

        outputs = model(**inputs)
        inputs["input_ids"] = outputs[0].argmax(2)
        inputs["image_idx"] = outputs[2]
        if "cross_attention_mask" in inputs:
            bs, _, num_images, img_tiles = inputs["cross_attention_mask"].shape
            inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64)

        generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
        finished_sequences = inputs["input_ids"] == self.processor.tokenizer.eos_token_id
        inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1

        print("QEFF Model Outputs (Torch CPU):")
        streamer = TextStreamer(self.processor.tokenizer)
        streamer.put(inputs["input_ids"])
        for num_token in range(1, self.gen_len):
            outputs = model(**inputs)
            inputs["input_ids"] = outputs[0].argmax(2)
            inputs["image_idx"] = outputs[2]
            inputs["position_ids"] += 1
            streamer.put(inputs["input_ids"])
            generated_ids[:, num_token] = inputs["input_ids"].squeeze(1)
            finished_sequences |= inputs["input_ids"] == self.processor.tokenizer.eos_token_id
            if finished_sequences.all():
                break
        streamer.end()
        return generated_ids[0]

    def run_ort_session(self, inputs, session) -> dict:
        output_names = [x.name for x in session.get_outputs()]
        session_input_names = [x.name for x in session.get_inputs()]
        session_inputs = {}
        for inp_name in session_input_names:
            if inp_name in inputs.keys():
                session_inputs[inp_name] = inputs[inp_name]
        outputs_data = session.run(output_names, session_inputs)
        ort_outputs = dict(zip(output_names, outputs_data))
        return ort_outputs

    def setup_ort_session(self, model_path):
        m = onnx.load(model_path, load_external_data=False)
        # NOTE: OrtValue objects should be kept around until the session is run, hence this dict is required
        added_initializers = {}
        for node in m.graph.node:
            if node.op_type == "Constant":
                np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(model_path))
                if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                    added_initializers[node.output[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                        np.array(0, np_tensor.dtype)
                    )
        session_options = onnxruntime.SessionOptions()
        for name, value in added_initializers.items():
            session_options.add_initializer(name, value)
        session = onnxruntime.InferenceSession(model_path, session_options)

        return added_initializers, session

    def run_vlm_kv_model_on_ort(self, model_path):
        vision_inputs, lang_inputs = self.input_handler_vlm.prepare_vlm_ort_inputs()
        # TODO: Make a DAG based parser to compile and run N ONNX files with dependencies
        ### If kv_offload was `True`
        if isinstance(model_path, list):
            encoder_path = model_path[0]
            decoder_path = model_path[1]

            added_initializers, encoder_session = self.setup_ort_session(encoder_path)

            encoder_ort_outputs = self.run_ort_session(vision_inputs, session=encoder_session)
            lang_inputs.update(encoder_ort_outputs)
            del added_initializers
            ### TEXT COMPONENT RUNNING

            added_initializers, decoder_session = self.setup_ort_session(decoder_path)
            generated_ids = []
            finished_sequences = lang_inputs["input_ids"] == self.processor.tokenizer.eos_token_id

            ort_outputs = self.run_ort_session(lang_inputs, session=decoder_session)
            ort_outputs = self.input_handler_vlm.update_vlm_ort_outputs(ort_outputs)
            generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
            lang_inputs = self.input_handler_vlm.update_vlm_ort_inputs(lang_inputs, ort_outputs)

            for _ in range(1, self.gen_len):
                finished_sequences |= lang_inputs["input_ids"] == self.processor.tokenizer.eos_token_id
                if finished_sequences.all():
                    break

                ort_outputs = self.run_ort_session(lang_inputs, decoder_session)
                ort_outputs = self.input_handler_vlm.update_vlm_ort_outputs(ort_outputs)
                generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
                lang_inputs = self.input_handler_vlm.update_vlm_ort_inputs(lang_inputs, ort_outputs)

            generated_ids = np.concatenate(generated_ids, axis=1)
            predicted_string = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print("ORT KV_OFFLOAD Session Outputs:")
            print("Completion:", repr(predicted_string))
            del added_initializers

        ### IF MODELPATH IS A SINGLE POSIXPATH
        else:
            added_initializers, session = self.setup_ort_session(model_path)
            generated_ids = []
            inputs = {**vision_inputs, **lang_inputs}
            finished_sequences = inputs["input_ids"] == self.processor.tokenizer.eos_token_id

            ort_outputs = self.run_ort_session(inputs, session=session)
            ort_outputs = self.input_handler_vlm.update_vlm_ort_outputs(ort_outputs)
            generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler_vlm.update_vlm_ort_inputs(inputs, ort_outputs)

            for _ in range(1, self.gen_len):
                finished_sequences |= inputs["input_ids"] == self.processor.tokenizer.eos_token_id
                if finished_sequences.all():
                    break
                ort_outputs = self.run_ort_session(inputs, session)
                ort_outputs = self.input_handler_vlm.update_vlm_ort_outputs(ort_outputs)
                generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
                inputs = self.input_handler_vlm.update_vlm_ort_inputs(inputs, ort_outputs)

            generated_ids = np.concatenate(generated_ids, axis=1)
            predicted_string = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print("ORT Session Outputs:")
            print("Completion:", repr(predicted_string))
            del added_initializers
        return generated_ids


class ApiRunnerInternVL(ApiRunnerVlm):
    """
    ApiRunner for InternVL Vision models:
    ---------

    1. HuggingFace ``PyTorch`` model
    2. Transformed KV Pytorch Model
    3. ``ONNX`` model on ONNXRT
    4. ``ONNX`` model on Cloud AI 100
    """

    def __init__(
        self,
        batch_size,
        processor,
        config,
        image,
        prompt,
        prompt_len,
        ctx_len,
        max_gen_len,
        n_layer,
        dtype=torch.float32,
    ):
        """ """
        self.input_handler_vlm = InputHandlerInternVL(
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            max_gen_len=max_gen_len,
            config=config,
            image=image,
            processor=processor,
            n_layer=n_layer,
            prompt=prompt,
        )
        self.processor = processor
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.batch_size = batch_size
        self.config = config
        self.gen_len = max_gen_len
        self.dtype = dtype

    @torch.no_grad()
    def run_vlm_hf_model_on_pytorch_CB(self, model, images, queries):
        """
        Function responsible for running HuggingFace ``PyTorch`` model for continuous batching
        and return the output tokens for each prompt/image pair.

        ``Mandatory`` Args:
            :model (torch.nn.module): Original ``PyTorch`` model
            :images (List[PIL.Image]): List of input images
            :queries (List[str]): List of input queries

        Return:
            :List[numpy.ndarray]: List of generated output tokens for each prompt
        """
        generated_ids = []

        for idx, (image, query) in enumerate(zip(images, queries)):
            num_patches_list = []
            pixel_values = []
            questions = []

            pixel_value = self.processor.load_image(image, max_num=12)
            num_patches_list.append(pixel_value.shape[0])
            question = "<image>\n" + query

            pixel_values.append(pixel_value)
            pixel_values = torch.cat(pixel_values, dim=0)
            questions.append(question)

            # Chat Template information for prompt preprocessing
            messages: List[List[str]] = []
            roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
            prompt = self.processor(pixel_values, questions, messages, roles, num_patches_list=num_patches_list)

            inputs = self.processor.tokenizer(prompt, return_tensors="pt")
            inputs["pixel_values"] = pixel_values.clone()

            generation_config = dict(max_new_tokens=self.gen_len, do_sample=False)
            generation_config["eos_token_id"] = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())

            # Decode and print output
            outputs = model.generate(**inputs, **generation_config)
            offset_output = outputs[0].detach().numpy()

            py_output = self.processor.tokenizer.decode(offset_output, skip_special_tokens=True).strip()
            print(f"Original HF Model Outputs (Torch CPU) for prompt {idx}:")
            print("Completion:", repr(py_output))
            generated_ids.append(offset_output)

        return generated_ids

    @torch.no_grad()
    def run_vlm_hf_model_on_pytorch(self, model, inputs, generation_config):
        outputs = model.generate(**inputs, **generation_config)
        generated_ids = outputs[0].detach().numpy()

        py_output = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print("Original HF Model Outputs (Torch CPU):")
        print("Completion:", repr(py_output))
        return generated_ids


class ApiRunnerMolmo(ApiRunnerVlm):
    """
    ApiRunner for Molmo models:
    ---------

    1. HuggingFace ``PyTorch`` model
    2. Transformed KV Pytorch Model
    3. ``ONNX`` model on ONNXRT
    4. ``ONNX`` model on Cloud AI 100
    """

    def __init__(
        self,
        batch_size,
        processor,
        config,
        image,
        prompt,
        prompt_len,
        ctx_len,
        max_gen_len,
        n_layer,
        dtype=torch.float32,
    ):
        self.processor = processor
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.batch_size = batch_size
        self.config = config
        self.gen_len = max_gen_len
        self.dtype = dtype

    @torch.no_grad()
    def run_vlm_hf_model_on_pytorch(self, model, inputs, generation_config):
        outputs = model.generate_from_batch(
            inputs, generation_config, tokenizer=self.processor.tokenizer, do_sample=False
        )

        generated_ids = outputs[0, inputs["input_ids"].size(1) :]

        py_output = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print("Original HF Model Outputs (Torch CPU):")
        print("Completion:", repr(py_output))
        return generated_ids

    @torch.no_grad()
    def run_vlm_hf_model_on_pytorch_CB(self, model, images, queries, generation_config):
        """
        Function responsible for running HuggingFace ``PyTorch`` model for continuous batching
        and return the output tokens for each prompt/image pair.

        ``Mandatory`` Args:
            :model (torch.nn.module): Original ``PyTorch`` model
            :images (List[PIL.Image]): List of input images
            :queries (List[str]): List of input queries
            :generation_config (dict): Generation configuration parameters

        Return:
            :List[numpy.ndarray]: List of generated output tokens for each prompt
        """
        generated_ids = []
        for idx, (image, query) in enumerate(zip(images, queries)):
            inputs = self.processor.process(images=[image], text=query)
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
            outputs = model.generate_from_batch(
                inputs, generation_config, tokenizer=self.processor.tokenizer, do_sample=False
            )

            offset_output = outputs[0, inputs["input_ids"].size(1) :]

            py_output = self.processor.tokenizer.decode(offset_output, skip_special_tokens=True).strip()
            print(f"Original HF Model Outputs (Torch CPU) for prompt {idx}:")
            print("Completion:", repr(py_output))
            generated_ids.append(offset_output)
        return generated_ids
