# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import time

import numpy as np
import onnxruntime
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import QEFFLogger

logger = QEFFLogger.get_logger("INFRA")


# 1. Data Loading
class WikiTextDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, ctx_len, stride=1024):
        self.dataset = load_dataset("wikitext", dataset_name, split="test")
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.stride = stride
        self.encodings = tokenizer("\n\n".join(self.dataset["text"]), return_tensors="np")
        self.input_ids, self.attention_mask = self._prepare_inputs()

    def _prepare_inputs(self):
        seq_len = self.encodings.input_ids.shape[1]
        input_ids = []
        attention_mask = []
        for i in range(0, seq_len - self.ctx_len + 1, self.stride):
            input_ids.append(self.encodings.input_ids[:, i : i + self.ctx_len])
            attention_mask.append(self.encodings.attention_mask[:, i : i + self.ctx_len])
        return input_ids, attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]


class WikiTextDataLoader:
    def __init__(self, dataset_name, tokenizer_name, ctx_len, batch_size, stride=1024, num_samples=-1):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="right")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.dataset = WikiTextDataset(dataset_name, self.tokenizer, ctx_len, stride)
        if num_samples > 0:
            self.dataset.input_ids = self.dataset.input_ids[:num_samples]
            self.dataset.attention_mask = self.dataset.attention_mask[:num_samples]
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        input_ids = np.concatenate([item[0] for item in batch], axis=0)
        attention_mask = np.concatenate([item[1] for item in batch], axis=0)
        return input_ids, attention_mask

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)


# 2. Model Inference
class InferenceSession:
    def run(self, inputs):
        raise NotImplementedError


class ONNXInferenceSession(InferenceSession):
    def __init__(self, model_path):
        logger.info(f"Creating ONNX session from {model_path}")
        providers = ["CPUExecutionProvider"]
        if onnxruntime and onnxruntime.get_device() == "GPU":
            providers = [("CUDAExecutionProvider", {"device_id": 0})] + providers
        self.session = onnxruntime.InferenceSession(model_path)
        self.n_layer = self._get_n_layer()

    def _get_n_layer(self):
        input_names = [input.name for input in self.session.get_inputs()]
        return sum(1 for name in input_names if name.startswith("past_key."))

    def _input_output_specs(self):
        input_names = [x.name for x in self.session.get_inputs()]
        input_shapes = [x.shape for x in self.session.get_inputs()]
        output_names = [x.name for x in self.session.get_outputs()]
        return input_names, input_shapes, output_names

    def run(self, inputs):
        # onnx_inputs = self._prepare_onnx_inputs(inputs)
        outputs = self.session.run(None, {k: v for k, v in inputs.items()})
        # return self._process_outputs(outputs)
        return outputs

    def _prepare_onnx_inputs(self, inputs):
        onnx_inputs = inputs.copy()
        past_key_values = []
        for i in range(self.n_layer):
            past_key = np.zeros(self.padding_shape, dtype=np.float32)
            past_value = np.zeros(self.padding_shape, dtype=np.float32)
            past_key_values.append((past_key, past_value))

        for i, (past_key, past_value) in enumerate(past_key_values):
            onnx_inputs[f"past_key.{i}"] = past_key
            onnx_inputs[f"past_value.{i}"] = past_value

        return onnx_inputs

    def _process_outputs(self, outputs):
        # Assuming the first output is logits and the rest are past_key_values
        logits = outputs[0]
        past_key_values = outputs[1:]
        return {"logits": logits, "past_key_values": past_key_values}


class QPCInferenceSession(InferenceSession):
    def __init__(self, model_path, device_id):
        self.session = QAICInferenceSession(model_path, device_id)
        self._skip_buffers()

    def _skip_buffers(self):
        self.session.skip_buffers([x for x in self.session.input_names if x.startswith("past_")])
        self.session.skip_buffers([x for x in self.session.output_names if x.endswith("_RetainedState")])

    def run(self, inputs):
        return self.session.run(inputs)


class PytorchInferenceSession(InferenceSession):
    def __init__(self, model_name, device=None):
        from QEfficient.utils._utils import login_and_download_hf_lm

        model_name = login_and_download_hf_lm(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.eval()

    def run(self, **inputs):
        return self.model(**inputs)


def generate_tokens(ctx_len, input_len, cache_index, model_type):
    if isinstance(model_type, str) and model_type == "qpc":
        for num_token in range(1, (ctx_len - input_len.max()) + 1):
            yield num_token
    else:
        while cache_index < ctx_len:
            yield cache_index


def to_torch(inputs):
    return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()}


def torch_perplexity(
    test_data_loader, batch_size, inference_session, loss, log_file, cnt, prompt_len, ctx_len, model_name
):
    loss_list = []
    for inp_ids, attn_mask in tqdm(test_data_loader):
        loop_s = time.time()
        pad_input_batch = None
        if inp_ids.shape[0] != batch_size and attn_mask.shape[0] != batch_size:
            pad_input_batch = batch_size - inp_ids.shape[0]
            for _ in range(pad_input_batch):
                inp_ids = torch.cat((inp_ids, inp_ids[-1, :][None, :]), axis=0)
                attn_mask = torch.cat((attn_mask, attn_mask[-1, :][None, :]), axis=0)

        inputs = {}

        inputs["input_ids"] = inp_ids

        # if "opt" not in model_name:
        #     inputs["position_ids"] = torch.tensor([i for i in range(ctx_len)])

        inputs["attention_mask"] = attn_mask

        inputs = to_torch(inputs)
        with torch.no_grad():
            outputs = inference_session.run(**inputs)

        if pad_input_batch is not None:
            outputs["logits"] = outputs["logits"][: batch_size - pad_input_batch]
            inputs["input_ids"] = inputs["input_ids"][: batch_size - pad_input_batch]

        for idx in range(outputs["logits"].shape[0]):
            op_loss = loss(outputs["logits"][idx][:-1, :], inputs["input_ids"][idx][1:])
            with open(log_file, "a") as fp:
                fp.write(f"Sample No:{cnt} \t AVG_LOSS: {op_loss.mean().item():.4f}\n")
            loss_list.append(op_loss)
            cnt += 1

        loop_time = time.time() - loop_s
        logger.info(
            f"E2E Sample Time: {(loop_time) / batch_size:.4f}s\t E2E TOKENS/S : {((ctx_len - prompt_len) * batch_size) / loop_time:.2f}"
        )

        del outputs

    return loss_list


# 3. Perplexity Calculation
def calculate_perplexity(
    data_loader, inference_session, ctx_len, prompt_len, model_type, batch_size, log_file, model_name
):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    total_loss = 0
    total_tokens = 0
    with open(log_file, "w") as fp:
        cnt = 0

    is_qaic = False
    is_torch = False
    if isinstance(model_type, str) and model_type == "qpc":
        is_qaic = True
    elif isinstance(model_type, str) and model_type == "torch":
        is_torch = True

    if is_torch:
        loss_list = torch_perplexity(
            data_loader, batch_size, inference_session, loss_fct, log_file, cnt, prompt_len, ctx_len, model_name
        )
        avg_loss = torch.stack(loss_list).mean()
        perplexity = np.exp(avg_loss)
        return perplexity, avg_loss

    else:
        for inp_ids, attn_mask in tqdm(data_loader):
            loop_s = time.time()
            pad_input_batch = None
            if inp_ids.shape[0] != batch_size and attn_mask.shape[0] != batch_size:
                pad_input_batch = batch_size - inp_ids.shape[0]
                inp_ids = np.concatenate((inp_ids, np.repeat(inp_ids[-1:], pad_input_batch, axis=0)))
                attn_mask = np.concatenate((attn_mask, np.repeat(attn_mask[-1:], pad_input_batch, axis=0)))

            inputs = {}
            targets_label = []
            outputs_logits = []
            input_ids = inp_ids[:, :prompt_len]
            attention_mask = attn_mask[:, :prompt_len]
            input_len = attention_mask.sum(1, keepdims=True)
            padded_len = input_ids.shape[1]

            inputs["input_ids"] = input_ids
            inputs["position_ids"] = np.where(attention_mask, np.arange(padded_len), -1)
            cache_index = np.array([0])

            # Run inference
            if is_qaic:
                outputs = inference_session.run(inputs)
                if len(outputs["logits"].shape) == 2:
                    outputs["logits"] = np.expand_dims(outputs["logits"], axis=1)
                if outputs["logits"].dtype == np.float16:
                    outputs["logits"] = outputs["logits"].astype(np.float32)
                outputs_logits.append(torch.tensor(outputs["logits"]))

                cache_index += 1
                inputs["input_ids"] = inp_ids[:, cache_index]
                targets_label.append(torch.from_numpy(inputs["input_ids"]))
                inputs["position_ids"] = input_len
            else:
                outputs = {}
                input_names, input_shapes, output_names = inference_session._input_output_specs()
                for i, iname in enumerate(input_names):
                    if "past" in iname:
                        inputs[iname] = np.zeros(
                            (1, input_shapes[i][1], prompt_len, input_shapes[i][3]), dtype="float32"
                        )
                cache_index += 1

            for token in generate_tokens(ctx_len, input_len, cache_index, model_type):
                if is_qaic:
                    outputs = inference_session.run(inputs)
                else:
                    outputs_ort = inference_session.run(inputs)
                    for i, oname in enumerate(output_names):
                        outputs[oname] = outputs_ort[i]

                logits = outputs["logits"]
                if len(logits.shape) == 2:
                    logits = np.expand_dims(logits, axis=1)
                if logits.dtype == np.float16:
                    logits = logits.astype(np.float32)

                cache_index += 1
                if cache_index < ctx_len:
                    inputs["input_ids"] = inp_ids[:, cache_index]
                    targets_label.append(torch.from_numpy(inputs["input_ids"]))
                    inputs["position_ids"] += 1

                if not is_qaic:
                    for i, iname in enumerate(input_names):
                        if f"{iname}_RetainedState" in outputs.keys() and not (
                            f"{iname}_RetainedState" == "attention_mask_RetainedState"
                        ):
                            inputs[iname] = np.concatenate(
                                (
                                    outputs[f"{iname}_RetainedState"],
                                    np.zeros((1, input_shapes[i][1], 1, input_shapes[i][3]), dtype="float32"),
                                ),
                                axis=2,
                            )

                outputs_logits.append(torch.tensor(logits))

            outputs_logits = torch.cat(outputs_logits[:-1], dim=1)
            targets_label = torch.cat(targets_label, dim=1)

            if pad_input_batch is not None:
                outputs_logits = outputs_logits[:-pad_input_batch, :, :]
                targets_label = targets_label[:-pad_input_batch, :]

            # Calculate loss for the entire sequence
            loss = loss_fct(outputs_logits.view(-1, outputs_logits.size(-1)), targets_label.view(-1))
            loss = loss.mean()  # Average loss over all tokens

            total_loss += loss.item() * targets_label.numel()
            total_tokens += targets_label.numel()

            with open(log_file, "a") as fp:
                fp.write(f"sample_no:{cnt} \t avg_loss: {loss.item():.6f}\n")
            cnt += 1

            loop_time = time.time() - loop_s
            logger.info(
                f"e2e sample time: {(loop_time) / batch_size:.4f}s\t e2e tokens/s : {((ctx_len - prompt_len) * batch_size) / loop_time:.2f}"
            )

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity, avg_loss


# 4. Main Execution Flow
def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity for ONNX or QPC models")
    parser.add_argument("--model_path", required=False, help="Path to ONNX or QPC model")
    parser.add_argument(
        "--model_type", choices=["onnx", "qpc", "torch"], required=True, help="Type of model (onnx or qpc or torch)"
    )
    parser.add_argument("--model_name", required=True, help="Name of the HuggingFace Model Card Name/tokenizer")
    parser.add_argument("--dataset_name", default="wikitext-2-raw-v1", help="Name of the dataset")
    parser.add_argument("--ctx_len", type=int, default=2048, help="Context length")
    parser.add_argument("--prompt_len", type=int, default=1, help="Prompt length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--stride", type=int, default=1024, help="Stride for dataset")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to use (-1 for all)")
    parser.add_argument(
        "--qpc_device_id",
        default=[0],
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        help="QAIC device ids (comma-separated)",
    )
    parser.add_argument("--log_file", default="perplexity_results.log", help="Log file name")
    args = parser.parse_args()

    start_time = time.time()

    # Load data
    data_loader = WikiTextDataLoader(
        args.dataset_name, args.model_name, args.ctx_len, args.batch_size, args.stride, args.num_samples
    )

    # Create inference session
    if args.model_type == "onnx":
        inference_session = ONNXInferenceSession(args.model_path)
    elif args.model_type == "torch":
        inference_session = PytorchInferenceSession(args.model_name)
    elif args.model_type == "qpc":
        inference_session = QPCInferenceSession(args.model_path, args.qpc_device_id)
    else:
        raise NotImplementedError

    # Calculate perplexity
    perplexity, loss = calculate_perplexity(
        data_loader,
        inference_session,
        args.ctx_len,
        args.prompt_len,
        args.model_type,
        args.batch_size,
        args.log_file,
        args.model_name,
    )

    # Log results
    logger.info(f"{args.model_type.upper()} Perplexity: {perplexity:.8f}")
    logger.info(f"{args.model_type.upper()} Loss: {loss:.8f}")
    logger.info(f"Total time for evaluation: {(time.time() - start_time) / 3600.0:.4f} hrs")

    with open(f"{args.log_file}", "a") as fp:
        print(f"\nLoading Dataset: {args.dataset_name}", file=fp)
        print(f"Loading Model From: {args.model_path}", file=fp)
        print(f"Model Type Mentioned: {args.model_type}", file=fp)
        print(f"Samples for Inference: {args.num_samples}", file=fp)
        print(f"Batch Size: {args.batch_size}", file=fp)
        print(f"Context Length: {args.ctx_len}", file=fp)
        print(f"Dataset Prompt Length: {args.prompt_len}", file=fp)
        print(f"Dataset Stride: {args.stride}", file=fp)
        print(f"Overall Loss: {loss}", file=fp)
        print(f"Perplexity: {perplexity}", file=fp)
        print(f"Total time for evaluation: {(time.time() - start_time) / 3600.0} hrs", file=fp)
        if isinstance(args.model_type, str) and args.model_type == "torch":
            print("\n*******************************************************", file=fp)
            print(f"Torch Original Perplexity: {perplexity}", file=fp)
            print(f"Target Perplexity for FP16 Precision: {(perplexity * 100.1) / 100}", file=fp)
            print(f"Target Perplexity for MXFP6/MXINT8 Precision: {(perplexity * 101) / 100}", file=fp)
            print("*******************************************************", file=fp)


if __name__ == "__main__":
    main()
