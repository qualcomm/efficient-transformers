# Perplexity Calculator

This script calculates the perplexity for ONNX, QPC, or Torch models using the WikiText-2 dataset. It supports different model types and configurations.

## Table of Contents

- Requirements
- Installation
- Usage
  - Example
- Arguments
- Output Details

## Requirements

- Python 3.8+
- Required Python packages:
  - `QEfficient`
  - `datasets==2.20`

## Installation

- Install QEfficient and update the datasets package to 2.20

## Usage

To run the script, use the following command:

```bash
python calculate_perplexity.py --model_type <model_type> --model_name <model_name> [--model_path <model_path>] [--dataset_name <dataset_name>] [--ctx_len <ctx_len>] [--prompt_len <prompt_len>] [--batch_size <batch_size>] [--stride <stride>] [--num_samples <num_samples>] [--qpc_device_id <qpc_device_id>] [--log_file <log_file>]

python perplexity_calculator_cloud.py --model_type torch --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_samples 1
```

## Arguments (Help Section)
```bash
--model_path: Path to ONNX or QPC model (optional for Torch Original models).
--model_type: Type of model (onnx, qpc, or torch) (required).
--model_name: Name of the HuggingFace Model Card Name/tokenizer (required).
--dataset_name: Name of the dataset (default: wikitext-2-raw-v1).
--ctx_len: Context length (default: 2048).
--prompt_len: Prompt length (default: 1).
--batch_size: Batch size (default: 1).
--stride: Stride for dataset (default: 1024).
--num_samples: Number of samples to use (-1 for all) (default: -1).
--qpc_device_id: QAIC device ids (comma-separated) (default: [0]).
--log_file: Log file name (default: perplexity_results.log).
--trust_remote_code: Load the torch reference with trust_remote_code=True (remote-code models such as IFM/K2-Horizon-*).
```

## Notes

- Keep `--prompt_len 1` (the default). After the first token the script feeds one token per step at the next position, so a larger prompt length does not match the data windows. Every QPC built by QEfficient has the seq_len 1 specialization this needs.
- `--ctx_len` must not exceed the ctx_len the QPC was compiled with.
- Use the same `--ctx_len`, `--stride` and `--num_samples` for the torch, fp16 and MXFP6 legs; the pass targets at the end of the torch log are ratios against that run.

## Output Details
The script logs the following information:

- Perplexity and loss for the specified model. (For Original Torch, it will dump the Target for FP16 and MXFP6 Precision too)
- Total time taken for evaluation.
- Detailed configuration and results in the specified log file.






