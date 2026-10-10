# Text Generation Examples

Examples for running inference on text-only language models on Qualcomm Cloud AI 100.


## Authentication

For private/gated models, export your HuggingFace token:
```bash
export HF_TOKEN=<your_huggingface_token>
```

## Supported Models

**QEff Auto Class:** `QEFFAutoModelForCausalLM`

For the complete list of supported text generation models, see the [Validated Models - Text Generation Section](../../docs/source/validate.md#text-only-language-models).

Popular model families include:
- Llama (2, 3, 3.1, 3.2, 3.3)
- Mistral, Mixtral, Codestral
- Qwen, Qwen2, Qwen3-MoE
- Gemma, CodeGemma
- GPT-2, GPT-J
- Falcon, MPT, Phi-3
- Granite, StarCoder
- OLMo 2
- K2 Horizon (dense sizes, remote code)

---

## Python Examples

### basic_inference.py
Simple text generation with any supported language model.

**Usage:**
```bash
python basic_inference.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --prompt "Hello, how are you?" \
    --prefill-seq-len 32 \
    --ctx-len 128 \
    --num-cores 16
```

This example:
- Demonstrates basic text generation workflow
- Loads any HuggingFace text model
- Compiles and runs inference on Cloud AI 100

### continuous_batching.py
Dynamic batching for processing multiple prompts efficiently.

**Usage:**
```bash
python continuous_batching.py \
    --model-name meta-llama/Llama-3.1-8B \
    --prompts "Hello|Hi there|Good morning|How are you" \
    --full-batch-size 4 \
    --prefill-seq-len 128 \
    --ctx-len 512 \
    --num-cores 16
```

This example:
- Demonstrates continuous batching mode
- Processes multiple prompts in parallel
- Improves throughput for multi-request scenarios
- Uses pipe-separated prompts

### gguf_models.py
GGUF format model support (quantized models). To run GGUF format models, you need to install the `gguf` package:

```bash
pip install gguf
```

**Usage:**
```bash
# With default parameters
python gguf_models.py

# With custom parameters
python gguf_models.py \
    --model-name MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF \
    --gguf-file Mistral-7B-Instruct-v0.3.fp16.gguf \
    --prompt "How are you?" \
    --prefill-seq-len 32 \
    --ctx-len 128 \
    --num-cores 16
```

This example:
- Loads models in GGUF format (quantized models)
- Demonstrates GGUF file loading from HuggingFace
- Compiles and runs inference on Cloud AI 100
- Supports custom GGUF files and prompts

---


### moe_inference.py
Mixture of Experts (MoE) model inference.

**Usage:**
```bash
python moe_inference.py \
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --prompt "Explain quantum computing" \
    --ctx-len 256 \
    --num-cores 16
```

This example:
- Demonstrates MoE model inference
- Uses sparse expert activation for efficiency
- Works with Qwen, Mixtral, and other MoE models
- Supports explicit ONNX subfunction enablement with `--use-onnx-subfunctions`

---

### Remote-code models: K2 Horizon
`IFM/K2-Horizon-*` ships its model class as remote code, so pass `--trust-remote-code`. The dense sizes 0.9B, 3.7B and 7B are validated (the 32B shares the architecture but has not been run); the MoVA/MoE sizes (36B-A4B, 375B-A23B) are refused at load.

**Usage:**
```bash
python basic_inference.py \
    --model-name IFM/K2-Horizon-7B \
    --trust-remote-code \
    --prompt "The capital of France is" \
    --prefill-seq-len 128 \
    --ctx-len 4096 \
    --num-cores 16 \
    --device-group '[0]'
```

The configs declare 524288 positions; the exported rotary tables are capped at `K2_HORIZON_MAX_POSITION_EMBEDDINGS` (65536), so keep `ctx_len` within that. Quote the device group in csh, tcsh and zsh.


## CLI Workflow

The QEfficient CLI provides a streamlined workflow for running text generation models on Cloud AI 100. You can use individual commands for each step or the all-in-one `infer` command.

### Quick Start: All-in-One Inference (Recommended)

The `infer` command handles export, compile, and execute in a single step:

```bash
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --batch_size 1 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0] \
    --prompt "Write a short story about AI" \
    --mxfp6 \
    --mxint8_kv_cache \
    --mos 1 \
    --aic_enable_depth_first
```

**What it does:**
1. Downloads and exports the model to ONNX 
2. Compiles to QPC 
3. Executes inference with your prompt

**CLI API Reference:** [`QEfficient.cloud.infer`](https://quic.github.io/efficient-transformers/source/cli_api.html#qefficient-cloud-infer)

### Step-by-Step Workflow

For more control, you can execute each step individually:

#### Step 1: Export Model to ONNX

Export the HuggingFace model to ONNX format optimized for Cloud AI 100:

```bash
python -m QEfficient.cloud.export \
    --model_name meta-llama/Llama-3.1-8B \
    --cache_dir ~/.cache/qeff_cache
```

This downloads the model and converts it to ONNX format. The ONNX model is saved in the QEfficient cache directory.

**CLI API Reference:** [`QEfficient.cloud.export`](https://quic.github.io/efficient-transformers/source/cli_api.html#qefficient-cloud-export)

#### Step 2: Compile Model to QPC

Compile the ONNX model to Qualcomm Program Container (QPC) format:

```bash
python -m QEfficient.cloud.compile \
    --onnx_path ~/.cache/qeff_cache/meta-llama/Llama-3.1-8B/onnx/model.onnx \
    --qpc_path ./qpc_output \
    --batch_size 1 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0] \
    --mxfp6 \
    --mos 1 \
    --aic_enable_depth_first
```

**Note:** The `compile` API is deprecated for direct use. Use the unified `infer` API instead for most use cases.

**CLI API Reference:** [`QEfficient.cloud.compile`](https://quic.github.io/efficient-transformers/source/cli_api.html#qefficient-cloud-compile)

#### Step 3: Execute Inference

Run inference using the pre-compiled QPC:

```bash
python -m QEfficient.cloud.execute \
    --model_name meta-llama/Llama-3.1-8B \
    --qpc_path ./qpc_output/qpcs \
    --prompt "Write a short story about AI" \
    --device_group [0]
```

This uses the pre-compiled QPC for fast inference. You can run this multiple times with different prompts without recompiling.

**CLI API Reference:** [`QEfficient.cloud.execute`](https://quic.github.io/efficient-transformers/source/cli_api.html#qefficient-cloud-execute)

### Common CLI Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--model_name` | HuggingFace model ID | Required | `meta-llama/Llama-3.1-8B` |
| `--prompt` | Input text prompt | Required | `"Hello, how are you?"` |
| `--prompt_len` | Maximum input sequence length | 32 | `128` |
| `--ctx_len` | Maximum context length (input + output) | 128 | `512` |
| `--batch_size` | Batch size for inference | 1 | `1` |
| `--num_cores` | AI 100 cores to use | 16 | `16` or `14` |
| `--device_group` | Device IDs to use | `[0]` | `[0]` or `[0,1,2,3]` |
| `--mxfp6` | Enable MXFP6 quantization | False | Add flag to enable |
| `--mxint8_kv_cache` | Enable MXINT8 KV cache | False | Add flag to enable |
| `--use-onnx-subfunctions` | Enable ONNX subfunctions for export/compile | False | Add flag to enable |
| `--mos` | Memory optimization strategy | 1 | `1` or `2` |
| `--aic_enable_depth_first` | Enable depth-first execution | False | Add flag to enable |


### Advanced Features

#### Multi-Device Inference (Multi-Qranium)

Run models across multiple devices for better performance:

```bash
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --batch_size 1 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0,1,2,3] \
    --prompt "Explain quantum computing" \
    --mxfp6 \
    --mxint8_kv_cache \
    --aic_enable_depth_first
```

**Documentation:** [Multi-Qranium Inference](https://quic.github.io/efficient-transformers/source/features_enablement.html#multi-qranium-inference)

#### Continuous Batching

Process multiple prompts efficiently with continuous batching:

```bash
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --full_batch_size 4 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0] \
    --prompt "Hello|Hi there|Good morning|How are you" \
    --mxfp6 \
    --mxint8_kv_cache
```

**Note:** Use pipe (`|`) to separate multiple prompts. When using continuous batching, do not specify `--batch_size`.

**Documentation:** [Continuous Batching](https://quic.github.io/efficient-transformers/source/features_enablement.html#continuous-batching)

#### Batch Processing from File

Process multiple prompts from a text file:

```bash
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --full_batch_size 8 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0] \
    --prompts_txt_file_path examples/sample_prompts/prompts.txt \
    --mxfp6 \
    --mxint8_kv_cache
```

### CLI Examples Script

For a comprehensive collection of copy-paste ready CLI commands, run:

```bash
bash cli_examples.sh
```

This script demonstrates:
- Complete 4-step workflow (Export → Compile → Execute → Infer)
- Multi-device inference
- Continuous batching
- Batch processing from file
- Parameter explanations and best practices

---


## Additional Resources

### Documentation
- [CLI API Reference](https://quic.github.io/efficient-transformers/source/cli_api.html) - Complete CLI command documentation
- [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html) - Getting started with QEfficient
- [Features Enablement](https://quic.github.io/efficient-transformers/source/features_enablement.html) - Advanced features guide
- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html) - Python API reference
- [Validated Models](https://quic.github.io/efficient-transformers/source/validate.html) - Supported models list


### Model Storage
By default, exported models and QPC files are stored in `~/.cache/qeff_cache`. Customize this with:
- `QEFF_HOME`: Primary cache directory
- `XDG_CACHE_HOME`: Alternative cache location
