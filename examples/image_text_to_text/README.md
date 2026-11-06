# Image-Text-to-Text (Vision-Language Models)

Multi-modal models that process both images and text.


## Authentication

For private/gated models, export your HuggingFace token:
```bash
export HF_TOKEN=<your_huggingface_token>
```
## Quick Start
### Generic VLM Inference
Generic script for vision-language models:

```bash
# With default parameters
python basic_vlm_inference.py

# With custom parameters
python basic_vlm_inference.py \
    --model-name llava-hf/llava-1.5-7b-hf \
    --image-url "https://example.com/image.jpg" \
    --query "Describe this image" \
    --prefill-seq-len 128 \
    --ctx-len 3000 \
    --generation-len 128 \
    --num-cores 16
```

### Single QPC Mode
Run the entire model (vision encoder + language model) in a single QPC:

```bash
python basic_vlm_inference.py \
    --model-name llava-hf/llava-1.5-7b-hf \
    --image-url "https://example.com/image.jpg" \
    --query "Describe this image" \
    --num-cores 16 \
    --num-devices 1
```

### Dual QPC Mode
Split the model into two QPCs (vision encoder + language model separately):

```bash
python basic_vlm_inference.py \
    --model-name llava-hf/llava-1.5-7b-hf \
    --image-url "https://example.com/image.jpg" \
    --query "Describe this image" \
    --kv-offload \
    --num-cores 16 \
    --num-devices 1
```

**Note:** In Dual QPC mode (`kv_offload=True`), the vision encoder runs in one QPC and the language model in another, with outputs transferred via host. This provides flexibility for independent execution of vision and language components.

### Disaggregated Vision/Prefill/Decode QPC Mode
Use the unified model example to export and compile three independent QPCs, then run greedy generation by chaining vision outputs into chunked language prefill and language decode:

```bash
export HF_HUB_CACHE=/home/huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1

python models/generic_disagg_inference.py \
    --model-name tiny-random/qwen3-vl-moe \
    --text-num-hidden-layers 2 \
    --vision-num-hidden-layers 2 \
    --prefill-seq-len 128 \
    --ctx-len 4096 \
    --height 354 \
    --width 536 \
    --mxfp6-matmul \
    --mxint8-kv-cache
```

The script chooses model-specific shape arguments automatically for Qwen-style (`height`/`width`), Kimi-style (`image_height`/`image_width`), and square-image (`img_size`) models. For a newly added model, use `--shape-arg-mode` plus `--compiler-options` for any wrapper-specific compile arguments.

To only export and compile the three QPCs without running QAIC inference, add `--compile-only`. To run from precompiled artifacts, pass `--skip-compile --vision-qpc-path <path> --prefill-qpc-path <path> --decode-qpc-path <path>`.

### Text-Only Execution (Skip Vision)
Run text-only inference without image processing:

```bash
python basic_vlm_inference.py \
    --model-name llava-hf/llava-1.5-7b-hf \
    --prompt "Tell me about yourself" \
    --skip-vision True
```

**Note:** Use `skip_vision=True` when you want to run the language model without processing any images. This is useful for text-only tasks on vision-language models.

### Continuous Batching
Dynamic batching for VLMs:

```bash
python continuous_batching_vlm.py \
    --model-name meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --full-batch-size 4 \
```

## Supported Models

**QEff Auto Class:** `QEFFAutoModelForImageTextToText`

For the complete list of supported vision-language models, see the [Validated Models - Vision-Language Models Section](../../docs/source/validate.md#vision-language-models-text--image-generation).

Popular model families include:
- Llama Vision (3.2, 4-Scout)
- Qwen VL (2.5)
- Mistral Vision (Small-3.1)
- Gemma-3
- Granite Vision (3.2)
- InternVL
- Molmo
- LLaVA
- Kimi-K2.5

### Model-Specific Examples

Some models have specialized examples demonstrating advanced features:

| Model | Location |
|-------|----------|
| **Llama-4**  | [models/llama4/](models/llama4/) |
| **Qwen** |  [models/qwen_vl/](models/qwen_vl/) |
| **Qwen 3.5** | [models/qwen3_5/](models/qwen3_5/) |
| **Qwen 3.5 MoE** | [models/qwen3_5_moe/](models/qwen3_5_moe/) |
| **Mistral** | [models/mistral_vision/](models/mistral_vision/) |
| **Gemma** | [models/gemma_vision/](models/gemma_vision/) |
| **Granite** | [models/granite_vision/](models/granite_vision/) |
| **InternVL** | [models/internvl/](models/internvl/) |
| **Molmo** | [models/molmo/](models/molmo/) |
| **Kimi-K2.5** | [../kimi_k2/export_kimi_k25_vision.py](../kimi_k2/export_kimi_k25_vision.py) |

### Kimi-K2.5 Vision

Kimi-K2.5 uses the `moonshotai/Kimi-K2.5` remote-code model with the `KimiK25ForConditionalGeneration` architecture. The example script loads either the full checkpoint or a smaller layer subset, wraps it with `QEFFAutoModelForImageTextToText`, and compiles the vision-language path with Kimi-specific image-size specializations.

```bash

python ../kimi_k2/export_kimi_k25_vision.py \
    --model-path "$HF_HUB_CACHE/models--moonshotai--Kimi-K2.5/snapshots/<snapshot-id>" \
    --image-url "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png" \
    --image-height 512 \
    --image-width 910 \
    --prompt "Describe this image."
```

For custom image sizes, pass only the input image pixel dimensions with `--image-height` and `--image-width`. Kimi-K2.5 compile derives the internal grid height, grid width, patch count, and image-token count from those pixel dimensions.

For reranker examples, see [../reranker/](../reranker/).


## Documentation
- **Full Guide**: [VLM Documentation](../../docs/source/quick_start.md#vision-language-models)
- **API Reference**: [QEFFAutoModelForImageTextToText](../../docs/source/qeff_autoclasses.md#QEFFAutoModelForImageTextToText)
