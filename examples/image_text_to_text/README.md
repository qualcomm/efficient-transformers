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

### Weight-Free Offline Prefill Blocking
Run the full cached Qwen3-VL 2B, Qwen3.5-35B-A3B, and Gemma 4 E2B checkpoints
serially through the `prefill_q`, `prefill_kv`, and `prefill_qkv` modes:

```bash
HF_HUB_CACHE=/home/huggingface_hub \
HF_HUB_ENABLE_HF_TRANSFER=1 \
python blocking_vlm_inference.py --model all --mode all \
  --output-root /home/sabhis/blocking/qeff_artifacts/full_model_prefill_serial_RUN_ID
```

The runner writes a `result.json` and `run.log` per model/mode, captures generated
token IDs in `generated_tokens.json`, and records export and compiler timings by
vision/prefill/decode subfunction. Failed cases do not prevent later serial cases
from running. Each case has an isolated `QEFF_HOME` and `TMPDIR`; choose a new,
unused output root for every retry. The script requires the full checkpoints to
already be present in the Hugging Face cache and does not fetch missing files.

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
