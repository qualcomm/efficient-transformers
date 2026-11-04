# QEfficient Examples

Examples for running models on Qualcomm Cloud AI 100.

For detailed documentation, see https://quic.github.io/efficient-transformers/


## Layerwise Requirements

For running layerwise pipelines, the following dependency is required:

- Install `onnx-ir` (specific version):
```bash
pip install onnx_ir==0.2.1
```
## Quick Navigation

### Text Generation
Language model inference.

| Example | Description | Script |
|---------|-------------|--------|
| Basic Inference | Simple text generation | [text_generation/basic_inference.py](text_generation/basic_inference.py) |
| FP8 ONNX Export | Export and verify retained FP8 weights | [text_generation/fp8_onnx_export.py](text_generation/fp8_onnx_export.py) |
| GGUF Models | GGUF format support | [text_generation/gguf_models.py](text_generation/gguf_models.py) |
| MoE Models | Mixture of Experts | [text_generation/moe_inference.py](text_generation/moe_inference.py) |
| Continuous Batching | Dynamic batching | [text_generation/continuous_batching.py](text_generation/continuous_batching.py) |

[See all text generation examples →](text_generation/)

### Dynamo
`torch.export` based export examples.

| Example | Description | Script |
|---------|-------------|--------|
| CausalLM Dynamo | Dynamo export with optional weight-free mode | [dynamo/causal_lm/basic_dynamo_inference.py](dynamo/causal_lm/basic_dynamo_inference.py) |

[See all dynamo examples →](dynamo/)

### Image-Text-to-Text
Vision-language models.

| Example | Model | Script |
|---------|---------------|---------------|
| Basic VLM | Most VLMs |  [image_text_to_text/basic_vlm_inference.py](image_text_to_text/basic_vlm_inference.py) |

[See all vision-language examples →](image_text_to_text/)

### Embeddings
Sentence and document embeddings.

| Example | Model | Script |
|---------|-------|--------|
| Text Embeddings | all-MiniLM-L6-v2 | [embeddings/text_embeddings.py](embeddings/text_embeddings.py) |
| Qwen3-VL Embedding | Qwen/Qwen3-VL-Embedding-8B | [embeddings/qwen3vl/qwen3_vl_embedding.py](embeddings/qwen3vl/qwen3_vl_embedding.py) |

[See all embedding examples →](embeddings/)

### Reranker
Multimodal reranker scoring examples.

| Example | Model | Script |
|---------|-------|--------|
| Qwen3-VL Reranker | Qwen/Qwen3-VL-Reranker-{2B,8B} | [reranker/qwen3vl/qwen3_vl_reranker.py](reranker/qwen3vl/qwen3_vl_reranker.py) |

[See all reranker examples →](reranker/)

### Audio
Speech processing models.

| Example | Model | Task | Script |
|---------|-------|------|--------|
| Speech-to-Text | Whisper | Transcription | [audio/speech_to_text.py](audio/speech_to_text.py) |
| CTC Speech Recognition | Wav2Vec2 | Recognition | [audio/wav2vec2_inference.py](audio/wav2vec2_inference.py) |

[See all audio examples →](audio/)

### PEFT
Parameter-efficient fine-tuning.

| Example | Description | Script |
|---------|-------------|--------|
| Single Adapter | Load and use one adapter | [peft/single_adapter.py](peft/single_adapter.py) |
| Multi-Adapter | Multiple adapters with CB | [peft/multi_adapter.py](peft/multi_adapter.py) |

**Note:** PEFT examples use hardcoded configurations to demonstrate specific adapter workflows. Modify the scripts directly to test different adapters or configurations.

[See all PEFT examples →](peft/)

### Performance
Optimization techniques.

| Example | Technique | Script |
|---------|-----------|--------|
| Draft-based SpD | Speculative decoding | [performance/speculative_decoding/draft_based.py](performance/speculative_decoding/draft_based.py) |
| Prompt Lookup | N-gram speculation | [performance/speculative_decoding/prompt_lookup.py](performance/speculative_decoding/prompt_lookup.py) |
| Multi-Projection | Turbo models | [performance/speculative_decoding/multi_projection.py](performance/speculative_decoding/multi_projection.py) |
| On-Device Sampling | Sampling parameters | [performance/on_device_sampling.py](performance/on_device_sampling.py) |
| Compute Context Length | Dynamic context optimization | [performance/compute_context_length/basic_inference.py](performance/compute_context_length/basic_inference.py) |
| C++ Execution | Native C++ API | [performance/cpp_execution/](performance/cpp_execution/) |

[See all performance examples →](performance/)

### Disaggregated Serving
Distributed inference across multiple devices.

| Example | Description | Script |
|---------|-------------|--------|
| Basic Disaggregated Serving | Multi-device serving | [disagg_serving/gpt_oss_disagg_mode.py](disagg_serving/gpt_oss_disagg_mode.py) |
| Chunking Disaggregated Serving | Multi-device serving | [disagg_serving/gpt_oss_disagg_mode_with_chunking.py](disagg_serving/gpt_oss_disagg_mode_with_chunking.py) |
| Qwen3-VL MDP Compile | MDP compile-only validation for Qwen3-VL | [disagg_serving/qwen3_vl_mdp_compile.py](disagg_serving/qwen3_vl_mdp_compile.py) |

## Installation

For installation instructions, see the [Quick Installation guide](../README.md#quick-installation) in the main README.


## Running Examples

### Python Scripts

Basic usage:
```bash
python text_generation/basic_inference.py \
    --model-name gpt2 \
    --prompt "Hello, how are you?"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new examples.

## Documentation

Full documentation: https://quic.github.io/efficient-transformers/
