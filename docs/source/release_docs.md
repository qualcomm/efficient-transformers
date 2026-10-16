# Efficient Transformer Library - 1.22.0 Release Notes

Welcome to the official release of **Efficient Transformer Library v1.22.0**! This release delivers major new model support including Gemma 4, Qwen3-VL, and GLM4-MOE, a full rebase to Transformers v5.5.4, Python 3.12 support, advanced attention blocking, Multi-head Latent Attention (MLA), and numerous performance and export improvements.

> ✅ All features and models listed below are available on the [`release/v1.22.0`](https://github.com/quic/efficient-transformers/tree/release/v1.22.0) branch and [`mainline`](https://github.com/quic/efficient-transformers/tree/main).

---

## Newly Supported Models

- **Gemma 4**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText) (multimodal)
  - Covers both dense and MoE configurations
  - Optimized chunked prefill and disaggregated (vision/language split) serving
  - [Gemma 4 Example Scripts](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/gemma_vision/gemma4)

- **Qwen3-VL (Vision Language)**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Dense and MoE variants supported
  - Continuous batching enabled
  - Disaggregated VL-MoE serving (three-QPC split)
  - [Qwen3-VL Example Scripts](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3vl)

- **Qwen3.5 / Qwen3.6 VLM Series**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Linear attention variants for Qwen3.5 and Qwen3.6 architectures
  - Continuous batching and disaggregated serving support
  - [Qwen3.5 Example Scripts](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3_5)

- **GLM4-MOE**
  - Executable via [`QEFFAutoModelForCausalLM`](#QEFFAutoModelForCausalLM)
  - Disaggregated prefill and decode support with chunked prefill
  - KV-blocked attention and ONNX subfunction export
  - [GLM4-MOE Disaggregated Example](https://github.com/quic/efficient-transformers/blob/main/examples/disagg_serving/glm4_moe_disagg_mode_with_chunking.py)

---

## Deprecations

- **Llama 3.2 Vision (mllama)**: Support for `meta-llama/Llama-3.2-11B-Vision` has been deprecated. Users should migrate to supported VLM alternatives.

---

## Key Features & Enhancements

- **Framework Upgrades**
  - Transformers rebased to `v5.5.4` — modeling wrappers, `cache_utils`, and export flows aligned; PyTorch/ORT parity restored
  - Intermediate rebase to `v4.57.3` also landed, unifying subfunction changes
  - Python `3.12` support added (while retaining `3.10` and `3.11` compatibility)

- **Causal Attention Blocking**
  - New KV, Q, Head, and Batch blocking strategies for CausalLM models
  - Supported architectures: Llama3, GPT-OSS, Gemma, Gemma2, Granite, GraniteMOE, Mistral
  - Configurable via `qaic_config` with `blocking_mode`; omit `blocking_mode` to disable it
  - Automatic blocking configurator with `skip_kv` enabled by default
  - Optimized nested loop implementation for attention blocking

- **NSP-Blocked Expert Dispatch for MoE Prefill**
  - Batched packed-prefix expert dispatch for Qwen3MOE and GPT-OSS chunked prefill
  - Configurable via `num_cores` and `qaic_config["moe_config"]["expert_parallel_chunk_size"]`
  - Replaces sequential per-expert loop for improved throughput

- **First-Block Caching for Diffusers (Wan)**
  - Block caching infrastructure for Wan non-unified transformer
  - Enable via `enable_first_block_cache=True` and `first_block_cache_downsample_factor`
  - Supports `Wan-AI/Wan2.2-T2V-A14B-Diffusers` at 720p

- **Named Graph Specializations**
  - `specializations.json` now carries meaningful graph names: `Prefill`, `Decode`, `Vision`, `Encoder`, `Embedding`
  - Automatic name inference based on input key patterns

- **Multiple TLM Decode Specializations**
  - `QEFFAutoModelForCausalLM.compile()` now accepts `num_speculative_tokens` as `List[int]`
  - Compiles multiple decode specializations in one QPC for per-step dispatch to the cheapest kernel
  - Backward compatible: plain `int` input still works

- **CCL Support for Disaggregated Prefill**
  - Compute-Context-Length (CCL) now supported during the prefill phase of disaggregated serving
  - Enables compilation at maximum context length with efficient TTFT for smaller prompt lengths

- **FP16/BF16 Export & Compile for VLMs**
  - Added `fp16`/`bf16` precision export and compile support for Vision Language Models

- **Proxy Model Export**
  - New `enable_proxy=True` flag to export a proxy model (disables Embedding Layer and LM Head)
  - Supported across text, embedding, vision, and audio models
  - Proxy-gated ONNX transforms: `FP16ClipTransform` and `SplitTensorsTransform` now only applied when `enable_proxy=True`
  - Usage: pass `enable_proxy=True` to `from_pretrained()` and compile normally; for CausalLM CLI flows, use `examples/text_generation/basic_inference.py --enable-proxy --compile-only`
  - Optional `num_hidden_layers` can be passed with `enable_proxy=True` to force a specific reduced proxy depth

- **On-Device Sampling for Qwen3**
  - Extended on-device sampling support to `Qwen3ForCausalLM`

- **ONNX Subfunction Support for VLMs**
  - `use_onnx_subfunctions=True` now supported for Vision Language Models
  - RoPE indexing hoisted out of decoder subfunctions for cached text models (performance improvement)
  - Subfunction changes for Qwen2.5 models

- **Weight Offloading Improvements**
  - Improved weight offloading to handle plain tensor attributes using `to_empty()`
  - Fixes for large model export OOM risk

- **Large Model ONNX Export**
  - `SplitTensorsTransform` added to `QEFFAutoModel` to prevent >2GB protobuf export failures
  - Improved proxy-only clip/split transform gating

- **Compiler Toolchain Update**
  - Compilation updated from deprecated `qaic-exec` to `qaic-compile`
  - Removed deprecated `-compile-only` compiler flag

- **Disaggregated Serving Enhancements**
  - ONNX reuse fix for disaggregated compile and diffusion modules
  - Qwen3-MoE disaggregated mode support
  - CCL input precision changed from `int8` to `int64` to align with compiler

- **Bug Fixes**
  - Fixed MLA KV head replication for correct head count handling
  - Fixed RoPE dtype for `custom_dtype` export
  - Fixed `fp16`/`bf16` export and compile for Qwen3-VL and Qwen3-VL-MoE models
  - Fixed continuous batching inconsistency for Qwen2.5-VL
  - Fixed Gemma3 sliding window issue
  - Fixed CCL support during disaggregated serving decode phase
  - Fixed `device_ids` type handling for single-device `qaicrt.Program`
  - Clarified CausalLM decode-only flag handling and fixed PL=1 specialization naming
  - Removed `custom_io` passing for BF16 dtype case

---

## Example Scripts

- [Gemma 4](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/gemma_vision/gemma4)
- [Qwen3-VL](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3vl)
- [Qwen3.5 VLM](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3_5)
- [Qwen3.5-MoE](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3_5_moe)
- [Qwen3-VL-MoE](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3_vl_moe)
- [GLM4-MOE Disaggregated](https://github.com/quic/efficient-transformers/blob/main/examples/disagg_serving/glm4_moe_disagg_mode_with_chunking.py)
- [Sequence Classification](https://github.com/quic/efficient-transformers/tree/main/examples/sequence_classification)
- [Wan Image-to-Video](https://github.com/quic/efficient-transformers/tree/main/examples/diffusers/wan)


# Efficient Transformer Library - 1.21.6 Release Notes

Welcome to the official release of **Efficient Transformer Library v1.21.6**! This targeted release builds on the v1.21 line with multi-resolution Vision Language Model workflows, Qwen3-VL stability fixes, on-device sampling enablement, online serving support for Gemma3 through vLLM, and compatibility updates for newer model and framework APIs.

> ✅ The exact release content is available on the [`release/v1.21.6`](https://github.com/quic/efficient-transformers/tree/release/v1.21.6) branch. The package version for this branch is `1.21.6.0`.

---

## Branch Summary

- **Release branch**: [`release/v1.21.6`](https://github.com/quic/efficient-transformers/tree/release/v1.21.6)
- **Release head**: `25e7c53` (`Updated release version to 1.21.6.0`)
- **Mainline comparison**: Reviewed against `upstream/main`; the release branch contains 11 release commits from merge base `d02f717`.

---

## Key Features & Enhancements

- **Multi-specialization vision compilation for Qwen VLMs**
  - Qwen2.5-VL, Qwen3-VL Dense can compile multiple vision resolution and frame configurations in one pass.
  - `height`, `width`, and `num_frames` can be supplied as lists when building specializations.
  - Runtime generation can select the matching specialization through the multi-frame generation path.
  - New example scripts are available for [Qwen2.5-VL](https://github.com/quic/efficient-transformers/tree/release/v1.21.6/examples/image_text_to_text/models/qwen2_5_vl), [Qwen3-VL Dense](https://github.com/quic/efficient-transformers/tree/release/v1.21.6/examples/image_text_to_text/models/qwen3vl).

- **Qwen3-VL Dense on-device sampling**
  - Registers Qwen3-VL Dense with the sampler transform path.
  - Handles Qwen3-VL Dense deepstack feature inputs and outputs for on-device sampling.
  - Adds sampler coverage to validate the new transform behavior.

- **Large embedding export robustness**
  - Adds `SplitTensorsTransform` to `QEFFAutoModel` ONNX transforms so large initializers are emitted as `*.onnx.data` sidecar files.
  - Prevents ONNX ModelProto parser failures when exports exceed the 2 GB protobuf limit.
  - Adds regression coverage for large embedding and reranker model export flows.

- **Qwen VLM runtime stability**
  - Fixes Qwen3-VL Dense continuous batching with multi-image, multi-prompt inputs by preserving the complete hidden-state tensor during broadcast.
  - Handles multi-resolution `vision_embeds` edge cases for Qwen2.5-VL, Qwen3-VL Dense, and Qwen3-VL-MoE.
  - Moves Qwen2.5-VL examples into a dedicated `qwen2_5_vl` example directory.

- **Gemma3 configuration compatibility**
  - Updates Gemma3 cache handling for the newer `_sliding_window_pattern` config field.
  - Preserves sliding-window behavior for Gemma3 models using updated Transformers configs.
  - Added online serving support for Gemma3 through vLLM

- **Llama4 compatibility with Transformers `4.57.3`**
  - Adds `**kwargs` support to `QEffLlama4VisionModel.forward()`.
  - Accepts `vision_feature_layer` and `vision_feature_select_strategy` forwarded by newer Transformers Llama4 APIs.
  - Fixes ONNX export failures for Llama4 vision models while remaining backward compatible.

- **GPT-OSS batch size flexibility**
  - Added GPT OSS 120B with BS>1 and GPT OSS 20B BS>2 support is enabled

---

## Validation & Quality Updates

- Added tests for Qwen3-VL Dense on-device sampling transformations.
- Added regression tests that verify large ONNX initializers are split into external data files.
- Updated image-text model configs and Qwen3-VL examples for continuous batching and multi-specialization workflows.
- Reverted a temporary Qwen VLM multi-image test/config change before landing the stable Qwen3-VL Dense continuous batching fix.

---

# Efficient Transformer Library - 1.21.0 Release Notes

Welcome to the official release of **Efficient Transformer Library v1.21.0**! This release introduces advanced attention mechanisms, expanded model support, optimized serving capabilities, and significant improvements to fine-tuning and deployment workflows.

> ✅ All features and models listed below are available on the [`release/v1.21.0`](https://github.com/quic/efficient-transformers/tree/release/v1.21.0) branch and [`mainline`](https://github.com/quic/efficient-transformers/tree/main).

---

## Newly Supported Models

- **Flux (Diffusers - Image Generation)**
  - Diffusion-based image generation model
  - [Flux.1 Schnell Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/diffusers/flux/flux_1_schnell.py)

- **WAN (Diffusers - Video Generation)**
  - Wide-Area Network Lightning support for distributed inference
  - [Wan_lightning Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/diffusers/wan/wan_lightning.py)

- **Qwen2.5-VL (Vision Language)**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Multi-image prompt support
  - Continuous batching enabled
  - [Qwen2.5-VL Usage Guide](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen_vl)

- **Mistral 3.1 (24B)**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - [Mistral-3.1 Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text/models/mistral_vision/mistral3_example.py)

- **Kimi-K2.5 (Vision Language)**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Supports `moonshotai/Kimi-K2.5` with `KimiK25ForConditionalGeneration`
  - Supports dual-QPC (`kv_offload=True`) vision-language compilation
  - [Kimi-K2.5 Vision Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/kimi_k2/export_kimi_k25_vision.py)


- **Disaggregated serving ready via vLLM GPT-OSS**
  > **Note**: If running GPT-OSS models natively via vLLM, PR-685 of the qefficient library is required for Python 3.12 compatibility.
    
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Separate prefill and decode compilation supported
  - Disaggregated serving ready
  - [GPT-OSS Example Scripts](https://github.com/quic/efficient-transformers/blob/main/examples/disagg_serving/gpt_oss_disagg_mode.py)

- **Olmo2**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Full CausalLM support with optimizations
  - Refer to [Text generation Example Scripts](https://github.com/quic/efficient-transformers/tree/main/examples/text_generation) for usage details.

- **Molmo**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Multi-modal capabilities
  - [Molmo Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text/models/molmo/molmo_example.py)

- **InternVL 3.5 Series**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Full Vision-Language support
  - Multi-image handling with continuous batching
  - Refer to [InternVL 3.5 Example Scripts](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/internvl) for usage details.

- **Qwen3-MOE (Mixture of Experts)**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Efficient expert routing
  - [Qwen3-MOE Example Scripts](https://github.com/quic/efficient-transformers/blob/main/examples/text_generation/moe_inference.py)

- **Wav2Vec2 (Audio)**
  - Executable via [`QEFFAutoModelForCTC`](#QEFFAutoModelForCTC)
  - Speech recognition and audio feature extraction
  - [Wav2Vec2 Example Scripts](https://github.com/quic/efficient-transformers/blob/main/examples/audio/wav2vec2_inference.py)

- **Multilingual-e5-Large (Embedding Model)**
  - Executable via [`QEffAutoModel`](#QEffAutoModel)
  - Multilingual text embedding capabilities
  - Refer [usage details](https://github.com/quic/efficient-transformers/tree/main/examples/embeddings) here.

---

## Key Features & Enhancements

- **Framework Upgrades**: Transformers `4.55`, PyTorch `2.7.0+cpu`, Torchvision `0.22.0+cpu`
- **Python Support**:  Requires Python `3.10`
- **ONNX Opset**: Updated to version `17` for broader operator support
- **Advanced Attention**: Flux blocking support, BlockedKV attention for CausalLM models
- **Diffusers Integration**: Full support for diffuser-based image generation and video generation models
- **Compute-Context-Length (CCL) support**: To optimize the throughput when handling very large context lengths
- **Prefill/Decode Separation**: Support for GPT OSS using disaggregate serving models
- **Continuous Batching (VLMs)**: Extended to Vision Language Models with multi-image handling
  - Supported models: Llava, Llava_Next, Gemma3, Mistral3, InternVL2_5, InternVL3_5, Molmo, Kimi-K2.5
- **ONNX Sub-Functions**: Feature enabling more efficient model compilation and execution on hardware. Users can enable the feature by passing `use_onnx_subfunctions=True` during export
- **Memory Profiling**: Built-in utilities for optimization analysis
- **Extend on-device Sampling**: Extend on-device sampling to dual QPC VLMs and Guided decoding for on-device sampling
- **ONNX transform, memory & time optimizations**: Optimizations for faster ONNX Transform and reduced memory footprint
- **Removed platform SDK dependency**: Support QPC generation on systems without the Platform SDK
- **Example Scripts Revamp**: New example scripts for audio, embeddings, and image-text-to-text tasks
- **Onboarding Guide**:
Simplified setup and deployment process for new users
  - [CausalLM Onboarding Guide](https://github.com/quic/efficient-transformers/tree/release/v1.21.0/examples/onboarding_guide/causallm)
  - [Custom ops](https://github.com/quic/efficient-transformers/tree/release/v1.21.0/examples/onboarding_guide/customop)
- Organized examples into domain-specific subdirectories [Examples](https://github.com/quic/efficient-transformers/tree/release/v1.21.0/examples)




---

## Embedding Model Upgrades

- **Multi-Sequence Length Support**: Auto-selects optimal graph at runtime
- **Enhanced Pooling**: Flexible pooling strategies for various embedding tasks

---

## Fine-Tuning Support

- **Checkpoint Management**: Resume from epochs with proper state restoration
- **Enhanced Loss Tracking**: Corrected data type handling for accurate loss computation
- **Custom Dataset Support**: Improved handling with better tokenization
- **Device-Aware Scaling**: Optimized GradScaler for multi-device training
- **Comprehensive Testing**: Unit tests for fine-tuning workflows

---


# Efficient Transformer Library - 1.20.0 Release Notes

Welcome to the official release of **Efficient Transformer Library v1.20.0**! This release introduces advanced attention mechanisms, expanded model support, optimized serving capabilities, and significant improvements to fine-tuning and deployment workflows.

> ✅ All features and models listed below are available on the [`release/v1.20.0`](https://github.com/quic/efficient-transformers/tree/release/v1.20.0) branch and [`mainline`](https://github.com/quic/efficient-transformers/tree/main).

---


## Newly Supported Models

- **Llama-4-Scout-17B-16E-Instruct**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Text & Image+Text support
  - Chunk attention, Single/Dual QPC support
  - Multi-image prompts enabled via VLLM interface
  - [Llama4 Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text/models/llama_vision/single_image.py)

- **Grok-1**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)

- **Gemma3**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Text & Image+Text support
  - Sliding window support
  - [Gemma3 Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text/models/gemma_vision/inference.py)


- **SwiftKV (Llama-3.1-SwiftKV-8B-Instruct)**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Supports both continuous and non-continuous batching

- **GGUF Models**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Execution support (non-quantized)
  - [Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/text_generation/gguf_models.py)

- **FP8 Compressed Quantization**
  - Support for [`Llama-3.3-70B-Instruct-FP8-Dynamic`](https://huggingface.co/Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic)

---

## Key Features & Enhancements

- **Transformer Upgrade**: Now using version `4.51.3`
- **SpD & Multi-Projection Heads**: Token speculation via post-attention projections
- **I/O Encryption**: `--io-encrypt` flag support in compile/infer APIs
- **Separate Prefill/Decode Compilation**: For disaggregated serving
- **On-Device Sampling**: Supported using VLLM, which reduces host-device latency for CausalLM models

---

## Embedding Model Upgrades

- **Flexible Pooling**: Choose from standard or custom strategies
- **Sentence Embedding**: Now runs directly on AI100
- **Multi-Seq Length Compilation**: Auto-selects optimal graph at runtime

---

## Fine-Tuning Support

- BERT fine-tuning support with templates and documentation
- Gradient checkpointing, device-aware `GradScaler`, and CLI `--help` added
