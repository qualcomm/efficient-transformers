<p align="center">
  <img src="./docs/image/Qualcomm-Dragonfly-Logo-1068x601.png" alt="Qualcomm Dragonfly logo" width="420" />
</p>

<p align="center">
  <img src="./docs/image/QualcommAIRack.png" alt="Qualcomm AI200 rack" width="640" />
</p>


---
# Efficient Transformers Library 
---

*Latest news* :fire: <br>

- [07/2026] Added support for Kimi-K2.5 vision-language model [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) via `QEFFAutoModelForImageTextToText`.
- [07/2026] Added `dynamo` flag to `QEFFAutoModelForCausalLM.export()` to support `torch.onnx.export` dynamo-based ONNX export for CausalLM models
- [06/2026] Added support for Gemma4 models, [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B), [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [06/2026] Added support for Qwen3.6 model [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [06/2026] Added support for Qwen3.5 model[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [06/2026] Added support for Qwen3 VL and Qwen3 MoE VL models, [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct), [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [06/2026] Added support for GLM-4.5 MoE model and enabled its disaggregated mode [zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5)
- [06/2026] Enabled disaggregated support for Qwen3 MoE model
- [04/2026] Added WAN non-unified execution support in `QEffWanPipeline` with separate `transformer_high` and `transformer_low` modules
- [04/2026] Added first-block-cache support for WAN non-unified mode and FLUX (`QEffWanPipeline`, `QEffFluxPipeline`)

<details>
<summary>More</summary>

- [12/2025] Enabled [disaggregated serving](examples/disagg_serving) for GPT-OSS model
- [12/2025] Added support for wav2vec2 Audio Model [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- [12/2025] Added support for diffuser video generation model [WAN 2.2 Model Card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- [12/2025] Added support for diffuser image generation model [FLUX.1 Model Card](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [12/2025] Added support for [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- [12/2025] Added support for [OpenGVLab/InternVL3_5-1B](https://huggingface.co/OpenGVLab/InternVL3_5-1B)
- [12/2025] Added support for Olmo Model [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)
- [10/2025] Added support for Qwen3 MOE Model [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [10/2025] Added support for Qwen2.5VL Multi-Model [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)
- [10/2025] Added support for Mistral3 Multi-Model [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
- [10/2025] Added support for Molmo Multi-Model [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924)
- [06/2025] Added support for Llama4 Multi-Model [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- [06/2025] Added support for Gemma3 Multi-Modal-Model [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- [06/2025] Added support of model `hpcai-tech/grok-1` [hpcai-tech/grok-1](https://huggingface.co/hpcai-tech/grok-1)
- [06/2025] Added support for sentence embedding which improves efficiency, Flexible/Custom Pooling configuration and compilation with multiple sequence lengths, [Embedding model](https://github.com/qualcomm/efficient-transformers/pull/424)
- [04/2025] Support for [SpD, multiprojection heads](https://qualcomm.github.io/efficient-transformers/source/quick_start.html#draft-based-speculative-decoding). Implemented post-attention hidden size projections to speculate tokens ahead of the base model
- [04/2025] [QNN Compilation support](https://github.com/qualcomm/efficient-transformers/pull/374) for AutoModel classes. QNN compilation capabilities for multi-models, embedding models and causal models.
- [04/2025] Added support for separate prefill and decode compilation for encoder (vision) and language models. This feature will be utilized for [disaggregated serving](https://github.com/qualcomm/efficient-transformers/pull/365).
- [04/2025] SwiftKV Support for both [continuous and non-continuous batching execution](https://github.com/qualcomm/efficient-transformers/pull/367) in SwiftKV.
- [04/2025] Support for [GGUF model execution](https://github.com/qualcomm/efficient-transformers/pull/368) (without quantized weights)
- [04/2025] Enabled FP8 model support on [replicate_kv_heads script](https://github.com/qualcomm/efficient-transformers/tree/main/scripts/replicate_kv_head)
- [04/2025] Added support for [gradient checkpointing](https://github.com/qualcomm/efficient-transformers/pull/338) in the finetuning script
- [03/2025] Added support for swiftkv model [Snowflake/Llama-3.1-SwiftKV-8B-Instruct](https://huggingface.co/Snowflake/Llama-3.1-SwiftKV-8B-Instruct)
- [02/2025] [VLMs support](https://github.com/qualcomm/efficient-transformers/pull/267) added for the models [InternVL-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B), [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) and [Mllama](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) *(Mllama support is deprecated — please migrate to Llama-4)*
- [01/2025] [FP8 models support](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127) Added support for inference of FP8 models.
- [01/2025] Added support for [Ibm-Granite] (https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)
- [01/2025] Added support for [Ibm-Granite-Guardian] (https://huggingface.co/ibm-granite/granite-guardian-3.1-8b)
- [11/2024] [finite adapters support](https://github.com/qualcomm/efficient-transformers/pull/153) allows mixed adapter usage for peft models.
- [11/2024] [Speculative decoding TLM](https://github.com/qualcomm/efficient-transformers/pull/119) QEFFAutoModelForCausalLM model can be compiled for returning more than 1 logits during decode for TLM.
- [11/2024] Added support for [Meta-Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [Meta-Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and [Meta-Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [09/2024] [AWQ](https://arxiv.org/abs/2306.00978)/[GPTQ](https://arxiv.org/abs/2210.17323) 4-bit quantized models are supported <br>
- [09/2024] Now we support [PEFT](https://huggingface.co/docs/peft/index) models
- [09/2024] Added support for [Gemma-2-Family](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)<br>
- [09/2024] Added support for [CodeGemma-Family](https://huggingface.co/collections/google/codegemma-release-66152ac7b683e2667abdee11)
- [09/2024] Added support for [Gemma-Family](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)
- [09/2024] Added support for [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [09/2024] Added support for [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [09/2024] Added support for [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
- [09/2024] Added support for [granite-20b-code-base](https://huggingface.co/ibm-granite/granite-20b-code-base-8k)
- [09/2024] Added support for [granite-20b-code-instruct-8k](https://huggingface.co/ibm-granite/granite-20b-code-instruct-8k)
- [09/2024] Added support for [Starcoder1-15B](https://huggingface.co/bigcode/starcoder)
- [08/2024] Added support for inference optimization technique ```continuous batching```
- [08/2024] Added support for [Jais-adapted-70b](https://huggingface.co/inceptionai/jais-adapted-70b)
- [08/2024] Added support for [Jais-adapted-13b-chat](https://huggingface.co/inceptionai/jais-adapted-13b-chat)
- [08/2024] Added support for [Jais-adapted-7b](https://huggingface.co/inceptionai/jais-adapted-7b)
- [06/2024] Added support for [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b)
- [06/2024] Added support for [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)
- [06/2024] Added support for [StarCoder2-15B](https://huggingface.co/bigcode/starcoder2-15b)
- [06/2024] Added support for [Phi3-Mini-4K-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [06/2024] Added support for [Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1)
- [06/2024] Added support for [Vicuna-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)
- [05/2024] Added support for [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) & [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).
- [04/2024] Initial release of [efficient transformers](https://github.com/qualcomm/efficient-transformers) for seamless inference on pre-trained LLMs.
</details>

# Overview

## Train anywhere, Infer on Qualcomm Cloud AI with a Developer-centric Toolchain

This library provides reimplemented blocks of LLMs which are used to make the models functional and highly performant on Qualcomm Cloud AIxxx (AI100, AI200 and so on).
There are several models which can be directly transformed from a pre-trained original form to a deployment ready optimized form.
For other models, there is comprehensive documentation to inspire upon the changes needed and How-To(s).

## Typically for LLMs, the library provides:
1. Reimplemented blocks from Transformers <link> which enable efficient on-device retention of intermediate states.
2. Graph transformations to enable execution of key operations in lower precision
3. Graph transformations to replace some operations to other mathematically equivalent operations
4. Handling for under-flows and overflows in lower precision
5. Patcher modules to map weights of original model's operations to updated model's operations
6. Exporter module to export the model source into a `ONNX` Graph.
7. Sample example applications and demo notebooks
8. Unit test templates. 

**It is mandatory for each Pull Request to include tests such as**:
1. If the PR is for adding support for a model, the tests should include successful execution of the model post changes (the changes included as part of PR) on Pytorch and ONNXRT. Successful exit criteria is MSE between output of original model and updated model.
2. If the PR modifies any common utilities, tests need to be included to execute tests of all models included in the library.


## Quick Installation
```bash

# Create Python virtual env and activate it. (Recommended Python 3.12)
sudo apt install python3.12-venv
python3.12 -m venv qeff_env
source qeff_env/bin/activate
pip install -U pip

# Clone and Install the QEfficient repository from the mainline branch
pip install git+https://github.com/qualcomm/efficient-transformers

# Clone and Install the QEfficient repository from a specific branch, tag or commit by appending @ref
# Release branch (e.g., release/v1.20.0):
pip install "git+https://github.com/qualcomm/efficient-transformers@release/v1.20.0"

# Or build wheel package using the below command.
pip install build wheel
python -m build --wheel --outdir dist
pip install dist/qefficient-0.0.1.dev0-py3-none-any.whl

``` 

For more details about using ``QEfficient`` via Cloud AIxxx (AI100, AI200 and so on) Apps SDK, visit [Linux Installation Guide](https://qualcomm.github.io/efficient-transformers/source/installation.html)


## Documentation

* [Quick Start Guide](https://qualcomm.github.io/efficient-transformers/source/quick_start.html)
* [QEFF API](https://qualcomm.github.io/efficient-transformers/source/qeff_autoclasses.html)
* [Validated Models](https://qualcomm.github.io/efficient-transformers/source/validate.html)
* [Models coming soon](https://qualcomm.github.io/efficient-transformers/source/validate.html#models-coming-soon)

> Note: More details are here: https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/

## Acknowledgements
Thanks to:
* HuggingFace transformers for work in LLM GenAI modeling implementation
*  ONNX, Pytorch,  ONNXruntime community.

## Support
If you run into any problems with the code, please file Github issues directly to this repo.

## Contributing
This project welcomes contributions and suggestions. Please check the License. Integration with a CLA Bot is underway.
