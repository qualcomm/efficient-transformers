# Fetaures Enablement Guide
Below guide highlights the steps to enable supported features in QEfficient.

(id-continuous-batching)=
## Continuous Batching

Users can compile a model utilizing the continuous batching feature by specifying full_batch_size <full_batch_size_value> in the infer and compiler APIs. If full_batch_size is not provided, the model will be compiled in the regular way.

When enabling continuous batching, batch size should not be specified.

Users can leverage multi-Qranium and other supported features along with continuous batching.

```bash
python -m QEfficient.cloud.infer --model_name TinyLlama/TinyLlama_v1.1 --prompt_len 32 --ctx_len 128 --num_cores 16 --device_group [0] --prompt "My name is|The flat earth theory is the belief that|The sun rises from" --mxfp6 --mos 1 --aic_enable_depth_first --full_batch_size 3
```

---

(id-multi-qranium-inference)=
## Multi-Qranium Inference

You can also enable MQ, just based on the number of devices. Based on the `--device-group` as input it will create TS config on the fly. If `--device-group [0,1]` it will create TS config for 2 devices and use it for compilation, if `--device-group [0]` then TS compilation is skipped and single soc execution is enabled.

```bash
python -m QEfficient.cloud.infer --model_name Salesforce/codegen-2B-mono --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device-group [0,1] --prompt "def fibonacci(n):" --mos 2 --aic_enable_depth_first
```

Above step will save the `qpc` files under `efficient-transformers/qeff_models/{model_card_name}`, you can use the execute API to run for different prompts. This will automatically pick the pre-compiled `qpc` files.

```bash
python -m QEfficient.cloud.execute --model_name Salesforce/codegen-2B-mono --qpc-path qeff_models/Salesforce/codegen-2B-mono/qpc_16cores_1BS_32PL_128CL_2devices_mxfp6/qpcs --prompt "def binary_search(array: np.array, k: int):" --device-group [0,1]
```

To disable MQ, just pass single soc like below, below step will compile the model again and reuse the `ONNX` file as only compilation argument are different from above commands.

```bash
python -m QEfficient.cloud.infer --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 16 --device-group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first
```

---

(id-weight-free-export)=
## Weight-Free CausalLM Export

Weight-free export traces a CausalLM model with meta tensors and keeps model
weights outside the ONNX graph. The export writes a `weight_spec.json` sidecar
beside the ONNX file so compile and ONNX Runtime validation can load the real
checkpoint tensors only when needed.

Use the `compile()` flag when exporting large CausalLM models where holding all
weights in memory during ONNX export is not practical:

```Python
from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct", weight_free=True)
qpc_path = model.compile(
    prefill_seq_len=128,
    ctx_len=128,
    num_cores=16,
    dynamo=True,
    use_onnx_subfunctions=True,
)
```

`weight_free=True` forces the dynamo export path. When constructing a
QEfficient wrapper manually from a meta-device model, pass
`pretrained_model_name_or_path` so the exporter can resolve the checkpoint files
referenced by `weight_spec.json`.

For ONNX Runtime validation, use
`QEfficient.exporter.weight_free.load_weight_free_ort_inputs()` to merge runtime
inputs with the external checkpoint tensors. The helper resolves weights from an
explicit `weights_root` argument first. If that is not provided, it checks
`AIC_EXTERNAL_DATA_ROOT`, the ONNX export directory, and the original checkpoint
location recorded in `weight_spec.json`.

---

(id-proxy-model-export)=
## Proxy Model Export

Proxy model export builds a representative graph for compile/runtime bring-up
without keeping the model's embedding and output projection work in the compiled
graph. Enable it with `enable_proxy=True` when loading a QEfficient model.

When proxy mode is enabled, QEfficient replaces the language model's embedding
layer and LM head with proxy modules. For models with repeated decoder-layer
patterns, QEfficient also reduces the language stack to the smallest
representative layer count that still includes repeated calls for each layer
type. This is useful for compiler validation, artifact generation, and external
harnesses that need a lightweight graph matching the model's runtime interface.

```Python
from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    enable_proxy=True,
    trust_remote_code=True,
)

qpc_path = model.compile(
    prefill_seq_len=32,
    ctx_len=128,
    use_onnx_subfunctions=True,
)
```

Use `num_hidden_layers` to override the automatically selected proxy layer
count when a specific reduced depth is required:

```Python
model = QEFFAutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    enable_proxy=True,
    num_hidden_layers=2,
    trust_remote_code=True,
)
```

The text generation example exposes the same flow from the CLI:

```bash
python examples/text_generation/basic_inference.py \
  --model-name Qwen/Qwen2-1.5B-Instruct \
  --enable-proxy \
  --use-onnx-subfunctions \
  --compile-only
```

Proxy mode can also be combined with `artifacts=True` to emit compiler and
runner replay files without invoking the compiler or runtime:

```bash
python examples/text_generation/basic_inference.py \
  --model-name Qwen/Qwen2-1.5B-Instruct \
  --enable-proxy \
  --use-onnx-subfunctions \
  --artifacts
```

---

(id-qnn-compilation-via-python-api)=
## QNN Compilation via Python API

Users can also use python API to export, compile and execute onnx models using QNN SDK.

```Python
# We can now export the modified models to ONNX framework
# This will generate single ONNX Model for both Prefill and Decode Variations which are optimized for
# Cloud AI 100 Platform.
from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

# Model-Card name (This is HF Model Card name) : https://huggingface.co/gpt2-xl
model_name = "gpt2"  # Similar, we can change model name and generate corresponding models, if we have added the support in the lib.

qeff_model = AutoModelForCausalLM.from_pretrained(model_name)

qnn_config_file_path = "QEfficient/compile/qnn_config.json"

generated_qpc_path = qeff_model.compile(
    num_cores=14,
    mxfp6=True,
    enable_qnn=True,
    qnn_config = qnn_config_file_path # QNN compilation configuration is passed.
)
```
---

(id-draft-based-speculative-decoding)=
## Draft-Based Speculative Decoding
Draft-based speculative decoding is a technique where a small Draft Language Model (DLM) makes `num_speculative_tokens` autoregressive speculations ahead of the Target Language Model (TLM). The objective is to predict what the TLM would have predicted if it would have been used instead of the DLM. This approach is beneficial when the autoregressive decode phase of the TLM is memory bound and thus, we can leverage the extra computing resources of our hardware by batching the speculations of the DLM as an input to TLM to validate the speculations.

To export and compile both DLM/TLM, add corresponding `qaic_config` and `num_speculative_tokens` for TLM and export DLM as you would any other QEfficient LLM model:

```Python
from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

tlm_name = "meta-llama/Llama-2-70b-chat-hf"
dlm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
k = 3 # DLM will make `k` speculations
qaic_config = dict(speculative_model_type="target")
tlm = AutoModelForCausalLM.from_pretrained(tlm_name, qaic_config=qaic_config)
dlm = AutoModelForCausalLM.from_pretrained(dlm_name)
tlm.compile(num_speculative_tokens=k)
dlm.compile()
```

The `qaic_config` dictionary is fed during the instantiation of the model because slight changes to the ONNX graph are required. Once complete, the user can specify `num_speculative_tokens` to define the actual number of speculations that the TLM will take as input during the decode phase. As for the DLM, no new changes are required at the ONNX or compile level.

## Export artifacts without compiler or runtime execution

CausalLM and ImageTextToText external harnesses can request the files for compilation and one `qaic-runner` invocation without executing either tool:

```python
compile_dir = model.compile(
    prefill_seq_len=32,
    ctx_len=128,
    artifacts=True,
)

io_dir = model.generate(
    tokenizer=tokenizer,
    prompts=["Hello"],
    artifacts=True,
)
```

The returned compile directory (`compile_dir`) contains `qaic-compile.sh`, `specializations.json`, `custom_io.yaml` and node precision info (NPI) files when required, and the compiler hash inputs. The returned generation directory contains `aic_batch_io.json` and raw host inputs under `data/`.

Image-text-to-text generation also requires an example `processor`, `images`, and `prompts`. For dual-QPC models, set exactly one of `skip_vision=True` or `skip_lang=True` to emit one independently replayable stage.
