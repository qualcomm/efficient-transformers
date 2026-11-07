# DFlash-DSpark

DSpark support is integrated into the existing DFlash text and vision entry points:

- `examples/performance/dflash/basic_inference_text.py`
- `examples/performance/dflash/basic_inference_vision.py`
- `examples/performance/dflash/utils.py`

No DSpark-specific example Python files are required.

## Supported checkpoints

| Name | Target | Draft | Mode |
|---|---|---|---|
| `Qwen3-4B-DSpark` | `Qwen/Qwen3-4B` | `deepseek-ai/dspark_qwen3_4b_block7` | Text |
| `Qwen3-8B-DSpark` | `Qwen/Qwen3-8B` | `deepseek-ai/dspark_qwen3_8b_block7` | Text |
| `Qwen3-14B-DSpark` | `Qwen/Qwen3-14B` | `deepseek-ai/dspark_qwen3_14b_block7` | Text |
| `Gemma4-12B-DSpark` | `google/gemma-4-12B-it` | `deepseek-ai/dspark_gemma4_12b_block7` | Text |
| `Qwen3.8-27B-DSpark` | `Qwen/Qwen3.8-27B` | `RadixArk/Qwen3.8-27B-DSpark` | Text and vision |

The DeepSpec block-7 checkpoints propose seven draft tokens and verify an
eight-token `[anchor] + proposals` block.

## Environment

```bash
source /path/to/qeff-venv/bin/activate
export HF_HUB_CACHE=/path/to/huggingface/cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME=/path/to/qeff/cache/dspark
export TMPDIR=/path/to/local/compiler/tmp
mkdir -p "$QEFF_HOME" "$TMPDIR"
```

Use a local `TMPDIR` that supports `fallocate`. Network filesystems can fail
constant serialization with `Operation not supported`.

## Text compile and inference

```bash
python examples/performance/dflash/basic_inference_text.py \
  --technique dspark \
  --model_name Qwen3-4B-DSpark \
  --prompt "Write a Python function that returns the nth Fibonacci number." \
  --tlm_devices 0,1,2,3 --dlm_devices 4,5,6,7 \
  --tlm_cores 8 --dlm_cores 8 \
  --prefill_seq_len 128 --ctx_len 4096 --generation_len 256
```

The command compiles missing target and draft QPCs, writes a `manifest.json`,
then runs greedy DSpark inference. Reuse the manifest with `--manifest`.

## Vision compile and inference

Vision DSpark currently uses the Qwen3.8 multimodal target.

```bash
python examples/performance/dflash/basic_inference_vision.py \
  --technique dspark \
  --model_name Qwen3.8-27B-DSpark \
  --image --image-path /path/to/image.png \
  --image_prompt "Describe this image." \
  --tlm_devices 0,1,2,3 --dlm_devices 4,5,6,7 \
  --vision_devices 0,1,2,3 \
  --height 448 --width 448 --generation_len 256
```

## Offline smoke checks

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/performance/dflash/basic_inference_text.py \
  --technique dspark --model_name Qwen3.8-27B-DSpark \
  --prompt smoke --smoke-test --generation_len 12

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/performance/dflash/basic_inference_vision.py \
  --technique dspark --model_name Qwen3.8-27B-DSpark \
  --image --smoke-test --generation_len 12
```

These checks compare tiny Qwen3.5 text and vision DSpark output against Hugging
Face greedy decoding. They do not validate full checkpoints or QAIC performance.

## Validated baselines

Existing Qwen3.8-27B QAIC results:

| Dataset | Samples | TLM rounds/s | DLM draft tok/s | SPD generated tok/s | Accepted length |
|---|---:|---:|---:|---:|---:|
| HumanEval | 164 | 9.6036 | 296.3083 | 49.1252 | 6.2854 |
| MathVision testmini | 150 | 9.6203 | 302.6226 | 34.4920 | 4.3894 |

Configuration: target/draft/vision `TS4C8`; MathVision image specialization
`352x544`.

HumanEval validation for the official DeepSpec checkpoints uses 164 samples,
512 maximum generated tokens, and target/draft `TS4C8` on separate devices:

| Draft checkpoint | TLM rounds/s | DLM draft tok/s | SPD generated tok/s | Accepted length |
|---|---:|---:|---:|---:|
| `dspark_qwen3_4b_block7` | 58.0562 | 506.9656 | 181.0816 | 5.6427 |
| `dspark_qwen3_8b_block7` | 39.7573 | 470.8052 | 142.8757 | 5.7449 |

The Qwen3-14B and Gemma4-12B rows require fresh manifests and are intentionally
omitted until their full 164-sample runs complete.

## Implementation notes

- Qwen3 and Gemma4 use ordinary retained-KV overwrite after rejection.
- Qwen3.5 linear attention cannot roll back convolution and recurrent state.
  It uses the PR #1472 fused replay/verification protocol with one width-16
  target call per speculation round.
- The trained `fc.weight` and `hidden_norm.weight` are moved to the target.
  The draft receives projected features with target hidden width.
- Target projector weights are scaled before FP16 lowering and normalized with
  the compiler-supported RMSNorm path.
- Draft Markov correction and greedy argmax execute on device, avoiding FP32
  vocabulary-logit transfers to the host.
- Compilation enables MXFP6 constant matmul, MXINT8 retained cache, MOS 1, and
  split model I/O for dual-QPC targets.

## Issues resolved

- **Disk quota:** set `QEFF_HOME` to the large DSpark cache and keep test caches
  disposable under `/tmp`.
- **Compiler temporary files:** move `TMPDIR` to a local filesystem supporting
  sparse allocation. QPC compilation also needs local atomic-renames, so both
  `QEFF_HOME` and `TMPDIR` must avoid NFS-backed paths.
- **Large target QPC:** MXFP6 target weights reduced the Qwen3.8 target QPC from
  about 51 GB to about 22 GB.
- **Slow Qwen3.5 TLM:** replacing separate replay and verification calls with the
  fused PR #1472 path raised the target from the legacy ~3.4 rounds/s result to
  about 9.6 rounds/s on TS4C8.
- **Slow prompt draft ingestion:** add a wide target-feature specialization so a
  128-token prompt does not require one draft call per eight features.
- **FP16 normalization overflow:** scale the target projector and use stable
  RMSNorm lowering; this restored acceptance quality.
