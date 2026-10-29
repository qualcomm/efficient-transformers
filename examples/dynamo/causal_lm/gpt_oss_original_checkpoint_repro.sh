#!/usr/bin/env bash
set -euo pipefail

# Reproduce original-checkpoint GPT-OSS export and the current QAIC dtype error.
# Usage:
#   ./gpt_oss_original_checkpoint_repro.sh                 # decode/BMM
#   ./gpt_oss_original_checkpoint_repro.sh prefill-ep       # prefill MoE EP

MODEL_ID="${MODEL_ID:-tiny-random/gpt-oss-bf16}"
HF_HUB_CACHE="${HF_HUB_CACHE:-/home/huggingface_hub}"
QEFF_HOME="${QEFF_HOME:-/tmp/qeff-gptoss-original}"
PYENV_VERSION="${PYENV_VERSION:-qeff.einsum}"
MODE="${1:-decode}"

case "${MODE}" in
    decode)
        EXPORT_DIR="${QEFF_HOME}/decode"
        ;;
    prefill-ep)
        EXPORT_DIR="${QEFF_HOME}/prefill-ep"
        ;;
    *)
        echo "Usage: $0 [decode|prefill-ep]" >&2
        exit 2
        ;;
esac

export HF_HUB_CACHE
export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME

mkdir -p "${EXPORT_DIR}"

exported_log="$(mktemp)"
trap 'rm -f "${exported_log}"' EXIT

MODEL_ID="${MODEL_ID}" MODE="${MODE}" EXPORT_DIR="${EXPORT_DIR}" \
PYENV_VERSION="${PYENV_VERSION}" pyenv exec python - <<'PY' | tee "${exported_log}"
import os

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained(
    os.environ["MODEL_ID"],
    weight_free=True,
    use_original_checkpoint=True,
)

export_kwargs = {
    "use_onnx_subfunctions": True,
    "offload_pt_weights": False,
}
if os.environ["MODE"] == "prefill-ep":
    export_kwargs.update(
        {
            "prefill_only": True,
            "prefill_seq_len": 4,
            "qaic_config": {"moe_config": {"expert_parallel_chunk_size": 4}},
        }
    )
else:
    export_kwargs["prefill_only"] = False

onnx_path = model.export(os.environ["EXPORT_DIR"], **export_kwargs)
print(f"EXPORT_ONNX={onnx_path}")
PY

onnx_path="$(sed -n 's/^EXPORT_ONNX=//p' "${exported_log}" | tail -n 1)"
if [[ -z "${onnx_path}" || ! -f "${onnx_path}" ]]; then
    echo "Could not find the exported ONNX path." >&2
    exit 1
fi

echo
echo "Original-checkpoint weight spec:"
spec_path="$(dirname "${onnx_path}")/weight_spec.json"
cat "${spec_path}"
checkpoint_dir_name="$(PYENV_VERSION="${PYENV_VERSION}" pyenv exec python -c \
    'import json, sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["model_id"]).name)' \
    "${spec_path}")"
checkpoint_link="$(dirname "${onnx_path}")/${checkpoint_dir_name}"
if [[ -L "${checkpoint_link}" ]]; then
    echo
    echo "External checkpoint link (symlink, not a copy):"
    readlink -f "${checkpoint_link}"
fi
echo
echo "Compiling ${onnx_path}"

/opt/qti-aic/exec/qaic-compile \
    -aic-hw \
    -aic-hw-version=ai100 \
    -m="${onnx_path}" \
    -retained-state \
    -convert-to-fp16 \
    -aic-num-cores=4 \
    -sub-functions \
    -onnx-define-symbol=batch_size,1 \
    -onnx-define-symbol=seq_len,4 \
    -onnx-define-symbol=ctx_len,8 \
    -onnx-define-symbol=sliding_window,128 \
    -aic-binary-dir="${QEFF_HOME}/${MODE}-qpc"
