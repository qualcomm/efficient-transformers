# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Continuous Batching Example for Gemma4 Vision Model

Demonstrates continuous batching with the Gemma4 vision-language model, processing
multiple image-text pairs simultaneously.  The pipeline is:

  1.  Load processor and reduced-layer model config (for testing).
  2.  Build the QEff model with  continuous_batching=True  and  kv_offload=True
      (dual-QPC: vision encoder + language decoder compiled separately).
  3.  Compile both QPCs with  full_batch_size  to enable CB scheduling.
  4.  Call  generate  with lists of images and prompts; the runtime fills the
      full_batch_size slots and streams completions as slots free up.

To run on real Gemma4 weights remove the layer-reduction block and set the
NUM_CORES / NUM_DEVICES / compilation flags to match your hardware.
"""

from gemma4_utils import normalize_generated_ids, remove_fp16clip_transform_if_disabled
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-E2B-it"

# ---------------------------------------------------------------------------
# Continuous-batching parameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 1  # Per-slot prefill batch size
FULL_BATCH_SIZE = 4  # Total concurrent CB slots

# ---------------------------------------------------------------------------
# Sequence-length budget
# ---------------------------------------------------------------------------
PREFILL_SEQ_LEN = 256  # Must be >= longest tokenised vision prompt
CTX_LEN = 2048
GENERATION_LEN = 100

# ---------------------------------------------------------------------------
# Testing knobs: reduce layers for fast end-to-end validation
# ---------------------------------------------------------------------------
NUM_LANG_HIDDEN_LAYER = 2
NUM_VISION_HIDDEN_LAYER = 2

# ---------------------------------------------------------------------------
# Compiler settings
# ---------------------------------------------------------------------------
NUM_CORES = 16
NUM_DEVICES = 2
NODE_PRECISION_INFO = True  # Auto-generate Gemma4 NPI file for mixed precision

# ---------------------------------------------------------------------------
# Sample inputs (FULL_BATCH_SIZE prompts / images)
# ---------------------------------------------------------------------------
IMAGE_URLS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
]

PROMPTS = [
    "Can you describe the image in detail?",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "What colors are predominant in the image?",
]


def _apply_reduced_layer_config(config, num_lang_layers: int, num_vision_layers: int):
    """Shrink layer counts so the model fits in CPU RAM during testing."""
    config.text_config.num_hidden_layers = num_lang_layers
    config.vision_config.num_hidden_layers = num_vision_layers

    if hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]

    if hasattr(config.text_config, "num_kv_shared_layers"):
        # Avoid invalid first_kv_shared_layer_idx=0 edge case with few layers.
        config.text_config.num_kv_shared_layers = 0

    return config


def main():
    # ------------------------------------------------------------------
    # STEP 1: Processor / tokenizer
    # ------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # ------------------------------------------------------------------
    # STEP 2: Config (with layer reduction for testing)
    # ------------------------------------------------------------------
    config = AutoConfig.from_pretrained(MODEL_ID)
    # For Testing Purpose Only
    config = _apply_reduced_layer_config(config, NUM_LANG_HIDDEN_LAYER, NUM_VISION_HIDDEN_LAYER)

    # ------------------------------------------------------------------
    # STEP 3: Build QEff model with continuous batching enabled
    # ------------------------------------------------------------------
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,  # Dual-QPC: vision encoder + LM decoder
        ignore_mismatched_sizes=True,
        continuous_batching=True,  # Enable CB scheduling
    )
    remove_fp16clip_transform_if_disabled(qeff_model, effective_fp16clip=True)

    # ------------------------------------------------------------------
    # STEP 4: Compile both QPCs for continuous batching
    # ------------------------------------------------------------------
    qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        batch_size=BATCH_SIZE,
        full_batch_size=FULL_BATCH_SIZE,  # Required for CB mode
        num_cores=NUM_CORES,
        num_devices=NUM_DEVICES,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
        split_model_io=True,
        node_precision_info=NODE_PRECISION_INFO,
        use_onnx_subfunctions=False,
    )

    # ------------------------------------------------------------------
    # STEP 5: Generate — CB runtime fills all FULL_BATCH_SIZE slots
    # ------------------------------------------------------------------
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=PROMPTS,
        processor=processor,
        images=IMAGE_URLS,
        generation_len=GENERATION_LEN,
    )

    # ------------------------------------------------------------------
    # STEP 6: Decode and print results
    # ------------------------------------------------------------------
    qeff_ids = normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
    generated_texts = tokenizer.batch_decode(qeff_ids, skip_special_tokens=True)

    for i, text in enumerate(generated_texts):
        print(f"\n--- Response [{i}] ---")
        print(text)

    print("\nExecution info:")
    print(output)


if __name__ == "__main__":
    main()
