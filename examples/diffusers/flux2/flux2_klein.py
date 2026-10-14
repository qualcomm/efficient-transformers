# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
FLUX.2-klein-4B Image Editting Example

This example demonstrates how to use the QEffFlux2KleinPipeline to edit
images using the FLUX.2-klein-4B model from Black Forest Labs.

"""

from PIL import Image
import os
import glob
import torch

from QEfficient import QEffFlux2KleinPipeline

# ---------------------------------------------------------------------------
# Initialize the FLUX.2-klein-4B pipeline from pretrained weights
# ---------------------------------------------------------------------------
pipeline = QEffFlux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B")

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

prompt = (
    "Preserve the tiger exactly as in the source image with no changes to its pose, fur pattern, stripe arrangement, facial features, expression, body shape, scale, color, sharpness, anatomy, or position. Only modify the background environment."
    "Transform the jungle into a dramatic monsoon rainforest scene with gentle rainfall, wet leaves, mist between the trees, atmospheric fog, puddles reflecting light, volumetric light rays filtering through storm clouds, cool blue-green color grading, cinematic lighting, realistic weather effects, enhanced depth and mood."
    "The tiger must remain perfectly unchanged and crisply focused while the surrounding environment reflects the new weather and lighting conditions. Photorealistic wildlife photography, seamless integration, natural shadows and reflections, ultra-detailed, high quality, no artifacts, no text, no watermark."
)

img = Image.open("tiger.png")
result = pipeline(
    image=img,  # Commented out for testing text-to-image without conditioning
    prompt=prompt,
    height=480,
    width=832,
    guidance_scale=1.0,
    num_inference_steps=4,
    max_sequence_length=512,
    generator=torch.Generator(device="cpu").manual_seed(42),
)

# ---------------------------------------------------------------------------
# Save the generated image
# ---------------------------------------------------------------------------
if result.images:
    result.images[0].save("tiger_edit.png")
