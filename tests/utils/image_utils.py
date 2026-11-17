# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple

import requests
from PIL import Image

DEFAULT_TEST_IMAGE_SIZE = (536, 354)
REPO_TEST_IMAGE_PATH = Path(__file__).resolve().parents[2] / "docs/image/QualcommAIRack.png"


def load_test_image(
    image_source: Any,
    size: Optional[Tuple[int, int]] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
) -> Image.Image:
    """Load a test image from PIL, URL, or file path.

    If an external URL returns invalid/unavailable bytes, fall back to a checked-in
    repo image at the requested size (or the default test size). This keeps the
    processor-derived vision token count compatible with the fixed QAIC test
    specialization.
    """
    is_url = isinstance(image_source, str) and image_source.startswith(("http://", "https://"))
    try:
        if isinstance(image_source, Image.Image):
            image = image_source.convert("RGB")
        elif is_url:
            requester = session or requests
            response = requester.get(image_source, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_source).convert("RGB")
    except Exception as exc:
        if not is_url:
            raise

        print(
            f"Failed to load test image from {image_source}; "
            f"using repo fallback image {REPO_TEST_IMAGE_PATH}. Error: {exc}"
        )
        image = Image.open(REPO_TEST_IMAGE_PATH).convert("RGB")
        if size is None:
            size = DEFAULT_TEST_IMAGE_SIZE

    return image.resize(size) if size is not None else image
