# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import re
import subprocess

# Known Qualcomm Cloud AI PCI device IDs (from lspci output).
# NOTE: These device IDs may change with future hardware revisions; update
# both sets when new Qualcomm Cloud AI hardware variants are introduced.
_AIC200_PCI_DEVICE_IDS = {"a110"}
_AIC100_PCI_DEVICE_IDS = {"a100", "a0dc"}


def _detect_hw_version_via_lspci() -> str:
    """Probe PCI devices to detect Qualcomm Cloud AI hardware generation.

    Runs ``lspci`` and filters lines containing ``"Qualcomm"``, then matches
    the device ID token to distinguish AI200 from AI100.  Returns ``"ai200"``
    when an AI200 device is found, ``"ai100"`` when an AI100 device is found,
    and ``"ai100"`` as the safe default when ``lspci`` is unavailable or no
    matching device is found.

    Returns:
        str: ``"ai200"`` if an AI200 PCI device is detected, otherwise ``"ai100"``.
    """
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
    except Exception:
        return "ai100"

    for line in output.splitlines():
        # lspci lines look like:
        #   "76:00.0 Processing accelerators: Qualcomm Technologies, Inc Device a110 (rev 01)"
        #   "03:00.0 Processing accelerators: Qualcomm Device a100"
        if "qualcomm" not in line.lower():
            continue
        device_match = re.search(r"\bDevice\s+([0-9a-fA-F]+)", line, re.IGNORECASE)
        if device_match:
            device_id = device_match.group(1).lower()
            if device_id in _AIC200_PCI_DEVICE_IDS:
                return "ai200"
            if device_id in _AIC100_PCI_DEVICE_IDS:
                return "ai100"

    return "ai100"


def get_default_aic_hw_version() -> str:
    """Detect the AIC hardware version from the first available device.

    Runs ``qaic-util -q`` and inspects the ``NSP IMAGE_VARIANT`` field of the
    first device (QID 0) to determine whether the hardware is ``ai100`` or
    ``ai200``.  When ``qaic-util`` is unavailable, falls back to probing PCI
    devices via ``lspci | grep Qualcomm`` and matching known Qualcomm Cloud AI
    PCI device IDs.  Returns ``"ai100"`` when neither method yields a result.

    Returns:
        str: ``"ai200"`` if an AI200 device is detected, otherwise ``"ai100"``.
    """
    qaic_util = "/opt/qti-aic/tools/qaic-util"
    try:
        result = subprocess.run(
            [qaic_util, "-q"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
    except Exception:
        return _detect_hw_version_via_lspci()

    match = re.search(r"NSP IMAGE_VARIANT\s*:\s*(\S+)", output)
    if match:
        variant = match.group(1).upper()
        if "AIC200" in variant:
            return "ai200"
        if "AIC100" in variant:
            return "ai100"
    return _detect_hw_version_via_lspci()


class AicHwDefaultsResolver:
    """Singleton resolving compiler/runtime defaults that vary by AIC hardware generation (e.g. AI100 vs AI200)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _is_ai200(aic_hw_version: str) -> bool:
        return aic_hw_version == "ai200"

    def get_num_cores(self, aic_hw_version: str) -> int:
        return 4 if self._is_ai200(aic_hw_version) else 16

    def get_vtcm_size_threshold(self, aic_hw_version: str) -> float:
        vtcm_size_mb = 16 if self._is_ai200(aic_hw_version) else 8
        return vtcm_size_mb * 1024 * 1024 * 0.75


def get_default_num_cores(aic_hw_version: str | None = None) -> int:
    """Return the default AIC compiler ``num_cores`` for the given (or detected) hardware version."""
    return AicHwDefaultsResolver().get_num_cores(aic_hw_version or get_default_aic_hw_version())


def get_default_vtcm_size_threshold(aic_hw_version: str | None = None) -> float:
    """Return the default VTCM size threshold (bytes) for the given (or detected) hardware version."""
    return AicHwDefaultsResolver().get_vtcm_size_threshold(aic_hw_version or get_default_aic_hw_version())


def get_default_weight_free_dtype(aic_hw_version: str | None = None) -> str:
    """Return the default weight-free graph dtype for the selected AIC generation."""
    return "bfloat16" if (aic_hw_version or get_default_aic_hw_version()) == "ai200" else "float16"


def validate_weight_free_target_dtype(target_dtype, aic_hw_version: str | None = None) -> None:
    """Reject weight-free dtypes that the selected AIC generation cannot compile."""
    hw_version = aic_hw_version or get_default_aic_hw_version()
    dtype_name = str(target_dtype).removeprefix("torch.")
    if hw_version != "ai200" and dtype_name == "bfloat16":
        raise ValueError(
            "weight-free target_dtype=torch.bfloat16 is not supported on AI100. "
            "Use target_dtype=torch.float16 instead; BF16 export may succeed but compilation will fail."
        )
