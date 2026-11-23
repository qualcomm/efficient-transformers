# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from collections.abc import Mapping
from importlib import metadata

from packaging.specifiers import SpecifierSet

DYNAMO_EXPORT_REQUIREMENTS = {
    "torch": "==2.13.0",
    "torchvision": "==0.28.0",
    "accelerate": "==1.10.1",
    "compressed-tensors": "==0.17.0",
    "onnxscript": "==0.6.2",
}
DYNAMO_REQUIREMENTS_INSTALL_COMMAND = "pip install -r examples/dynamo/causal_lm/requirements.txt"


def validate_dynamo_export_requirements(feature_name: str = "dynamo=True") -> None:
    """Validate the installed environment against the dynamo export package pins."""
    validate_runtime_requirements(
        DYNAMO_EXPORT_REQUIREMENTS,
        feature_name=feature_name,
        install_command=DYNAMO_REQUIREMENTS_INSTALL_COMMAND,
    )


def validate_runtime_requirements(
    requirements: Mapping[str, str],
    feature_name: str,
    install_command: str,
) -> None:
    """Raise when required packages are missing or version-incompatible."""
    mismatches = [_requirement_mismatch(package, specifier) for package, specifier in requirements.items()]
    mismatches = [mismatch for mismatch in mismatches if mismatch]

    if mismatches:
        details = "\n".join(f"  - {mismatch}" for mismatch in mismatches)
        raise AssertionError(
            f"{feature_name} requires the Dynamo export environment, but package validation failed.\n"
            "Mismatched packages:\n"
            f"{details}\n"
            "Install or repair the environment with:\n"
            f"  {install_command}"
        )


def _requirement_mismatch(package: str, specifier: str) -> str | None:
    """Return a human-readable mismatch for one package, or ``None`` when it is satisfied."""
    try:
        installed_version = metadata.version(package)
    except metadata.PackageNotFoundError:
        return f"{package}: not installed, expected {specifier}"

    if not SpecifierSet(specifier).contains(installed_version, prereleases=True):
        return f"{package}: installed {installed_version}, expected {specifier}"

    return None
