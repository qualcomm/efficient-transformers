# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from pathlib import Path

QEFF_HOME: Path = None
if "QEFF_HOME" in os.environ:
    QEFF_HOME = Path(os.environ["QEFF_HOME"])
elif "XDG_CACHE_HOME" in os.environ:
    QEFF_HOME = Path(os.environ["XDG_CACHE_HOME"]) / "qeff_models"
else:
    QEFF_HOME = Path("~/.cache/qeff_models").expanduser()

QEFF_CHECKPOINT_HOME: Path | None = None
_wf_home = os.environ.get("QEFF_CHECKPOINT_HOME") or os.environ.get("QEFF_WF_HOME")
if _wf_home:
    QEFF_CHECKPOINT_HOME = Path(_wf_home)
