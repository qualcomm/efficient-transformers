# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

# Jenkins overrides the ordinary AI100 test compile default to leave NSP headroom.
NUM_AIC_CORES = int(os.environ.get("NUM_AIC_CORES", "16"))
