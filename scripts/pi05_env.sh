#!/usr/bin/env bash
# Source for the independent pinned OpenPI environment.
source /data/jaka-vla/env.sh
export HF_LEROBOT_HOME=/data/jaka-vla/data/openpi
export OPENPI_DATA_HOME=/data/pi05-cache/models
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/data/jaka-vla:/data/jaka-vla/repos/openpi/src:/data/jaka-vla/repos/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
