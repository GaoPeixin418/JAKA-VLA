#!/usr/bin/env bash
set -euo pipefail
source /data/jaka-vla/env.sh
source /data/jaka-vla/envs/robotwin/bin/activate
cd /data/jaka-vla/repos/RoboTwin
python /data/jaka-vla/scripts/configure_smoke.py
python /data/jaka-vla/scripts/patch_sim_timing.py
python scripts/update_embodiment_config_path.py
timeout 1200 python -u scripts/collect_data.py handover_block jaka_smoke
