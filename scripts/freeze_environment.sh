#!/usr/bin/env bash
set -euo pipefail
source /data/jaka-vla/env.sh
mkdir -p /data/jaka-vla/manifests
/data/jaka-vla/envs/vla/bin/uv pip freeze --python /data/jaka-vla/envs/robotwin/bin/python > /data/jaka-vla/manifests/robotwin-packages.txt
/data/jaka-vla/envs/vla/bin/python -m pip freeze > /data/jaka-vla/manifests/vla-packages.txt
git -C /data/jaka-vla/repos/RoboTwin diff > /data/jaka-vla/manifests/robotwin-timing-proprioception.patch
git -C /data/jaka-vla/repos/RoboTwin/XPolicyLab diff > /data/jaka-vla/manifests/xpolicylab-zu7.patch
python3 - <<'PY'
import json, subprocess
from pathlib import Path
root = Path('/data/jaka-vla')
sources = {}
for name, rel in [('robotwin', 'repos/RoboTwin'), ('xpolicylab', 'repos/RoboTwin/XPolicyLab'),
                  ('curobo', 'repos/RoboTwin/envs/curobo'), ('lerobot', 'repos/lerobot')]:
    sources[name] = subprocess.check_output(['git', '-C', str(root / rel), 'rev-parse', 'HEAD'], text=True).strip()
sources['smolvla_weights'] = '5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
sources['smolvlm_backbone'] = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
(root / 'manifests/sources.json').write_text(json.dumps(sources, indent=2))
PY
