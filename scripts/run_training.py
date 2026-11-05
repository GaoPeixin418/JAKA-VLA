"""Run upstream LeRobot training and record resource use for this exact run."""
import json
import sys
import time
from pathlib import Path
import torch
from lerobot.scripts.lerobot_train import main

output = next(Path(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--output_dir='))
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
ok = False
try:
    main()
    torch.cuda.synchronize()
    ok = True
finally:
    report = {'completed': ok, 'wall_seconds_including_load_and_save': time.perf_counter() - start,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    if output.exists():
        (output / 'resource_profile.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
