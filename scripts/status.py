"""Print compact job progress without reading credentials or unrelated files."""
import json
import os
from pathlib import Path

root = Path('/data/jaka-vla')
status = root / 'results/pilot_status.txt'
pid_path = root / 'results/pilot.pid'
pid = int(pid_path.read_text()) if pid_path.exists() else None
alive = False
if pid:
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        pass
raw = root / 'data/raw/jaka_pilot/handover_block/aloha_agilex'
seeds = raw / 'seed.txt'
report = {'stage': status.read_text().strip() if status.exists() else 'not_started',
          'pid': pid, 'process_exists': alive,
          'planned_episodes': len(seeds.read_text().split()) if seeds.exists() else 0,
          'saved_episodes': len(list((raw / 'data').glob('*.hdf5'))),
          'disk_free_gib': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 2**30}
for name in ('collect-pilot', 'convert-pilot', 'train-pilot'):
    path = root / f'logs/{name}.log'
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 5000))
            lines = stream.read().decode(errors='replace').replace('\r', '\n').splitlines()
        report[f'{name}_last_lines'] = [line for line in lines if line][-3:]
print(json.dumps(report, indent=2, ensure_ascii=False))
