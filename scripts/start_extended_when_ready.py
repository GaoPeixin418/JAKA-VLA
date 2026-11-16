"""Run the predeclared extension only after the original benchmark has finished."""
from pathlib import Path
from datetime import datetime
import json,os,subprocess,time
root=Path('/data/jaka-vla')
first=root/'results/lab_pi05_benchmark_20261116'
cutoff=datetime.fromisoformat('2026-11-16T20:10:00+08:00').timestamp()
while not (first/'execution_finished.json').exists():
    if time.time()>=cutoff:raise SystemExit('Deadline reached before extension')
    time.sleep(10)
workers=[json.loads(p.read_text()) for p in first.glob('worker_*.json')]
if len(workers)!=2 or any(w.get('fatal_error') for w in workers):
    raise SystemExit('Primary benchmark requires intervention; not starting extension')
if time.time()>=cutoff:raise SystemExit('Deadline reached')
os.execv(str(root/'envs/robotwin/bin/python'),[str(root/'envs/robotwin/bin/python'),'-u',str(root/'scripts/evaluate_lab_extended.py'),'--checkpoint',str(root/'checkpoints/openpi/pi05_zu7_lora_h50/lab_pi05_three_targets_20261112/5744'),'--name','lab_pi05_benchmark_extended_20261116','--no-new-after','2026-11-16T20:10:00+08:00','--hard-stop','2026-11-16T20:18:00+08:00','--execute-prepared'])
