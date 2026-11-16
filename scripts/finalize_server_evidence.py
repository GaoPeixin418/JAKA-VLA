"""Finalize evidence after bounded evaluation ends; never launches new trials."""
from pathlib import Path
from datetime import datetime
import subprocess,time
r=Path('/data/jaka-vla')
run=r/'results/lab_pi05_benchmark_extended_20261116'
deadline=datetime.fromisoformat('2026-11-16T20:20:00+08:00').timestamp()
while not (run/'execution_finished.json').exists():
 if time.time()>=deadline:raise SystemExit('Evaluation did not finish; preserve partial evidence and inspect')
 time.sleep(15)
subprocess.run(['python3','scripts/verify_saved_benchmark.py','--run',str(run),'--output','results/benchmark_20261116_extended_manifest.json'],cwd=r,check=True)
subprocess.run(['python3','scripts/verify_benchmark_videos.py','--runs','results/lab_pi05_benchmark_20261116','results/lab_pi05_benchmark_extended_20261116','--output','results/benchmark_20261116_video_verification.json'],cwd=r,check=True)
print('FINAL_EVIDENCE_VERIFIED',flush=True)
