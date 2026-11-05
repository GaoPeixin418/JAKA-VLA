"""Run a bounded set of synchronous fixed-K pilot evaluations."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/data/jaka-vla')
checkpoint = ROOT / 'checkpoints/smolvla_pilot_seed42/checkpoints/last/pretrained_model'
if not (checkpoint / 'model.safetensors').exists():
    raise FileNotFoundError(checkpoint)
socket_path = ROOT / 'run/policy.sock'
summary = []
for k in (4, 8, 16):
    if socket_path.exists():
        raise RuntimeError('Policy socket already exists; check its owner before restarting')
    with (ROOT / f'logs/policy-server-k{k}.log').open('w') as server_log:
        server = subprocess.Popen([str(ROOT / 'envs/vla/bin/python'), '-u',
            str(ROOT / 'scripts/policy_server.py'), '--checkpoint', str(checkpoint)],
            stdout=server_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while not socket_path.exists():
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server exited: {server.returncode}')
                if time.monotonic() > deadline:
                    raise TimeoutError('Policy server did not become ready')
                time.sleep(1)
            output = ROOT / f'results/pilot_k{k}'
            with (ROOT / f'logs/eval-k{k}.log').open('w') as log:
                subprocess.run([str(ROOT / 'envs/robotwin/bin/python'), '-u',
                    str(ROOT / 'scripts/eval_fixed.py'), '--k', str(k), '--seed-start', '1000',
                    '--episodes', '3', '--output', str(output)], check=True,
                    stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            episodes = json.loads((output / 'results.json').read_text())
            summary.append({'k': k, 'episodes': len(episodes),
                            'successes': sum(ep['success'] for ep in episodes),
                            'success_rate': sum(ep['success'] for ep in episodes) / len(episodes),
                            'results': str(output / 'results.json'),
                            'mode': 'pilot_only_synchronous_zero_simulated_latency'})
            (ROOT / 'results/pilot_baselines.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary[-1]), flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            if socket_path.exists():
                socket_path.unlink()
