"""Bounded sequential expert collection and same-condition servo acceptance."""
import argparse
from datetime import datetime
import json
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/data/jaka-vla')


def accepted(probe, replay):
    if not (probe/'result.json').exists() or not (replay/'results.json').exists():
        return False
    p = json.loads((probe/'result.json').read_text())
    r = json.loads((replay/'results.json').read_text())[0]
    if p['seed'] != r['seed'] or not p['grasp_success'] or p['failure'] or not r['grasp_success'] or r['failure_reason']:
        return False
    pc = p['initial_condition']['left_joint_offset_rad']
    rc = r['initial_condition']['left_joint_offset_rad']
    if pc != rc:
        raise ValueError('Expert and replay initial conditions differ')
    contacts = json.loads((replay/f'seed_{p["seed"]}_contacts.json').read_text())
    return not any(any(s.startswith('left:') for s in c['pair']) and any(s.startswith('right:') for s in c['pair']) for c in contacts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first-seed', type=int, default=14)
    p.add_argument('--max-attempts', type=int, default=7)
    p.add_argument('--additional-accepted', type=int, default=5)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = []
    for seed in range(a.first_seed, a.first_seed+a.max_attempts):
        now = datetime.now(ZoneInfo('Asia/Shanghai'))
        if (now.hour, now.minute) >= (17, 5):
            break
        probe = ROOT/f'results/zu7_perturbed_collect_seed{seed}'
        replay = ROOT/f'results/zu7_perturbed_replay_seed{seed}'
        if probe.exists() or replay.exists():
            raise FileExistsError('This batch does not overwrite or silently reuse an earlier seed')
        commands = [
            [str(ROOT/'envs/robotwin/bin/python'), str(ROOT/'scripts/probe_zu7_short_grasp.py'),
             '--seed', str(seed), '--initial-offset', '.04', '--collect', '--output', str(probe)],
            [str(ROOT/'envs/robotwin/bin/python'), str(ROOT/'scripts/eval_v2.py'),
             '--expert-hdf5', str(probe/'raw/data'), '--seed-start', str(seed), '--episodes', '1',
             '--review-video', '--task-scope', 'left_grasp_lift_hold',
             '--instruction', 'Use the left arm to grasp the red block, lift it, and hold it.', '--output', str(replay)]]
        error = None
        for kind, cmd in zip(('collect', 'replay'), commands):
            try:
                with (ROOT/f'logs/zu7-perturbed-{kind}-seed{seed}.log').open('x') as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
                if kind == 'collect':
                    r = json.loads((probe/'result.json').read_text())
                    if not r['grasp_success'] or r['failure']:
                        error = 'expert_probe_rejected'
                        break
            except (subprocess.SubprocessError, OSError) as exc:
                error = f'{type(exc).__name__}: {exc}'
                break
        ok = error is None and accepted(probe, replay)
        rows.append({'seed': seed, 'accepted': ok, 'probe': str(probe), 'replay': str(replay), 'error': error})
        a.output.write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(rows[-1]), flush=True)
        if sum(r['accepted'] for r in rows) >= a.additional_accepted:
            break


if __name__ == '__main__':
    main()
