"""Bounded recovery demonstrations from explicitly intervened learner states."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/data/jaka-vla')
CASES = [
    (6, 'zu7_coverage3000_seen_seed6', 35),
    (101, 'zu7_coverage3000_new_seed100_102', 45),
    (0, 'zu7_stable1500_seed0_1', 40),
    (6, 'zu7_coverage3000_seen_seed6', 25),
    (101, 'zu7_coverage3000_new_seed100_102', 35),
]


def main():
    output = ROOT/'results/zu7_recovery_batch_20261106.json'
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for seed, folder, step in CASES:
        probe = ROOT/f'results/zu7_recovery_collect_seed{seed}_step{step}'
        replay = ROOT/f'results/zu7_recovery_replay_seed{seed}_step{step}'
        if probe.exists() or replay.exists():
            raise FileExistsError('Refuse to repeat a recovery case')
        trace = ROOT/f'results/{folder}/seed_{seed}_trace.npz'
        commands = [
            [str(ROOT/'envs/robotwin/bin/python'), str(ROOT/'scripts/probe_zu7_short_grasp.py'),
             '--seed', str(seed), '--start-trace', str(trace), '--start-step', str(step), '--collect', '--output', str(probe)],
            [str(ROOT/'envs/robotwin/bin/python'), str(ROOT/'scripts/eval_v2.py'),
             '--expert-hdf5', str(probe/'raw/data'), '--seed-start', str(seed), '--episodes', '1',
             '--review-video', '--task-scope', 'left_grasp_lift_hold',
             '--instruction', 'Use the left arm to grasp the red block, lift it, and hold it.', '--output', str(replay)]]
        error = None
        for kind, cmd in zip(('collect', 'replay'), commands):
            try:
                with (ROOT/f'logs/zu7-recovery-{kind}-seed{seed}-step{step}.log').open('x') as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
                if kind == 'collect':
                    report = json.loads((probe/'result.json').read_text())
                    if not report['grasp_success'] or report['failure']:
                        error = 'expert_recovery_rejected'
                        break
            except (subprocess.SubprocessError, OSError) as exc:
                error = f'{type(exc).__name__}: {exc}'
                break
        accepted = False
        if error is None:
            r = json.loads((replay/'results.json').read_text())[0]
            accepted = bool(r['grasp_success'] and r['failure_reason'] is None)
        row = {'seed': seed, 'source_step': step, 'probe': str(probe), 'replay': str(replay),
               'accepted': accepted, 'error': error, 'scope': 'intervened_recovery_start'}
        rows.append(row)
        output.write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
