"""Train once on the explicitly accepted, merged grasp coverage batch."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/data/jaka-vla')


def main():
    rows = json.loads((ROOT/'results/zu7_perturbed_batch_20261106.json').read_text())
    seeds = [13]+[r['seed'] for r in rows if r['accepted']]
    if not 3 <= len(seeds) <= 6:
        raise ValueError('Require three to six accepted new scenes')
    probes = [ROOT/f'results/zu7_shortpath_collect_seed{s}' for s in (0, 1, 6)]
    replays = [ROOT/f'results/zu7_shortpath_replay_seed{s}' for s in (0, 1, 6)]
    probes += [ROOT/f'results/zu7_perturbed_collect_seed{s}' for s in seeds]
    replays += [ROOT/f'results/zu7_perturbed_replay_seed{s}' for s in seeds]
    raw = ROOT/'data/raw/zu7_grasp_coverage/handover_block/jaka_zu7_proxy_jaka_zu7_proxy'
    converted = ROOT/'data/lerobot/zu7_grasp_coverage_rawstats'
    stable = ROOT/'data/lerobot/zu7_grasp_coverage_stable'
    robotwin = str(ROOT/'envs/robotwin/bin/python')
    vla = str(ROOT/'envs/vla/bin/python')
    commands = [
        [robotwin, 'scripts/assemble_grasp_curriculum.py', '--probes', *map(str, probes),
         '--replays', *map(str, replays), '--output', str(raw)],
        [vla, 'scripts/convert_dataset_v2.py', '--source', str(raw/'data'), '--output', str(converted),
         '--repo-id', 'jaka-local/zu7-grasp-coverage'],
        [robotwin, 'scripts/stabilize_zu7_statistics.py', '--source', str(converted), '--raw', str(raw/'data'), '--output', str(stable)],
        [vla, 'scripts/prepare_training_v2.py', '--dataset', str(stable), '--repo-id', 'jaka-local/zu7-grasp-coverage',
         '--robot-id', 'dual_jaka_zu7', '--steps', '3000', '--batch-size', '8',
         '--output', str(ROOT/'checkpoints/smolvla_zu7_coverage3000_seed42'),
         '--config', str(ROOT/'configs/zu7_coverage3000.json')],
        [vla, 'scripts/run_training_v2.py', '--config', str(ROOT/'configs/zu7_coverage3000.json')]]
    for name, command in zip(('assemble', 'convert', 'stabilize', 'prepare', 'train'), commands):
        print('Starting', name, flush=True)
        with (ROOT/f'logs/{name}-zu7-coverage3000.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=720 if name == 'train' else 180)
    print('Complete coverage training; accepted new seeds', seeds, flush=True)


if __name__ == '__main__':
    main()
