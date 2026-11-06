"""One bounded merged-data recovery curriculum, with explicit case provenance."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/data/jaka-vla')


def run(name, command, timeout=180):
    print('Starting', name, flush=True)
    with (ROOT/f'logs/{name}-zu7-recovery4000.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def main():
    rows = json.loads((ROOT/'results/zu7_recovery_batch_20261106.json').read_text())
    recovery = [(ROOT/'results/zu7_recovery_collect_seed0_step50', ROOT/'results/zu7_recovery_replay_seed0_step50')]
    recovery += [(Path(r['probe']), Path(r['replay'])) for r in rows if r['accepted']]
    if not 3 <= len(recovery) <= 6:
        raise ValueError('Require three to six accepted recovery starts')
    for probe, replay in recovery:
        p = json.loads((probe/'result.json').read_text())
        r = json.loads((replay/'results.json').read_text())[0]
        if not p['grasp_success'] or p['failure'] or not r['grasp_success'] or r['failure_reason']:
            raise ValueError('Recovery replay failed acceptance')
    probes = [ROOT/f'results/zu7_shortpath_collect_seed{s}' for s in (0,1,6)]
    replays = [ROOT/f'results/zu7_shortpath_replay_seed{s}' for s in (0,1,6)]
    probes += [ROOT/f'results/zu7_perturbed_collect_seed{s}' for s in (13,14,15,16,17)]
    replays += [ROOT/f'results/zu7_perturbed_replay_seed{s}' for s in (13,14,15,16,17)]
    probes += [p for p, _ in recovery]
    replays += [r for _, r in recovery]
    raw = ROOT/'data/raw/zu7_grasp_recovery/handover_block/jaka_zu7_proxy_jaka_zu7_proxy'
    converted = ROOT/'data/lerobot/zu7_grasp_recovery_rawstats'
    stable = ROOT/'data/lerobot/zu7_grasp_recovery_stable'
    config = ROOT/'configs/zu7_recovery4000.json'
    sim = str(ROOT/'envs/robotwin/bin/python')
    vla = str(ROOT/'envs/vla/bin/python')
    run('assemble', [sim, 'scripts/assemble_grasp_curriculum.py', '--probes', *map(str, probes),
                    '--replays', *map(str, replays), '--output', str(raw)])
    run('convert', [vla, 'scripts/convert_dataset_v2.py', '--source', str(raw/'data'), '--output', str(converted),
                    '--repo-id', 'jaka-local/zu7-grasp-recovery'])
    run('stabilize', [sim, 'scripts/stabilize_zu7_statistics.py', '--source', str(converted),
                      '--raw', str(raw/'data'), '--output', str(stable)])
    run('prepare', [vla, 'scripts/prepare_training_v2.py', '--dataset', str(stable),
                    '--repo-id', 'jaka-local/zu7-grasp-recovery', '--robot-id', 'dual_jaka_zu7',
                    '--steps', '4000', '--batch-size', '8',
                    '--output', str(ROOT/'checkpoints/smolvla_zu7_recovery4000_seed42'), '--config', str(config)])
    cfg = json.loads(config.read_text())
    cfg['save_freq'] = 4000
    config.write_text(json.dumps(cfg, indent=2)+'\n')
    run('train', [vla, 'scripts/run_training_v2.py', '--config', str(config)], timeout=1000)
    print('Recovery4000 complete; accepted recovery starts', len(recovery), flush=True)


if __name__ == '__main__':
    main()
