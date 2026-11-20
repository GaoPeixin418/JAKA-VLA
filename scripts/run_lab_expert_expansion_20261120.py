"""Collect a bounded set of diverse Zu7 lab grasp demonstrations with replay gates."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/jaka-vla')
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='lab_expert_expansion_20261120')
    parser.add_argument('--layouts', type=int, nargs='+', default=[4010, 4011, 4012])
    parser.add_argument('--variants', type=int, default=2)
    parser.add_argument('--deadline', default='2026-11-20T13:10:00+08:00')
    args = parser.parse_args()
    if not args.name.startswith('lab_expert_') or not all(c.isalnum() or c == '_' for c in args.name):
        raise ValueError('Invalid run name')
    if not args.layouts or len(set(args.layouts)) != len(args.layouts) or args.variants != 2:
        raise ValueError('Use unique layouts and exactly two documented path variants')

    run = ROOT / 'results' / args.name
    run.mkdir(parents=True, exist_ok=False)
    for directory in ('scenes', 'logs'):
        (run / directory).mkdir()
    report = {
        'completed': False,
        'scope': 'new multi-layout scripted expert paths; not policy-prefix DAgger',
        'design': {'layouts': args.layouts, 'targets_per_layout': [0, 1, 2],
                   'variants': {'0': 'nominal robot start', '1': 'seeded left-joint offset up to 0.035 rad'},
                   'expected_attempts': len(args.layouts) * 3 * args.variants,
                   'split': 'train; layouts 4010+ reserved for expert training'},
        'deadline': args.deadline, 'attempts': [], 'accepted': 0,
    }
    deadline = dt.datetime.fromisoformat(args.deadline).timestamp()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
    py = ROOT / 'envs/robotwin/bin/python'
    from jaka_vla.lab_scene import scene_spec
    from jaka_vla.contact_acceptance import lab_forbidden_contact

    for layout in args.layouts:
        for target in range(3):
            spec = scene_spec(layout, target, 'train')
            for variant in range(args.variants):
                if time.time() >= deadline - 240:
                    report['stop_reason'] = 'collection_cutoff_for_training_and_evaluation'
                    break
                seed = 100000 + layout * 10 + target * 2 + variant
                key = f'layout_{layout:04d}_target_{target}_variant_{variant}'
                scene = run / 'scenes' / f'{key}.json'
                scene.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
                probe = run / f'{key}_probe'
                replay = run / f'{key}_replay'
                row = {'layout': layout, 'target': target, 'color': spec['objects'][target]['color'],
                       'variant': variant, 'seed': seed, 'initial_offset_rad': 0.0 if variant == 0 else 0.035,
                       'scene': str(scene), 'probe': str(probe), 'replay': str(replay), 'accepted': False}
                report['attempts'].append(row)
                atomic_json(run / 'progress.json', report)
                try:
                    probe_cmd = [str(py), '-u', 'scripts/probe_zu7_short_grasp.py', '--rig-config',
                                 'configs/robots/dual_zu7_lab.json', '--scene-spec', str(scene), '--seed', str(seed),
                                 '--output', str(probe), '--collect', '--initial-offset', str(row['initial_offset_rad'])]
                    if variant:
                        probe_cmd.append('--no-review-video')
                    with (run / 'logs' / f'{key}_probe.log').open('w') as log:
                        subprocess.run(probe_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=min(600, max(1, deadline - time.time())), check=True)
                    result = json.loads((probe / 'result.json').read_text())
                    row['expert_success'] = bool(result['grasp_success'] and result['failure'] is None)
                    probe_contacts = json.loads((probe / 'contacts.json').read_text())
                    bad_probe = [c['pair'] for c in probe_contacts if lab_forbidden_contact(c['pair'])]
                    row['probe_forbidden_contacts'] = bad_probe
                    if not row['expert_success'] or bad_probe:
                        row['failure'] = result.get('failure') or 'probe_failed_acceptance'
                        continue
                    replay_cmd = [str(py), '-u', 'scripts/eval_v2.py', '--robot-id', 'dual_jaka_zu7_lab_archive',
                                  '--rig-config', 'configs/robots/dual_zu7_lab.json', '--scene-spec', str(scene),
                                  '--seed-start', str(seed), '--episodes', '1', '--expert-hdf5', str(probe / 'raw/data'),
                                  '--output', str(replay), '--instruction', spec['instruction'],
                                  '--task-scope', 'left_grasp_lift_hold']
                    if variant == 0:
                        replay_cmd.append('--review-video')
                    with (run / 'logs' / f'{key}_replay.log').open('w') as log:
                        subprocess.run(replay_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=min(360, max(1, deadline - time.time())), check=True)
                    replay_result = json.loads((replay / 'results.json').read_text())[0]
                    contacts = json.loads((replay / f'seed_{seed}_contacts.json').read_text())
                    bad_replay = [c['pair'] for c in contacts if lab_forbidden_contact(c['pair'])]
                    interarm = [c['pair'] for c in contacts if any(s.startswith('left:') for s in c['pair'])
                                and any(s.startswith('right:') for s in c['pair'])]
                    failure = replay_result.get('failure_reason')
                    safe_failure = not failure or (isinstance(failure, dict) and
                                                    failure.get('type') == 'step_budget_exhausted')
                    row.update(replay_success=bool(replay_result.get('grasp_success')),
                               replay_failure=failure, replay_forbidden_contacts=bad_replay,
                               replay_interarm_contacts=interarm)
                    row['accepted'] = bool(row['replay_success'] and safe_failure and not bad_replay and not interarm)
                    if not row['accepted']:
                        row['failure'] = 'same-scene-replay-failed-acceptance'
                except Exception as exc:
                    row['failure'] = repr(exc)
                finally:
                    report['accepted'] = sum(bool(x.get('accepted')) for x in report['attempts'])
                    atomic_json(run / 'progress.json', report)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
            if report.get('stop_reason'):
                break
        if report.get('stop_reason'):
            break

    accepted = [r for r in report['attempts'] if r.get('accepted')]
    report['attempt_count'] = len(report['attempts'])
    if accepted:
        probes = [r['probe'] for r in accepted]
        replays = [r['replay'] for r in accepted]
        dataset = ROOT / 'data/raw' / args.name
        with (run / 'logs/assemble.log').open('w') as log:
            subprocess.run([str(py), 'scripts/assemble_grasp_curriculum.py', '--probes', *probes,
                            '--replays', *replays, '--output', str(dataset)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        report['raw_dataset'] = str(dataset)
    report['completed'] = True
    report.setdefault('stop_reason', 'attempt_budget_complete')
    atomic_json(run / 'progress.json', report)
    print(json.dumps({'attempts': report['attempt_count'], 'accepted': report['accepted'],
                      'raw_dataset': report.get('raw_dataset'), 'stop_reason': report['stop_reason']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
