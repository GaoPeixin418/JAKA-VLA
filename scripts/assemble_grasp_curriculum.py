"""Accept only identified expert grasps verified under the learned-policy servo."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probes', type=Path, nargs='+', required=True)
    p.add_argument('--replays', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if len(args.probes) != len(args.replays):
        raise ValueError('Each probe needs its own same-seed replay')
    accepted = []
    identity = None
    for probe, replay in zip(args.probes, args.replays):
        report = json.loads((probe/'result.json').read_text())
        review = json.loads((replay/'results.json').read_text())
        if len(review) != 1 or review[0]['seed'] != report['seed']:
            raise ValueError('Replay seed mismatch')
        replay_report = review[0]
        source_identity = json.loads((probe/'raw/robot_identity.json').read_text())
        contacts = json.loads((replay/f'seed_{report["seed"]}_contacts.json').read_text())
        interarm = [c for c in contacts if any(x.startswith('left:') for x in c['pair']) and any(x.startswith('right:') for x in c['pair'])]
        if not report['grasp_success'] or report['failure'] is not None or not replay_report['grasp_success'] or interarm:
            raise ValueError(f'Grasp acceptance failed: {probe}')
        if replay_report['failure_reason'] and replay_report['failure_reason']['type'] != 'step_budget_exhausted':
            raise ValueError(f'Controller protection fired: {replay}')
        current = {k: source_identity[k] for k in ('robot_id', 'rig', 'urdf_sha256', 'task_scope')}
        if identity is not None and identity != current:
            raise ValueError('Mixed robot/task identities')
        identity = current
        accepted.append((probe, replay, report['seed']))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'data').mkdir()
    manifest = []
    for index, (probe, replay, seed) in enumerate(accepted):
        source = probe/'raw/data/episode_0000000.hdf5'
        dest = args.output/f'data/episode_{index:07d}.hdf5'
        shutil.copy2(source, dest)
        manifest.append({'episode': index, 'seed': seed, 'source': str(source), 'replay': str(replay),
                         'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    (args.output/'seed.txt').write_text(' '.join(str(s) for _, _, s in accepted)+'\n')
    (args.output/'robot_identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    (args.output/'acceptance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
