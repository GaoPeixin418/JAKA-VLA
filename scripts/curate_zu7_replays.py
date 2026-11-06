"""Select task-valid expert replays without modifying or relabeling source data."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Raw episode root containing seed.txt')
    parser.add_argument('--replays', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    seeds = [int(x) for x in (source / 'seed.txt').read_text().split()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Empty or duplicated source seeds')
    files = sorted((source / 'data').glob('episode_*.hdf5'))
    if len(files) != len(seeds):
        raise ValueError('Seed and HDF5 counts differ')
    identity = json.loads((source / 'robot_identity.json').read_text())
    if identity['robot_id'] != 'dual_jaka_zu7':
        raise ValueError('Source is not a Zu7 dataset')
    rows = []
    for index, (seed, path) in enumerate(zip(seeds, files)):
        if path.name != f'episode_{index:07d}.hdf5':
            raise ValueError('Non-contiguous source episode numbering')
        report = args.replays / f'seed_{seed}' / 'results.json'
        values = json.loads(report.read_text())
        if len(values) != 1:
            raise ValueError(f'Expected one replay for seed {seed}')
        row = values[0]
        if (row['seed'], row['robot_id'], row['rig_urdf_sha256'], row['mode']) != (
                seed, identity['robot_id'], identity['urdf_sha256'], 'expert_replay'):
            raise ValueError(f'Replay identity mismatch for seed {seed}')
        accepted = bool(row['success']) and row.get('failure_reason') is None and row['hard_limit_events'] == 0
        rows.append(dict(seed=seed, source_episode=str(path), source_sha256=sha256(path),
                         replay_report=str(report.resolve()), replay_sha256=sha256(report),
                         accepted=accepted, failure_reason=row.get('failure_reason')))
    accepted = [row for row in rows if row['accepted']]
    if not accepted:
        raise ValueError('No task-valid replays; do not train')
    (output / 'data').mkdir(parents=True)
    for index, row in enumerate(accepted):
        target = output / 'data' / f'episode_{index:07d}.hdf5'
        target.symlink_to(os.path.relpath(row['source_episode'], target.parent))
        row['output_episode'] = target.name
    (output / 'seed.txt').write_text(' '.join(str(row['seed']) for row in accepted) + '\n')
    (output / 'robot_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    manifest = dict(source=str(source), replays=str(args.replays.resolve()), episodes=rows,
                    accepted_count=len(accepted), rejected_count=len(rows)-len(accepted),
                    criterion='Task success, no failure_reason, zero hard-limit events under current replay controller.',
                    limitation='Acceptance verifies task replay only; contacts and measured-speed peaks remain separate diagnostics, not a hardware safety certificate.')
    (output / 'curation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
