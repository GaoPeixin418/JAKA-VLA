"""Combine only accepted, compatible Zu7 grasp demonstrations into one raw dataset."""
import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/data/jaka-vla')
PATTERN = re.compile(r'^episode_(\d+)\.hdf5$')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_source(root):
    identity_path = root / 'robot_identity.json'
    identity = json.loads(identity_path.read_text())
    if identity.get('task_scope') != 'left_grasp_lift_hold':
        raise ValueError(f'Unexpected task scope in {root}')
    paths = sorted((root / 'data').glob('episode_*.hdf5'), key=lambda p: int(PATTERN.match(p.name).group(1)))
    if not paths:
        raise ValueError(f'No episodes in {root}')
    indexes = [int(PATTERN.match(p.name).group(1)) for p in paths]
    if indexes != list(range(len(paths))):
        raise ValueError(f'Nonconsecutive episode IDs in {root}')
    scenes = identity.get('scenes_by_episode', {})
    conditions = identity.get('initial_conditions_by_episode', {})
    if set(scenes) != {str(i) for i in indexes} or set(conditions) != {str(i) for i in indexes}:
        raise ValueError(f'Per-episode scene/initial-condition identity is incomplete in {root}')
    records = []
    for i, path in enumerate(paths):
        scene = scenes[str(i)]
        if scene.get('split') != 'train':
            raise ValueError(f'Nontraining scene in source: {root} episode {i}')
        if not 0 <= int(scene.get('target_index', -1)) < 3:
            raise ValueError(f'Invalid target identity in {root} episode {i}')
        with h5py.File(path, 'r') as f:
            if f.attrs.get('robot_id') != identity.get('robot_id') or f.attrs.get('rig_urdf_sha256') != identity.get('urdf_sha256'):
                raise ValueError(f'Robot/URDF identity mismatch in {path}')
            if f.attrs.get('state_semantics') != 'measured_qpos' or f.attrs.get('action_semantics') != 'next_sample_drive_targets':
                raise ValueError(f'Unexpected state/action semantics in {path}')
            obs = np.asarray(f['observation_timestamps'], dtype=np.float64)
            act = np.asarray(f['action_timestamps'], dtype=np.float64)
            if len(obs) < 2 or len(act) != len(obs) or not np.isfinite(obs).all() or not np.isfinite(act).all():
                raise ValueError(f'Invalid timestamps in {path}')
            if not np.allclose(np.diff(obs), .04, atol=1e-6) or not np.allclose(act - obs, .04, atol=1e-6):
                raise ValueError(f'Not a 25 Hz next-sample trajectory in {path}')
            for side in ('left', 'right'):
                s = np.asarray(f[f'state/{side}_arm_joint_states'])
                a = np.asarray(f[f'action/{side}_arm_joint_states'])
                sg = np.asarray(f[f'state/{side}_ee_joint_states'])
                ag = np.asarray(f[f'action/{side}_ee_joint_states'])
                if s.shape != (len(obs), 6) or a.shape != s.shape or sg.shape != (len(obs), 1) or ag.shape != sg.shape:
                    raise ValueError(f'Invalid 14D joint/aperture schema in {path}')
                if not np.isfinite(s).all() or not np.isfinite(a).all() or not np.isfinite(sg).all() or not np.isfinite(ag).all():
                    raise ValueError(f'Nonfinite state/action in {path}')
            for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                colors = f[f'vision/{camera}/colors']
                if len(colors) != len(obs):
                    raise ValueError(f'RGB frame count mismatch in {path} {camera}')
            if int(f['additional_info/frequency'][()]) != 25:
                raise ValueError(f'Unexpected collection frequency in {path}')
        records.append({'source_episode': i, 'frames': len(obs), 'layout_id': int(scene['layout_id']),
                        'target_index': int(scene['target_index']), 'sha256': sha256(path)})
    return identity, paths, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expansion-evidence', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    evidence = json.loads(a.expansion_evidence.read_text())
    if not evidence.get('completed') or evidence.get('accepted') != 29 or evidence.get('data_frames') != 12559:
        raise ValueError('Expansion acceptance evidence is incomplete or unexpected')
    expansion_root = Path(evidence['raw_dataset']).resolve()
    if expansion_root not in [s.resolve() for s in a.sources]:
        raise ValueError('The fully accepted expansion dataset must be among the sources')
    loaded = [(root.resolve(), *load_source(root.resolve())) for root in a.sources]
    base_identity = loaded[0][1]
    identity_keys = ('robot_id', 'urdf_sha256', 'rig', 'task_scope')
    for root, ident, _, _ in loaded[1:]:
        if any(ident.get(k) != base_identity.get(k) for k in identity_keys):
            raise ValueError(f'Incompatible hardware/task identity in {root}')
    temp = a.output.with_name(a.output.name + '.tmp')
    if temp.exists():
        raise FileExistsError(temp)
    (temp / 'data').mkdir(parents=True)
    merged_identity = dict(base_identity)
    merged_identity['scenes_by_episode'] = {}
    merged_identity['initial_conditions_by_episode'] = {}
    merged_identity['initial_conditions_by_seed'] = {}
    source_records = []
    seen = set()
    all_records = []
    offset = 0
    try:
        for source_index, (root, ident, paths, records) in enumerate(loaded):
            begin = offset
            for rec, path in zip(records, paths):
                src_ep = rec['source_episode']
                scene = ident['scenes_by_episode'][str(src_ep)]
                key = (scene.get('version'), rec['layout_id'], rec['target_index'])
                if key in seen:
                    raise ValueError(f'Duplicate layout/target condition: {key}')
                seen.add(key)
                dst = temp / 'data' / f'episode_{offset:07d}.hdf5'
                shutil.copy2(path, dst)
                merged_identity['scenes_by_episode'][str(offset)] = scene
                merged_identity['initial_conditions_by_episode'][str(offset)] = ident['initial_conditions_by_episode'][str(src_ep)]
                all_records.append({'episode_index': offset, 'source': str(root), **rec})
                offset += 1
            seed_rows = ident.get('initial_conditions_by_seed', {})
            for seed, value in seed_rows.items():
                merged_identity['initial_conditions_by_seed'][f'{root.name}:{seed}'] = value
            source_records.append({'path': str(root), 'source_index': source_index, 'episode_start': begin,
                                   'episode_end_exclusive': offset, 'episodes': len(paths),
                                   'identity_sha256': sha256(root / 'robot_identity.json'),
                                   'split_assignment_provenance': ident.get('split_assignment_provenance')})
        merged_identity['dataset_sources'] = source_records
        merged_identity['merged_training_dataset'] = True
        (temp / 'robot_identity.json').write_text(json.dumps(merged_identity, indent=2) + '\n')
        report = {'completed': True, 'episodes': offset, 'frames': sum(x['frames'] for x in all_records),
                  'distinct_layouts': len({x['layout_id'] for x in all_records}),
                  'layout_counts': dict(Counter(x['layout_id'] for x in all_records)),
                  'target_counts': dict(Counter(x['target_index'] for x in all_records)),
                  'sources': source_records, 'expansion_acceptance_evidence': str(a.expansion_evidence.resolve()),
                  'expansion_evidence_sha256': sha256(a.expansion_evidence), 'episodes_manifest': all_records,
                  'split': 'all train; no development or blind-test episodes',
                  'image_decoding': 'delegated to the subsequent full LeRobot converter'}
        (temp / 'merge_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        os.replace(temp, a.output)
        print(json.dumps({k: v for k, v in report.items() if k != 'episodes_manifest'}, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()
