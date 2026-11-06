"""Copy a dataset and recompute low-dimensional stats in float64 with scale floors.

Small float32 cancellation errors and almost static joints must not amplify
physical tracking noise by orders of magnitude. Preserve all original episodes.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

PACK = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--joint-floor', type=float, default=.05)
    p.add_argument('--gripper-floor', type=float, default=.05)
    a = p.parse_args()
    if a.output.exists() or min(a.joint_floor, a.gripper_floor) <= 0:
        raise ValueError('Require new output and positive normalization floors')
    shutil.copytree(a.source, a.output)
    stats_file = a.output/'meta/stats.json'
    old = json.loads(stats_file.read_text())
    stats = json.loads(stats_file.read_text())
    report = {'source': str(a.source), 'raw': str(a.raw), 'output': str(a.output),
              'method': 'float64_two_pass_standard_deviation_with_physical_scale_floor_v1',
              'joint_floor_rad': a.joint_floor, 'gripper_floor_fraction': a.gripper_floor, 'features': {}}
    for key, raw_key in [('observation.state', 'state'), ('action', 'action')]:
        values = []
        for path in sorted(a.raw.glob('episode_*.hdf5')):
            with h5py.File(path) as f:
                values.append(np.concatenate([f[f'{raw_key}/{part}'][:] for part in PACK], axis=1).astype(np.float64))
        x = np.concatenate(values)
        assert x.ndim == 2 and x.shape[1] == 14 and np.isfinite(x).all()
        floors = np.full(14, a.joint_floor)
        floors[[6, 13]] = a.gripper_floor
        actual = x.std(axis=0, dtype=np.float64)
        mean = x.mean(axis=0, dtype=np.float64)
        effective = np.maximum(actual, floors)
        stats[key]['mean'] = mean.tolist()
        stats[key]['std'] = effective.tolist()
        stats[key]['min'] = x.min(axis=0).tolist()
        stats[key]['max'] = x.max(axis=0).tolist()
        stats[key]['count'] = [len(x)]
        z = (x-mean)/effective
        report['features'][key] = {'frames': len(x), 'old_mean': old[key]['mean'], 'old_std': old[key]['std'],
                                   'float64_std': actual.tolist(), 'effective_std': effective.tolist(),
                                   'floored_dimensions': np.flatnonzero(actual < floors).tolist(),
                                   'normalized_training_abs_max': np.abs(z).max(axis=0).tolist()}
    stats_file.write_text(json.dumps(stats, indent=2)+'\n')
    report['source_stats_sha256'] = hashlib.sha256((a.source/'meta/stats.json').read_bytes()).hexdigest()
    report['effective_stats_sha256'] = hashlib.sha256(stats_file.read_bytes()).hexdigest()
    (a.output/'meta/normalization_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    identity_file = a.output/'meta/project_identity.json'
    identity = json.loads(identity_file.read_text())
    identity['normalization'] = {k: report[k] for k in ['method', 'joint_floor_rad', 'gripper_floor_fraction', 'effective_stats_sha256']}
    identity_file.write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
