"""Audit all collected expert RGB/state/action frames and same-scene acceptance evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path('/data/jaka-vla')
PACK = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']
CAMERAS = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, nargs='+', required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    progress = [json.loads((run / 'progress.json').read_text()) for run in a.run]
    if any(not item.get('completed') for item in progress):
        raise ValueError('Collection batch is incomplete')
    attempts = [row for item in progress for row in item['attempts']]
    accepted = [row for row in attempts if row.get('accepted')]
    manifest = json.loads((a.raw / 'acceptance.json').read_text())
    if len(accepted) != len(manifest):
        raise ValueError('Accepted progress count and assembled manifest differ')

    rows = []
    source_hashes = set()
    total_frames = 0
    total_rgb = 0
    rgb_pass = 0
    timestamp_pass = 0
    for row, entry in zip(accepted, manifest):
        probe = Path(row['probe'])
        hdf = probe / 'raw/data/episode_0000000.hdf5'
        dataset_hdf = a.raw / f"data/episode_{entry['episode']:07d}.hdf5"
        digest = sha256(hdf)
        if digest in source_hashes:
            raise ValueError(f'Byte-identical duplicate expert data: {hdf}')
        source_hashes.add(digest)
        if digest != entry['sha256'] or digest != sha256(dataset_hdf):
            raise ValueError(f'Source/assembled HDF5 mismatch: episode {entry["episode"]}')
        with h5py.File(hdf, 'r') as f:
            n = len(f['action/left_arm_joint_states'])
            if n < 2:
                raise ValueError(f'Empty episode: {hdf}')
            state = np.concatenate([f[f'state/{key}'][:] for key in PACK], axis=1)
            action = np.concatenate([f[f'action/{key}'][:] for key in PACK], axis=1)
            command = np.concatenate([f[f'commanded_state/{key}'][:] for key in PACK], axis=1)
            if state.shape != (n, 14) or action.shape != (n, 14) or command.shape != (n, 14):
                raise ValueError(f'State/action schema mismatch: {hdf}')
            if not all(np.isfinite(x).all() for x in (state, action, command)):
                raise ValueError(f'Nonfinite state/action: {hdf}')
            obs_t = f['observation_timestamps'][:]
            act_t = f['action_timestamps'][:]
            if len(obs_t) != n or len(act_t) != n:
                raise ValueError(f'Timestamp count mismatch: {hdf}')
            if not np.allclose(np.diff(obs_t), 0.04, atol=1e-5, rtol=0):
                raise ValueError(f'Observation timestamps are not 25Hz: {hdf}')
            if not np.allclose(act_t - obs_t, 0.04, atol=1e-5, rtol=0):
                raise ValueError(f'Action timestamps are not one observation ahead: {hdf}')
            timestamp_pass += 1
            camera_counts = {}
            for camera in CAMERAS:
                encoded = f[f'vision/{camera}/colors']
                if len(encoded) != n:
                    raise ValueError(f'{camera} frame count mismatch: {hdf}')
                declared = tuple(int(x) for x in f[f'vision/{camera}/shape'][:])
                if declared != (240, 320, 3):
                    raise ValueError(f'{camera} declared shape {declared}: {hdf}')
                for frame_index in range(n):
                    image = cv2.imdecode(np.frombuffer(encoded[frame_index], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape != (240, 320, 3):
                        raise ValueError(f'{camera} RGB decode failed at frame {frame_index}: {hdf}')
                camera_counts[camera] = n
            rgb_pass += 1
            rgb_count = sum(camera_counts.values())
        total_frames += n
        total_rgb += rgb_count
        rows.append({'episode': entry['episode'], 'layout': row['layout'], 'target': row['target'],
                     'variant': row['variant'], 'seed': row['seed'], 'frames': n, 'rgb_frames': rgb_count,
                     'timestamp_25hz_pass': True, 'all_three_rgb_streams_decode': True,
                     'source_assembled_sha256': digest})
    report = {'attempted': len(attempts),
              'expert_grasp_pass': sum(bool(x.get('expert_success')) for x in attempts),
              'complete_timestamp_audit_pass': timestamp_pass,
              'complete_rgb_decode_pass': rgb_pass,
              'same_scene_replay_pass': sum(bool(x.get('replay_success')) and not x.get('replay_forbidden_contacts') and not x.get('replay_interarm_contacts') for x in attempts),
              'accepted_for_training': len(accepted), 'assembled_episodes': len(manifest),
              'frames': total_frames, 'decoded_rgb_frames': total_rgb,
              'unique_trajectory_sha256': len(source_hashes), 'episode_checks': rows,
              'scope': 'Full accepted RGB/state/action/timestamp and replay audit; no learned-policy result.'}
    if not (timestamp_pass == rgb_pass == len(accepted) == len(manifest)):
        raise AssertionError('Not every accepted episode passed full input audit')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'episode_checks'}))


if __name__ == '__main__':
    main()
