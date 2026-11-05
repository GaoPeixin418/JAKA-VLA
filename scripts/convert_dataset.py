"""Convert timestamp-validated RoboTwin RGB/joint demonstrations to LeRobot v3."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
from PIL import Image

PROJECT = Path('/data/jaka-vla')
sys.path.insert(0, str(PROJECT / 'repos/RoboTwin'))
from XPolicyLab.utils.data_loader import load_xspark_v1
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo-id', default='jaka-local/handover-smoke')
    p.add_argument('--fps', type=int, default=25)
    args = p.parse_args()
    paths = sorted(args.source.glob('episode_*.hdf5'))
    if not paths:
        raise FileNotFoundError(args.source)
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace dataset: {args.output}')
    dims = get_robot_action_dim_info('aloha_agilex')
    camera_map = {'cam_head': 'camera1', 'cam_left_wrist': 'camera2', 'cam_right_wrist': 'camera3'}
    dataset = None
    reports = []
    for ep, path in enumerate(paths):
        with h5py.File(path, 'r') as hf:
            state_semantics = hf.attrs.get('state_semantics', 'command_targets_legacy')
        if state_semantics != 'measured_qpos':
            raise ValueError(f'Recollect legacy command-target state data before training: {path}')
        raw = load_xspark_v1(str(path), decode_images=True)
        state = pack_robot_state(raw, 'joint', dims, source_type='dataset').astype(np.float32)
        action = pack_robot_state(raw, 'joint', dims, source_type='dataset', state_type='action').astype(np.float32)
        times = np.asarray(raw['observation_timestamps'], dtype=np.float64)
        next_times = np.asarray(raw['action_timestamps'], dtype=np.float64)
        assert len(state) == len(action) == len(times) == len(next_times)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        np.testing.assert_allclose(np.diff(times), 1 / args.fps, atol=1e-6)
        np.testing.assert_allclose(next_times - times, 1 / args.fps, atol=1e-6)
        assert state.shape == action.shape and state.shape[1] == sum(dims['arm_dim']) + sum(dims['ee_dim'])
        images = {dst: raw['vision'][src]['colors'] for src, dst in camera_map.items()}
        for name, values in images.items():
            assert values.dtype == np.uint8 and values.shape[0] == len(state) and values.shape[-1] == 3, name
        if dataset is None:
            features = {
                'observation.state': {'dtype': 'float32', 'shape': (state.shape[1],), 'names': None},
                'action': {'dtype': 'float32', 'shape': (action.shape[1],), 'names': None},
            }
            for name, values in images.items():
                features[f'observation.images.{name}'] = {'dtype': 'video', 'shape': tuple(values.shape[1:]), 'names': ['height', 'width', 'channels']}
            dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, root=args.output,
                robot_type='aloha_agilex', features=features, video_backend='pyav', vcodec='h264',
                streaming_encoding=True, encoder_threads=2)
            preview = np.concatenate([images[name][0] for name in camera_map.values()], axis=1)
            Image.fromarray(preview).save(PROJECT / 'results/dataset_camera_preview.png')
        task = raw['instructions'][0]
        for i in range(len(state)):
            frame = {'observation.state': state[i], 'action': action[i], 'task': task}
            frame.update({f'observation.images.{name}': values[i] for name, values in images.items()})
            dataset.add_frame(frame)
        dataset.save_episode()
        reports.append({'source': str(path), 'frames': len(state), 'action_dim': int(state.shape[1]), 'state_semantics': state_semantics,
                        'fps': args.fps, 'duration_s': float(next_times[-1] - times[0]), 'task': task})
        print(json.dumps(reports[-1]), flush=True)
    dataset.finalize()
    (PROJECT / 'results' / f'{args.output.name}_validation.json').write_text(json.dumps(reports, indent=2))
    print(f'Converted {len(reports)} episodes to {args.output}')


if __name__ == '__main__':
    main()
