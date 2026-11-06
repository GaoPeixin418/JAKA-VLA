"""Bounded teacher-forced action audit: two observations per demonstration.

    Uses fixed noise 42 and expert observations; no simulation, training or claim
that low action error implies closed-loop grasp success.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.perf_counter()
    dataset = LeRobotDataset(args.repo_id, root=args.dataset, video_backend='pyav')
    identity = json.loads((args.dataset / 'meta/project_identity.json').read_text())
    model_identity = json.loads((args.checkpoint / 'project_identity.json').read_text())
    for key in ('robot_id', 'urdf_sha256', 'task_scope'):
        if identity[key] != model_identity[key]:
            raise ValueError('Dataset/checkpoint identity mismatch: ' + key)
    episodes = list(dataset.meta.episodes)
    if len(episodes) != 14:
        raise ValueError('This bounded audit expects exactly 14 demonstrations')
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to('cuda').eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint),
                                        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    if policy.config.chunk_size != 50:
        raise ValueError('Expected prediction length 50')
    rows, saved, targets = [], [], []
    pack_keys = ('left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states')
    for episode in episodes:
        index = int(episode['episode_index'])
        with h5py.File(args.raw / f'episode_{index:07d}.hdf5') as raw:
            action = np.concatenate([raw['action/' + key][:] for key in pack_keys], axis=1).astype(np.float32)
            state = np.concatenate([raw['state/' + key][:] for key in pack_keys], axis=1).astype(np.float32)
        if len(action) != int(episode['length']):
            raise ValueError('Raw/converted episode length mismatch')
        closes = np.flatnonzero(action[:, 6] < .95)
        if not len(closes):
            raise ValueError('No gripper closing command in demonstration')
        # This threshold is only a sampling heuristic. Recovery starts can
        # already be below .95, in which case both sampled frames are zero.
        frames = [('initial', 0), ('before_expert_close', max(0, int(closes[0]) - 8))]
        condition = identity.get('initial_conditions_by_episode', {}).get(str(index)) or {}
        group = 'recovery_demonstration' if condition.get('scope') == 'intervened_recovery_start_not_executed_prefix' else 'ordinary_demonstration'
        for phase, frame in frames:
            item = dataset[int(episode['dataset_from_index']) + frame]
            if int(item['frame_index']) != frame or int(item['episode_index']) != index:
                raise ValueError('Episode/global frame mapping mismatch')
            np.testing.assert_allclose(item['observation.state'].numpy(), state[frame], atol=1e-6)
            observation = {key: value for key, value in item.items() if key.startswith('observation.') or key == 'task'}
            policy.reset()
            generator = torch.Generator(device='cuda').manual_seed(42)
            noise = torch.randn((1, 50, policy.config.max_action_dim), device='cuda', generator=generator)
            with torch.inference_mode():
                prediction = post(policy.predict_action_chunk(pre(observation), noise=noise))[0].float().cpu().numpy()
            target = action[np.minimum(np.arange(frame, frame + 50), len(action) - 1)]
            error = abs(prediction - target)
            row = {'episode': index, 'phase': phase, 'frame': frame, 'group': group,
                   'left_joint_mae_deg_h1': float(np.rad2deg(error[0, :6]).mean()),
                   'left_joint_mae_deg_h8': float(np.rad2deg(error[:8, :6]).mean()),
                   'left_aperture_mae_h8': float(error[:8, 6].mean()),
                   'predicted_left_aperture_h1': float(prediction[0, 6]),
                   'expert_left_aperture_h1': float(target[0, 6]),
                   'initial_condition': condition}
            rows.append(row)
            saved.append(prediction)
            targets.append(target)
    summary = {}
    for group in sorted({row['group'] for row in rows}):
        for phase in ('initial', 'before_expert_close'):
            chosen = [row for row in rows if row['group'] == group and row['phase'] == phase]
            summary[group + '/' + phase] = {'observations': len(chosen), **{
                key: float(np.mean([row[key] for row in chosen])) for key in
                ('left_joint_mae_deg_h1', 'left_joint_mae_deg_h8', 'left_aperture_mae_h8')}}
    args.output.mkdir(parents=True)
    report = {'scope': 'Teacher-forced offline inference on training observations; not closed-loop success',
              'calls': len(rows), 'noise_seed': 42, 'elapsed_s': time.perf_counter() - start,
              'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': hashlib.sha256((args.checkpoint / 'model.safetensors').read_bytes()).hexdigest(),
              'summary': summary, 'rows': rows,
              'limitations': ['Two calls per episode, potentially the same frame; no full error distribution.',
                              'before_expert_close means eight frames before the first command below .95; it is a heuristic, not a semantic phase label. Recovery starts can already be below .95.',
                              'Expert observations and fixed noise differ from learner rollouts.',
                              'Joint MAE is not TCP error or a grasp success metric.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'predictions.npz', predictions=np.stack(saved), targets=np.stack(targets))
    print(json.dumps({'calls': len(rows), 'elapsed_s': report['elapsed_s'], 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
