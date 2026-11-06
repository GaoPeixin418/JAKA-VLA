"""Audit actual preprocessing and teacher-forced progress, without robot motion."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = Path('/data/jaka-vla')
PACK = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    dataset = LeRobotDataset(a.repo_id, root=a.dataset, video_backend='pyav')
    policy = SmolVLAPolicy.from_pretrained(a.checkpoint).to('cuda').eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(a.checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    source = []
    for path in sorted(a.raw.glob('episode_*.hdf5')):
        with h5py.File(path) as f:
            source.append({key: np.concatenate([f[f'{key}/{part}'][:] for part in PACK], axis=1).astype(np.float32)
                           for key in ('state', 'action')})
    rows = []
    saved = []
    for ep in list(dataset.meta.episodes):
        ei = int(ep['episode_index'])
        for frame in (0, 20, 40, 60, 80):
            item = dataset[int(ep['dataset_from_index'])+frame]
            obs = {k: v for k, v in item.items() if k.startswith('observation.') or k == 'task'}
            batch = pre(obs)
            normalized = batch['observation.state'].detach().cpu().numpy()[0]
            row = {'episode': ei, 'frame': frame, 'normalized_abs_max': float(abs(normalized).max()),
                   'normalized_state': normalized.tolist(), 'predictions': []}
            for seed in (6, 42):
                g = torch.Generator(device='cuda').manual_seed(seed)
                noise = torch.randn((1, 50, policy.config.max_action_dim), generator=g, device='cuda')
                with torch.inference_mode():
                    pred = post(policy.predict_action_chunk(batch, noise=noise))[0].float().cpu().numpy()
                indices = np.minimum(np.arange(frame, frame+50), len(source[ei]['action'])-1)
                target = source[ei]['action'][indices]
                row['predictions'].append({'noise_seed': seed,
                    'left_joint_mae_deg_h8': float(np.rad2deg(abs(pred[:8, :6]-target[:8, :6])).mean()),
                    'left_joint_mae_deg_h50': float(np.rad2deg(abs(pred[:, :6]-target[:, :6])).mean()),
                    'left_gripper_h8_mean': float(pred[:8, 6].mean()),
                    'target_gripper_h8_mean': float(target[:8, 6].mean()),
                    'left_joint_predicted_progress_h8_rad': float(np.linalg.norm(pred[7, :6]-source[ei]['state'][frame, :6])),
                    'left_joint_expert_progress_h8_rad': float(np.linalg.norm(target[7, :6]-source[ei]['state'][frame, :6]))})
                saved.append(pred)
            rows.append(row)
    # Apply exactly the saved inference processor to existing real rollout states.
    trace = np.load(a.rollout)['control_trace']
    item = dataset[0]
    obs = {k: v for k, v in item.items() if k.startswith('observation.') or k == 'task'}
    rollout_rows = []
    for step in (0, 40, 80, 120, 160, 199):
        step = min(step, len(trace)-1)
        obs['observation.state'] = torch.from_numpy(trace[step, 1:15].astype(np.float32))
        norm = pre(obs)['observation.state'].detach().cpu().numpy()[0]
        rollout_rows.append({'step': step, 'normalized_state': norm.tolist(), 'abs_max': float(abs(norm).max())})
    report = {'checkpoint': str(a.checkpoint), 'dataset': str(a.dataset), 'calls': len(saved),
              'elapsed_seconds': time.perf_counter()-start, 'teacher_forced': rows, 'rollout_preprocessing': rollout_rows,
              'scope': 'Offline diagnostic; rollout images are not re-predicted; no task-success claim'}
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(a.output/'predictions.npz', predictions=np.stack(saved))
    print(json.dumps({'calls': len(saved), 'rollout_abs_max': [r['abs_max'] for r in rollout_rows],
                       'teacher_forced_mean_left_joint_mae_h8_deg': float(np.mean([p['left_joint_mae_deg_h8'] for r in rows for p in r['predictions']]))}))


if __name__ == '__main__':
    main()
