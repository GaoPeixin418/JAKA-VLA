"""Compare pi0.5 action sensitivity with task prompts on identical observations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from openpi.policies.policy_config import create_trained_policy
from openpi.training.data_loader import create_torch_dataset
from jaka_vla.pi05_config import build_config


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs=3, default=[0, 1, 2])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = json.loads((args.checkpoint / 'project_identity.json').read_text())
    root = Path('/data/jaka-vla')
    cfg = build_config(identity['dataset'], identity['experiment'], steps=6000, horizon=identity['horizon'])
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = create_torch_dataset(data_config, identity['horizon'], cfg.model)
    episodes = [json.loads(line) for line in
                (root / 'data/openpi' / identity['dataset'] / 'meta/episodes.jsonl').read_text().splitlines()]
    chosen, offset = [], 0
    for episode in episodes:
        if episode['episode_index'] in args.episodes:
            chosen.append((offset, episode))
        offset += episode['length']
    if len(chosen) != 3:
        raise ValueError('Could not find all three requested episodes')
    samples = [dataset[offset] for offset, _ in chosen]
    state_diffs, image_diffs = [], []
    for left in range(3):
        for right in range(left + 1, 3):
            state_diffs.append(float(np.max(np.abs(np.asarray(samples[left]['state']) - np.asarray(samples[right]['state'])))))
            for key in ('image', 'left_wrist_image', 'right_wrist_image'):
                # LeRobot decodes floating RGB in [0,1]; integer casting erased differences.
                image_diffs.append(float(np.max(np.abs(np.asarray(samples[left][key],dtype=np.float32) -
                                                        np.asarray(samples[right][key],dtype=np.float32)))))

    policy = create_trained_policy(cfg, args.checkpoint, sample_kwargs={'num_steps': 10})
    noise = np.random.default_rng(42).standard_normal((identity['horizon'], 32)).astype(np.float32)
    expert = [np.asarray(sample['actions']) for sample in samples]
    predictions = []
    for image_source in range(3):
        observation = {key: value for key, value in samples[image_source].items() if key != 'actions'}
        for prompt_source in range(3):
            frame = dict(observation)
            frame['prompt'] = samples[prompt_source]['prompt']
            prediction = policy.infer(frame, noise=noise)['actions']
            if prediction.shape != (identity['horizon'], 14) or not np.isfinite(prediction).all():
                raise ValueError('Invalid policy output')
            predictions.append({'image_source': image_source, 'prompt_source': prompt_source,
                                'prompt': frame['prompt'], 'actions': prediction})

    sensitivity = []
    for image_source in range(3):
        rows = [item for item in predictions if item['image_source'] == image_source]
        for left in range(3):
            for right in range(left + 1, 3):
                sensitivity.append({'image_source': image_source, 'prompt_pair': [left, right],
                                    'predicted_k8_joint_rms_rad': rms(rows[left]['actions'][:8, :6] - rows[right]['actions'][:8, :6]),
                                    'predicted_h50_joint_rms_rad': rms(rows[left]['actions'][:, :6] - rows[right]['actions'][:, :6]),
                                    'predicted_k8_gripper_rms': rms(rows[left]['actions'][:8, 6] - rows[right]['actions'][:8, 6]),
                                    'expert_k8_joint_rms_rad': rms(expert[left][:8, :6] - expert[right][:8, :6]),
                                    'expert_h50_joint_rms_rad': rms(expert[left][:, :6] - expert[right][:, :6])})

    cross_errors = []
    for item in predictions:
        for target in range(3):
            cross_errors.append({'image_source': item['image_source'], 'prompt_source': item['prompt_source'],
                                 'expert_target': target,
                                 'k8_left_joint_rmse_rad': rms(item['actions'][:8, :6] - expert[target][:8, :6]),
                                 'h50_left_joint_rmse_rad': rms(item['actions'][:, :6] - expert[target][:, :6])})
    report = {'checkpoint': str(args.checkpoint), 'dataset': identity['dataset'],
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'episodes': [{'episode_index': episode['episode_index'], 'prompt': samples[i]['prompt'],
                            'length': episode['length']} for i, (_, episode) in enumerate(chosen)],
              'same_initial_state_max_abs_difference': max(state_diffs),
              'same_initial_image_max_abs_difference': max(image_diffs),
              'image_difference_units': 'decoded float RGB [0,1]',
              'prompt_sensitivity': sensitivity, 'cross_target_errors': cross_errors,
              'scope': 'Each prompt comparison holds its image source and state exactly fixed. Different video sources may differ after codec decoding. Fixed noise seed 42; sensitivity alone does not establish correctness; no rollout or weight updates.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'same_initial_state_max_abs_difference': max(state_diffs),
                      'same_initial_image_max_abs_difference': max(image_diffs),
                      'prompt_sensitivity': sensitivity}, indent=2), flush=True)


if __name__ == '__main__':
    main()
