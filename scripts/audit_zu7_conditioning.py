"""Bounded offline input-sensitivity audit; no training or robot execution.

Interventions are not labelled tasks. Output differences demonstrate sensitivity,
not instruction following, localization accuracy or generalization.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = Path('/data/jaka-vla')
CAMERAS = [f'observation.images.camera{i}' for i in (1, 2, 3)]
JOINTS = np.r_[np.arange(6), np.arange(7, 13)]


def main():
    output = ROOT / 'results/zu7_conditioning_audit_20261106'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    configs = [
        ('6000', 'smolvla_zu7_refine3000_seed42', 'zu7_collision_smoke', 'jaka-local/zu7-collision-smoke'),
        ('9000', 'smolvla_zu7_independent3000_seed42', 'zu7_independent_accepted', 'jaka-local/zu7-independent-accepted'),
    ]
    variants = ['original', 'repeat', 'black_images', 'other_episode_images', 'empty_instruction',
                'right_arm_instruction', 'evaluation_instruction']
    report = {'scope': 'Offline teacher-forced sensitivity; no training, simulator or success-rate evaluation',
              'limitations': ['Six training observations per checkpoint, one fixed noise seed.',
                             'Image/state mismatches and unseen instructions may be out of distribution.',
                             'Differences do not establish correct understanding; small differences do not prove no use.',
                             'Do not compare absolute deltas across input modalities as attribution weights.'],
              'checkpoints': []}
    arrays = {}; calls = 0
    for label, checkpoint_name, dataset_name, repo_id in configs:
        checkpoint = ROOT / 'checkpoints' / checkpoint_name / 'checkpoints/last/pretrained_model'
        dataset = LeRobotDataset(repo_id, root=ROOT/'data/lerobot'/dataset_name, video_backend='pyav')
        episodes = list(dataset.meta.episodes)[:3]
        samples = [dataset[int(ep['dataset_from_index'])+frame] for ep in episodes for frame in (0,60)]
        observations = [{k:v for k,v in item.items() if k.startswith('observation.') or k=='task'} for item in samples]
        policy = SmolVLAPolicy.from_pretrained(checkpoint).to('cuda').eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': 'cuda'}})
        generator = torch.Generator(device='cuda').manual_seed(42)
        noise = torch.randn((1,policy.config.chunk_size,policy.config.max_action_dim),generator=generator,device='cuda')
        rows = []; saved = []
        for index, original in enumerate(observations):
            predictions = []
            for variant in variants:
                obs = dict(original)
                if variant == 'black_images':
                    for key in CAMERAS: obs[key] = torch.zeros_like(original[key])
                elif variant == 'other_episode_images':
                    for key in CAMERAS: obs[key] = observations[(index+2)%len(observations)][key]
                elif variant == 'empty_instruction': obs['task'] = ''
                elif variant == 'right_arm_instruction':
                    obs['task'] = 'Use the right arm to pick up the red block and place it directly on the blue pad without handing it over.'
                elif variant == 'evaluation_instruction':
                    obs['task'] = 'Use the left arm to grab the red block, hand it to the right arm, and place it on the blue pad.'
                policy.reset()
                with torch.inference_mode():
                    action = post(policy.predict_action_chunk(pre(obs),noise=noise.clone()))[0].float().cpu().numpy()
                predictions.append(action); calls += 1
            predictions = np.stack(predictions); saved.append(predictions)
            assert np.isfinite(predictions).all()
            delta = np.abs(predictions-predictions[0:1])
            assert delta[1].max()<1e-6, 'Fixed observation/noise repeat is not deterministic'
            row = {'episode': int(samples[index]['episode_index']), 'frame': int(samples[index]['frame_index']),
                   'training_instruction': str(original['task']), 'variants': {}}
            for vi,variant in enumerate(variants):
                row['variants'][variant] = {str(h):{
                    'joint_delta_mae_deg': float(np.rad2deg(delta[vi,:h][:,JOINTS]).mean()),
                    'gripper_delta_mae': float(delta[vi,:h][:,[6,13]].mean()),
                } for h in (1,8,50)}
            rows.append(row)
            print(f'{label} observation {index+1}/6 complete',flush=True)
        arrays[f'predictions_{label}'] = np.stack(saved)
        summary = {variant: {str(h): {key:float(np.mean([x['variants'][variant][str(h)][key] for x in rows]))
                   for key in ('joint_delta_mae_deg','gripper_delta_mae')} for h in (1,8,50)} for variant in variants}
        report['checkpoints'].append({'name':checkpoint_name,'dataset':dataset_name,'weight_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                                      'noise_seed':42,'observations':rows,'summary':summary})
        del policy,pre,post,dataset
        torch.cuda.empty_cache()
    report.update(prediction_calls=calls,elapsed_seconds=time.perf_counter()-started,variant_order=variants)
    np.savez_compressed(output/'predictions.npz',**arrays)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'calls':calls,'elapsed_seconds':report['elapsed_seconds'],
                      'summary':{x['name']:x['summary'] for x in report['checkpoints']}},indent=2),flush=True)


if __name__ == '__main__': main()
