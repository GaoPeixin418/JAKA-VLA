"""Bounded offline diagnosis of the existing pilot; no training or simulator rollout."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path('/data/jaka-vla')
OUT = ROOT / 'results/offline_audit_20261105_v2'
sys.path.insert(0, str(ROOT/'repos/RoboTwin'))
from XPolicyLab.utils.process_data import decode_image_bit
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
KEYS = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']


def pack(hf, name):
    return np.concatenate([hf[name][key][:] for key in KEYS], axis=1).astype(np.float32)


def stats(values):
    values = np.asarray(values)
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'max': float(values.max())}


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    report = {'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
              'scope': 'Offline inference on training demonstration observations; no training or closed-loop rollout.'}
    validation = json.loads((ROOT/'results/handover_pilot_validation.json').read_text())
    offsets = np.cumsum([0] + [ep['frames'] for ep in validation])
    states, actions, commands, aligns = [], [], [], []
    raw_files = sorted((ROOT/'data/raw/jaka_pilot/handover_block/aloha_agilex/data').glob('episode_*.hdf5'))
    for path in raw_files:
        with h5py.File(path) as hf:
            state, action, command = [pack(hf, key) for key in ('state', 'action', 'commanded_state')]
        states.append(state); actions.append(action); commands.append(command)
        aligns.append(float(np.max(np.abs(action[:-1] - command[1:]))))
    expert_deltas = np.concatenate([np.abs(np.diff(a[:, JOINTS], axis=0)) for a in actions])
    tracking = np.concatenate([np.abs(s[:, JOINTS]-c[:, JOINTS]) for s,c in zip(states,commands)])
    report['dataset'] = {'episodes': len(actions), 'frames': int(offsets[-1]),
                         'sample_start_exposures': 2000*8, 'nominal_passes': float(2000*8/offsets[-1]),
                         'action_next_command_alignment_max_abs': max(aligns),
                         'expert_joint_step_abs_rad': stats(expert_deltas),
                         'expert_tracking_abs_rad': stats(tracking)}
    dataset = LeRobotDataset('jaka-local/handover-pilot', root=ROOT/'data/lerobot/handover_pilot', video_backend='pyav')
    checkpoint = ROOT/'checkpoints/smolvla_pilot_seed42/checkpoints/last/pretrained_model'
    policy = SmolVLAPolicy.from_pretrained(checkpoint).to('cuda').eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    samples = {}
    def observation(ep, t):
        key = (ep,t)
        if key not in samples:
            frame = dataset[int(offsets[ep])+t]
            samples[key] = {k:v for k,v in frame.items() if k.startswith('observation.') or k=='task'}
        return samples[key]
    calls = 0
    def predict(ep, t, seed):
        nonlocal calls
        policy.reset()
        torch.manual_seed(seed)
        with torch.inference_mode():
            predicted = post(policy.predict_action_chunk(pre(observation(ep,t))))
        calls += 1
        return predicted[0].float().cpu().numpy()
    obs0 = observation(0,0)
    normalized = pre(obs0)
    mean = torch.as_tensor(dataset.meta.stats['observation.state']['mean'], device='cuda')
    std = torch.as_tensor(dataset.meta.stats['observation.state']['std'], device='cuda')
    expected = (obs0['observation.state'].to('cuda')-mean)/(std+1e-8)
    state_tensor = normalized['observation.state']
    padded = policy.prepare_state(normalized)
    action_mean = torch.as_tensor(dataset.meta.stats['action']['mean'])
    with torch.inference_mode():
        post_zero = post(torch.zeros((1,50,14), device='cuda'))
    report['interfaces'] = {'config_state_shape': list(policy.config.input_features['observation.state'].shape),
        'dataset_state_shape': list(obs0['observation.state'].shape),
        'preprocessed_state_shape': list(state_tensor.shape), 'padded_state_shape': list(padded.shape),
        'state_norm_max_error_vs_dataset': float((state_tensor[0]-expected).abs().max()),
        'state_first14_preserved': bool(torch.equal(padded[:,:14], state_tensor)),
        'post_zero_max_error_vs_dataset_mean': float((post_zero[0,0].cpu()-action_mean).abs().max()),
        'dataset_state_vs_hdf5_max_error': float(np.abs(obs0['observation.state'].numpy()-states[0][0]).max()),
        'image_pixel_comparison': {}}
    with h5py.File(raw_files[0]) as hf:
        for idx, camera in enumerate(['cam_head','cam_left_wrist','cam_right_wrist'],1):
            raw_image=decode_image_bit(hf['vision'][camera]['colors'][0])
            video_image=obs0[f'observation.images.camera{idx}'].permute(1,2,0).numpy()*255
            report['interfaces']['image_pixel_comparison'][camera] = {
                'shape': list(raw_image.shape), 'rgb_mae_255': float(np.abs(video_image-raw_image).mean()),
                'channel_reversed_mae_255': float(np.abs(video_image-raw_image[...,::-1]).mean())}
    print('Data and interface audit complete.', flush=True)
    repeats = np.stack([predict(0,0,seed) for seed in range(32)])
    repeat_std = repeats[:,0,:].std(axis=0)
    repeat_pairs = np.abs(np.diff(repeats[:,0,JOINTS], axis=0))
    same_seed = predict(0,0,0)
    report['same_observation_sampling'] = {'repetitions':32, 'first_action_std_each_dimension':repeat_std.tolist(),
        'first_joint_action_std_deg': np.rad2deg(repeat_std[JOINTS]).tolist(),
        'consecutive_samples_first_joint_delta_rad':stats(repeat_pairs),
        'identical_seed_max_abs_error':float(np.abs(same_seed-repeats[0]).max()),
        'first_action_joint_error_vs_expert_rad':stats(np.abs(repeats[:,0,JOINTS]-actions[0][0,JOINTS])),
        'within_chunk_joint_step_abs_rad':stats(np.abs(np.diff(repeats[:,:16,JOINTS],axis=1)))}
    print('Same-observation sampling audit complete.', flush=True)
    predictions, targets, indices, details = [], [], [], []
    for ep in (0,10,25,49):
        for t in (0,50,150,250):
            pred = np.stack([predict(ep,t,seed) for seed in (100,101,102,103)])
            target = actions[ep][t:t+50]
            predictions.append(pred); targets.append(target); indices.append([ep,t])
            details.append({'episode':ep,'frame':t,
                'first_joint_mae_deg':float(np.rad2deg(np.abs(pred[:,0,JOINTS]-target[0,JOINTS])).mean()),
                'first_gripper_mae':np.abs(pred[:,0,[6,13]]-target[0,[6,13]]).mean(axis=0).tolist(),
                'expert_first_grippers':target[0,[6,13]].tolist(),
                'predicted_first_grippers_mean':pred[:,0,[6,13]].mean(axis=0).tolist()})
    pred = np.stack(predictions); target = np.stack(targets)
    error = np.abs(pred - target[:,None])
    report['training_observation_prediction'] = {'observations':16,'samples_per_observation':4,
        'first_joint_abs_error_rad':stats(error[:,:,0,JOINTS]),
        'first_joint_mae_deg':float(np.rad2deg(error[:,:,0,JOINTS]).mean()),
        'first_gripper_abs_error':stats(error[:,:,0,[6,13]]),
        'chunk_joint_abs_error_rad':stats(error[:,:,:,JOINTS]),
        'predicted_gripper_outside_0_1_fraction':float(np.mean((pred[:,:,:,[6,13]]<0)|(pred[:,:,:,[6,13]]>1))),
        'observations_detail':details}
    print('Offline demonstration prediction audit complete.', flush=True)
    trajectory = {t:predict(0,t,2000+t) for t in range(0,65,4)}
    report['expert_observation_chunk_boundaries'] = {}
    for k in (4,8,16):
        jumps = np.stack([np.abs(trajectory[t][0,JOINTS]-trajectory[t-k][k-1,JOINTS]) for t in range(k,65,k)])
        reference = np.stack([np.abs(actions[0][t,JOINTS]-actions[0][t-1,JOINTS]) for t in range(k,65,k)])
        report['expert_observation_chunk_boundaries'][str(k)] = {
            'predicted_boundary_joint_delta_rad':stats(jumps),
            'expert_boundary_joint_delta_rad':stats(reference),
            'note':'Teacher-forced observations along training episode 0; not measured rollout jitter.'}
    report['calls'] = calls
    report['elapsed_seconds'] = time.perf_counter()-start
    np.savez_compressed(OUT/'offline_arrays.npz', repeats=repeats, predictions=pred, targets=target,
                        indices=indices, expert_initial=actions[0][:50],
                        boundary_times=np.array(list(trajectory)), boundary_predictions=np.stack(list(trajectory.values())))
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('training_observation_prediction',)},ensure_ascii=False,indent=2),flush=True)
    print(json.dumps({k:v for k,v in report['training_observation_prediction'].items() if k!='observations_detail'},indent=2),flush=True)


if __name__=='__main__':
    main()
