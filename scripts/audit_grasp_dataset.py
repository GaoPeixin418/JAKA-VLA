"""Check saved LeRobot observations/actions against raw identified recovery data."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PACK = ['left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    dataset=LeRobotDataset(a.repo_id,root=a.dataset,video_backend='pyav')
    identity=json.loads((a.raw.parent/'robot_identity.json').read_text())
    rows=[]
    for ep in list(dataset.meta.episodes):
        index=int(ep['episode_index'])
        with h5py.File(a.raw/f'episode_{index:07d}.hdf5') as f:
            assert f.attrs['state_semantics']=='measured_qpos'
            assert f.attrs['rig_urdf_sha256']==identity['urdf_sha256']
            state=np.concatenate([f[f'state/{k}'][:] for k in PACK],axis=1).astype(np.float32)
            action=np.concatenate([f[f'action/{k}'][:] for k in PACK],axis=1).astype(np.float32)
            command=np.concatenate([f[f'commanded_state/{k}'][:] for k in PACK],axis=1).astype(np.float32)
            times=f['observation_timestamps'][:]
            next_times=f['action_timestamps'][:]
            np.testing.assert_allclose(np.diff(times),.04,atol=1e-6)
            np.testing.assert_allclose(next_times-times,.04,atol=1e-6)
            np.testing.assert_allclose(action[:-1],command[1:],atol=1e-6)
        assert len(state)==int(ep['length'])
        differences=[]
        for frame in (0,len(state)//2,len(state)-1):
            item=dataset[int(ep['dataset_from_index'])+frame]
            assert int(item['episode_index'])==index and int(item['frame_index'])==frame
            s=item['observation.state'].numpy();act=item['action'].numpy()
            differences.append(float(max(abs(s-state[frame]).max(),abs(act-action[frame]).max())))
            np.testing.assert_allclose(s,state[frame],atol=1e-6)
            np.testing.assert_allclose(act,action[frame],atol=1e-6)
            assert item['task']=='Use the left arm to grasp the red block, lift it, and hold it.'
            for camera in (1,2,3):
                image=item[f'observation.images.camera{camera}']
                assert tuple(image.shape)==(3,240,320) and image.min()>=0 and image.max()<=1
        condition=identity.get('initial_conditions_by_episode',{}).get(str(index))
        initial_error=None
        if condition is not None:
            initial_error=float(abs(state[0,:6]-condition['measured_initial_left_joint_rad']).max())
            assert initial_error<1e-5
        rows.append({'episode':index,'frames':len(state),'max_sample_state_action_difference':max(differences),
                     'initial_condition_error_rad':initial_error})
    report={'episodes':len(rows),'frames':sum(r['frames'] for r in rows),'checks':rows,
            'scope':'raw-to-converted sampling, labels, initial conditions, RGB shape/range; no policy success claim'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'episodes':report['episodes'],'frames':report['frames'],'all_contract_checks_passed':True}))


if __name__=='__main__':main()
