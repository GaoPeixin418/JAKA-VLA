"""Reload a trained pi0.5 checkpoint and serve the existing Zu7 control protocol."""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path('/data/jaka-vla');sys.path.insert(0,str(ROOT))
from openpi.policies.policy_config import create_trained_policy
from openpi.training.data_loader import create_torch_dataset
from jaka_vla.pi05_config import build_config
from jaka_vla.identity import physical_rig_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--connections',type=int,default=1,help='Number of sequential evaluation clients before closing')
    p.add_argument('--robot-id',required=True);p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--noise-mode',choices=['fixed','random','zero'],default='fixed');p.add_argument('--noise-seed',type=int,default=42)
    p.add_argument('--observation-dir',type=Path,help='Optional exact inference inputs and outputs for offline diagnosis')
    args=p.parse_args()
    if args.observation_dir is not None:args.observation_dir.mkdir(parents=True,exist_ok=False)
    if not 1<=args.connections<=12:raise ValueError('Invalid connection budget')
    identity=json.loads((args.checkpoint/'project_identity.json').read_text())
    if identity['robot_id']!=args.robot_id or identity['model']!='pi05':raise ValueError('Checkpoint identity mismatch')
    config=build_config(identity['dataset'],identity['experiment'],horizon=identity['horizon'])
    start=time.monotonic();policy=create_trained_policy(config,args.checkpoint,sample_kwargs={'num_steps':10})
    data_config=config.data.create(config.assets_dirs,config.model)
    dataset=create_torch_dataset(data_config,config.model.action_horizon,config.model)
    sample=dataset[0];sample.pop('actions',None)
    noise=np.random.default_rng(args.noise_seed).standard_normal((identity['horizon'],32)).astype(np.float32)
    first=policy.infer(sample,noise=noise)['actions']
    if first.shape!=(identity['horizon'],14) or not np.isfinite(first).all():raise ValueError('Reload inference invalid')
    np.save(args.checkpoint/'reload_first_actions.npy',first)
    (args.checkpoint/'reload_verification.json').write_text(json.dumps({'success':True,'fresh_process':True,
        'elapsed_seconds':time.monotonic()-start,'output_shape':list(first.shape),'output_finite':True,
        'checkpoint':str(args.checkpoint),'sample':'training dataset first frame','task_success_claim':False},indent=2)+'\n')
    args.socket.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if args.socket.exists():raise FileExistsError(args.socket)
    with Listener(str(args.socket),family='AF_UNIX') as listener:
        print('READY',args.socket,flush=True)
        for connection_index in range(args.connections):
            prediction_index=0
            with listener.accept() as conn:
                while True:
                    request=conn.recv()
                    if request['op']=='close':conn.send({'closed':True});break
                    if request['op']=='reset':
                        if request['robot_id']!=identity['robot_id'] or request.get('urdf_sha256')!=identity['urdf_sha256'] or physical_rig_hash(request['rig'])!=physical_rig_hash(identity['rig']):
                            raise ValueError('Evaluation rig differs from training rig')
                        seed=request['seed'] if args.noise_seed is None else args.noise_seed
                        noise=np.random.default_rng(seed).standard_normal((identity['horizon'],32)).astype(np.float32)
                        if args.noise_mode=='zero':noise.fill(0)
                        conn.send({'reset':True,'robot_id':args.robot_id,'noise_mode':args.noise_mode,
                                   'noise_seed':seed,'chunk_size':identity['horizon']});continue
                    obs=request['observation'];state=np.asarray(obs['state'],dtype=np.float32)
                    if state.shape!=(14,) or not np.isfinite(state).all() or len(obs['images'])!=3:raise ValueError('Invalid observation')
                    frame={'state':state,'prompt':obs['instruction'],
                           **dict(zip(['image','left_wrist_image','right_wrist_image'],obs['images']))}
                    start=time.monotonic();transformed=policy._input_transform(frame)
                    normalized_state_abs_max=float(np.abs(transformed['state']).max())
                    actions=policy.infer(frame,noise=None if args.noise_mode=='random' else noise)['actions']
                    seconds=time.monotonic()-start
                    if actions.shape!=(identity['horizon'],14) or not np.isfinite(actions).all():raise ValueError('Invalid actions')
                    if args.observation_dir is not None:
                        np.savez_compressed(args.observation_dir/f'connection_{connection_index:02d}_prediction_{prediction_index:04d}.npz',
                            **frame,actions=actions,noise=noise if args.noise_mode!='random' else np.empty(0))
                    prediction_index+=1
                    conn.send({'actions':actions,'inference_seconds':seconds,'noise_mode':args.noise_mode,
                               'normalized_state_abs_max':normalized_state_abs_max})

if __name__=='__main__':main()
