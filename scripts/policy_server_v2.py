"""Strict robot-bound SmolVLA inference with explicit sampling mode and metadata."""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import sys,time
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
ROOT=Path('/data/jaka-vla')
sys.path.insert(0,str(ROOT))
from jaka_vla.identity import physical_rig_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--robot-id',required=True)
    p.add_argument('--socket',type=Path,default=ROOT/'run/policy-v2.sock')
    p.add_argument('--noise-mode',choices=['fixed','random','zero'],default='fixed')
    p.add_argument('--noise-seed',type=int);args=p.parse_args()
    identity=json.loads((args.checkpoint/'project_identity.json').read_text())
    if identity['robot_id']!=args.robot_id: raise ValueError('Checkpoint robot mismatch; cross-robot reuse is not permitted')
    policy=SmolVLAPolicy.from_pretrained(args.checkpoint).to('cuda').eval()
    size=identity['state_dim']
    if policy.config.input_features['observation.state'].shape!=(size,) or policy.config.action_feature.shape!=(size,):
        raise ValueError('Checkpoint input/output shape does not match robot identity')
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(args.checkpoint),preprocessor_overrides={'device_processor':{'device':'cuda'}})
    args.socket.parent.mkdir(mode=0o700,exist_ok=True);args.socket.parent.chmod(0o700)
    if args.socket.exists():raise FileExistsError(args.socket)
    noise=None
    with Listener(str(args.socket),family='AF_UNIX') as listener:
        print('READY',args.socket,flush=True)
        with listener.accept() as conn:
            while True:
                request=conn.recv()
                if request['op']=='close':conn.send({'closed':True});break
                if request['op']=='reset':
                    if request['robot_id']!=args.robot_id: raise ValueError('Client robot mismatch')
                    if request.get('urdf_sha256')!=identity.get('urdf_sha256') or physical_rig_hash(request['rig'])!=physical_rig_hash(identity['rig']):
                        raise ValueError('Client rig differs from checkpoint training rig')
                    noise_seed=request['seed'] if args.noise_seed is None else args.noise_seed
                    torch.manual_seed(noise_seed);policy.reset()
                    noise=torch.randn((1,policy.config.chunk_size,policy.config.max_action_dim),device='cuda')
                    if args.noise_mode=='zero':noise.zero_()
                    conn.send({'reset':True,'robot_id':args.robot_id,'noise_mode':args.noise_mode,'noise_seed':noise_seed,'chunk_size':policy.config.chunk_size});continue
                obs=request['observation'];state=np.asarray(obs['state'],dtype=np.float32)
                if state.shape!=(size,) or not np.isfinite(state).all(): raise ValueError('Invalid state tensor')
                frame={'observation.state':torch.from_numpy(state),'task':obs['instruction']}
                camera_keys=list(policy.config.image_features)
                if len(obs['images'])!=len(camera_keys):raise ValueError('Camera count mismatch')
                for key,rgb in zip(camera_keys,obs['images']):
                    rgb=np.asarray(rgb)
                    expected=policy.config.image_features[key].shape
                    if rgb.dtype!=np.uint8 or rgb.shape!=(expected[1],expected[2],3):raise ValueError(f'RGB shape mismatch {key}: {rgb.shape} vs {expected}')
                    frame[key]=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255.
                with torch.inference_mode():
                    torch.cuda.synchronize();start=time.perf_counter()
                    batch=pre(frame)
                    output=post(policy.predict_action_chunk(batch,noise=None if args.noise_mode=='random' else noise))
                    torch.cuda.synchronize();duration=time.perf_counter()-start
                action=output[0].float().cpu().numpy()
                if action.shape!=(policy.config.chunk_size,size) or not np.isfinite(action).all():raise ValueError('Invalid prediction')
                conn.send({'actions':action,'inference_seconds':duration,'noise_mode':args.noise_mode})


if __name__=='__main__':main()
