"""Create a complete training JSON from dataset metadata, avoiding stale base shapes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

ROOT=Path('/data/jaka-vla')
BASE=ROOT/'cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--repo-id',required=True)
    p.add_argument('--robot-id',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--initial-checkpoint',type=Path,default=BASE)
    p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    info=json.loads((args.dataset/'meta/info.json').read_text())
    if info['robot_type']!=args.robot_id: raise ValueError('Dataset robot identity does not match requested robot')
    if args.initial_checkpoint!=BASE:
        initial_identity=json.loads((args.initial_checkpoint/'project_identity.json').read_text())
        data_identity=json.loads((args.dataset/'meta/project_identity.json').read_text())
        if initial_identity['robot_id']!=args.robot_id or initial_identity['urdf_sha256']!=data_identity['urdf_sha256']:
            raise ValueError('Initial checkpoint belongs to another robot or tool geometry')
    cfg=json.loads((ROOT/'checkpoints/smolvla_pilot_seed42/checkpoints/last/pretrained_model/train_config.json').read_text())
    policy=json.loads((BASE/'config.json').read_text())
    policy['input_features']={}; policy['output_features']={}
    for key,feature in info['features'].items():
        shape=feature['shape']
        if key.startswith('observation.images.'):
            if shape[-1]!=3: raise ValueError('Expected dataset HWC RGB shape')
            policy['input_features'][key]={'type':'VISUAL','shape':[3,shape[0],shape[1]]}
        elif key=='observation.state': policy['input_features'][key]={'type':'STATE','shape':shape}
        elif key=='action': policy['output_features'][key]={'type':'ACTION','shape':shape}
    assert policy['input_features']['observation.state']['shape']==policy['output_features']['action']['shape']
    policy.update(pretrained_path=str(args.initial_checkpoint),device='cuda',use_amp=True,push_to_hub=False,repo_id=None)
    policy['scheduler_warmup_steps']=200;policy['scheduler_decay_steps']=args.steps
    cfg.update(policy=policy,output_dir=str(args.output),job_name=args.output.name,steps=args.steps,batch_size=args.batch_size,
               save_freq=min(2000,args.steps),eval_freq=0,log_freq=100,num_workers=4,seed=42,resume=False,checkpoint_path=None)
    cfg['dataset'].update(root=str(args.dataset),repo_id=args.repo_id,episodes=None,video_backend='pyav')
    cfg['wandb']['enable']=False
    cfg['scheduler'].update(num_warmup_steps=200,num_decay_steps=args.steps)
    if args.output.exists() or args.config.exists(): raise FileExistsError('Refuse to overwrite run/config')
    args.config.parent.mkdir(parents=True,exist_ok=True)
    args.config.write_text(json.dumps(cfg,indent=2)+'\n')
    identity={'robot_id':args.robot_id,'state_semantics':'measured_qpos','action_semantics':'next_sample_drive_targets','dataset':str(args.dataset),
              'state_dim':policy['input_features']['observation.state']['shape'][0],'control_hz':info['fps'],
              'initial_checkpoint':str(args.initial_checkpoint),'optimizer_resume':False}
    if args.robot_id=='dual_jaka_zu7':
        source_identity=json.loads((args.dataset/'meta/project_identity.json').read_text())
        if source_identity['robot_id']!=args.robot_id:raise ValueError('Identity mismatch')
        identity.update(rig=source_identity['rig'],urdf_sha256=source_identity['urdf_sha256'])
        if 'task_scope' in source_identity:
            identity['task_scope']=source_identity['task_scope']
    args.config.with_suffix('.identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(args.config)


if __name__=='__main__':main()
