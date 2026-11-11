"""Nine fixed teacher-forced predictions on saved expert observations, no rollouts."""
import argparse,json,time
from pathlib import Path
import numpy as np
from openpi.policies.policy_config import create_trained_policy
from openpi.training.data_loader import create_torch_dataset
from jaka_vla.pi05_config import build_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    identity=json.loads((a.checkpoint/'project_identity.json').read_text());cfg=build_config(identity['dataset'],identity['experiment'],horizon=50)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model);dataset=create_torch_dataset(dc,50,cfg.model)
    root=Path('/data/jaka-vla');episodes=[json.loads(x) for x in (root/'data/openpi'/identity['dataset']/'meta/episodes.jsonl').read_text().splitlines()]
    policy=create_trained_policy(cfg,a.checkpoint,sample_kwargs={'num_steps':10})
    noise=np.random.default_rng(42).standard_normal((50,32)).astype(np.float32);rows=[];offset=0
    for episode in episodes[:3]:
        length=episode['length']
        for label,frame in [('initial',0),('middle',length//2),('late',int(length*.75))]:
            sample=dataset[offset+frame];expected=np.asarray(sample['actions']).copy();observation={k:v for k,v in sample.items() if k!='actions'}
            start=time.perf_counter();prediction=policy.infer(observation,noise=noise)['actions'];seconds=time.perf_counter()-start
            if prediction.shape!=(50,14) or not np.isfinite(prediction).all():raise ValueError('Invalid prediction')
            row={'episode':episode['episode_index'],'frame':frame,'phase_sample':label,'instruction':sample['prompt'],
                 'k8_left_joint_rmse_rad':float(np.sqrt(np.mean((prediction[:8,:6]-expected[:8,:6])**2))),
                 'k8_left_gripper_mae':float(np.abs(prediction[:8,6]-expected[:8,6]).mean()),
                 'h50_left_joint_rmse_rad':float(np.sqrt(np.mean((prediction[:,:6]-expected[:,:6])**2))),
                 'inference_seconds':seconds}
            rows.append(row);print(json.dumps(row),flush=True)
        offset+=length
    report={'checkpoint':str(a.checkpoint),'dataset':identity['dataset'],'predictions':rows,'calls':len(rows),'horizon':50,'noise_seed':42,
        'scope':'Nine fixed saved-expert observations from seen layout 0; descriptive fitting audit only, no new trajectory or intervention, no model updates.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
