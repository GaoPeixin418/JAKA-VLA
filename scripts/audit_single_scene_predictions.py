"""Bounded same-scene fit audit using saved inputs, never a control intervention."""
import argparse,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from openpi.policies.policy_config import create_trained_policy
from openpi.training.data_loader import create_torch_dataset
from jaka_vla.pi05_config import build_config
from jaka_vla.openpi_zu7 import _image

def rms(x):return float(np.sqrt(np.mean(np.square(x))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 out=a.run/'prediction_fit_audit.json'
 if out.exists():raise FileExistsError(out)
 identity=json.loads((a.checkpoint/'project_identity.json').read_text());root=Path('/data/jaka-vla')
 c=build_config(identity['dataset'],identity['experiment'],horizon=50);d=create_torch_dataset(c.data.create(c.assets_dirs,c.model),50,c.model)
 table=pq.read_table(root/'data/openpi'/identity['dataset']/'data/chunk-000/episode_000000.parquet').to_pydict()
 states=np.asarray(table['state']);targets=np.asarray(table['actions'])
 inputs=sorted((a.run/'policy_observations').glob('*.npz'))
 k=json.loads((a.run/'summary.json').read_text())['conditions']['K']
 image_keys=['image','left_wrist_image','right_wrist_image'];rows=[];online=[];initial={}
 policy=create_trained_policy(c,a.checkpoint,sample_kwargs={'num_steps':10})
 noise=np.random.default_rng(42).standard_normal((50,32)).astype(np.float32)
 for i in [0,40,80,120,160,200,240,280,300,329]:
  sample=d[i];expert=np.asarray(sample['actions']);obs={k:v for k,v in sample.items() if k!='actions'}
  pred=policy.infer(obs,noise=noise)['actions']
  rows.append({'frame':i,'left_k8_rmse_rad':rms(pred[:8,:6]-expert[:8,:6]),'left_h50_rmse_rad':rms(pred[:,:6]-expert[:,:6]),'left_k8_bias_rad':(pred[:8,:6]-expert[:8,:6]).mean(0).tolist(),'gripper_k8_rmse':rms(pred[:8,6]-expert[:8,6]),'predicted_gripper':pred[:8,6].tolist(),'expert_gripper':expert[:8,6].tolist()})
  if i==0:initial['offline_actions']=pred;initial['expert_actions']=expert;initial['sample']=sample
 for j,path in enumerate(inputs):
  with np.load(path) as z:data={k:z[k] for k in z.files}
  state=data['state'];actions=data['actions'];dist=np.sqrt(np.mean((states[:,:6]-state[:6])**2,axis=1));near=int(dist.argmin())
  expert=np.asarray(d[near]['actions'])
  online.append({'call':j,'control_step':j*k,'nearest_expert_frame':near,'nearest_state_rmse_rad':float(dist[near]),'k8_action_rmse_to_nearest_expert_rad':rms(actions[:8,:6]-expert[:8,:6]),'left_state':state[:6].tolist(),'first_action':actions[0,:7].tolist()})
  if j==0:
   initial['online_actions']=actions;initial['online_state']=state;initial['online_images']={k:data[k] for k in image_keys}
   reloaded=policy.infer({k:data[k] if k!='prompt' else str(data[k]) for k in ['state','prompt',*image_keys]},noise=noise)['actions']
   initial['saved_action_reproduction_max_error']=float(np.abs(reloaded-actions).max())
 sample=initial.pop('sample');images=initial.pop('online_images');state=initial.pop('online_state');off=initial.pop('offline_actions');on=initial.pop('online_actions');exp=initial.pop('expert_actions')
 initial.update(state_max_abs_diff=float(np.abs(state-np.asarray(sample['state'])).max()),online_k8_expert_rmse=rms(on[:8,:6]-exp[:8,:6]),offline_k8_expert_rmse=rms(off[:8,:6]-exp[:8,:6]),online_offline_action_max_diff=float(np.abs(on-off).max()),images={})
 canvas=Image.new('RGB',(960,480))
 for n,k in enumerate(image_keys):
  left=_image(sample[k]);right=_image(images[k]);diff=np.abs(left.astype(float)-right.astype(float))
  initial['images'][k]={'max_difference_255':float(diff.max()),'mean_difference_255':float(diff.mean())}
  canvas.paste(Image.fromarray(left),(n*320,0));canvas.paste(Image.fromarray(right),(n*320,240))
 canvas.save(a.run/'initial_dataset_vs_online.png')
 report={'checkpoint':str(a.checkpoint),'initial':initial,'expert_observation_fit':rows,'online_nearest_training_state':online,'scope':'episode0 only; offline nearest-state comparison, no oracle fed to policy; montage top=decoded training RGB, bottom=actual online RGB'}
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
