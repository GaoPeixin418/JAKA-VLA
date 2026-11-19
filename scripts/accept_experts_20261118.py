"""Verify collected RGB/state/action episodes, replay with the fixed servo, then assemble."""
import concurrent.futures, datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path('/data/jaka-vla'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/RoboTwin')]
from jaka_vla.contact_acceptance import lab_forbidden_contact
from XPolicyLab.utils.data_loader import load_xspark_v1
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state
B=ROOT/'results/expert_multilayout_20261118'; OUT=B/'acceptance_1913'
OUT.mkdir(exist_ok=True); (OUT/'logs').mkdir(exist_ok=True); (OUT/'replays').mkdir(exist_ok=True)
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def verify(probe):
 saved=OUT/f'{probe.name}.json'
 if saved.exists():
  old=json.loads(saved.read_text())
  replay=Path(old.get('replay',''))
  results=replay/'results.json'
  if old.get('stage')=='checked' and old.get('data_check_passed') and results.is_file():
   try:
    rr=json.loads(results.read_text())[0]
    contacts=replay/f"seed_{old['seed']}_contacts.json"
    bad=[c['pair'] for c in json.loads(contacts.read_text()) if lab_forbidden_contact(c['pair'])]
    old.update(replay_grasp_success=rr['grasp_success'],replay_failure=rr['failure_reason'],forbidden_contacts=bad,accepted=bool(rr['grasp_success'] and not rr['failure_reason'] and not bad),stage='checked')
    save(saved,old)
    return old
   except Exception:
    pass
 row={'probe':str(probe),'accepted':False,'stage':'checking'}
 report=json.loads((probe/'result.json').read_text()); seed=report['seed']; spec=report['scene_spec']
 row.update(layout=spec['layout_id'],target=spec['target_index'],seed=seed,color=spec['objects'][spec['target_index']]['color'])
 try:
  assert report['grasp_success'] and report['failure'] is None
  assert not any(lab_forbidden_contact(c['pair']) for c in json.loads((probe/'contacts.json').read_text()))
  h=probe/'raw/data/episode_0000000.hdf5'
  raw=load_xspark_v1(str(h),decode_images=True)
  dims=get_robot_action_dim_info('dual_jaka_zu7_lab_archive')
  state=pack_robot_state(raw,'joint',dims,source_type='dataset')
  action=pack_robot_state(raw,'joint',dims,source_type='dataset',state_type='action')
  t=np.asarray(raw['observation_timestamps']);a=np.asarray(raw['action_timestamps'])
  assert state.shape==action.shape==(len(t),14) and np.isfinite(state).all() and np.isfinite(action).all()
  np.testing.assert_allclose(np.diff(t),.04,atol=1e-6);np.testing.assert_allclose(a-t,.04,atol=1e-6)
  for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:
   rgb=raw['vision'][cam]['colors']; assert rgb.dtype==np.uint8 and rgb.shape==(len(t),240,320,3),(cam,rgb.shape)
  row.update(frames=len(t),rgb_frames=3*len(t),hdf5_sha256=hashlib.sha256(h.read_bytes()).hexdigest(),data_check_passed=True)
  del raw,state,action

  replay=OUT/'replays'/probe.name;row['replay']=str(replay)
  scene=B/'scenes'/f'{probe.name}.json'
  cmd=[sys.executable,'-u',str(ROOT/'scripts/eval_v2.py'),'--robot-id','dual_jaka_zu7_lab_archive','--rig-config',str(ROOT/'configs/robots/dual_zu7_lab.json'),'--scene-spec',str(scene),'--seed-start',str(seed),'--episodes','1','--expert-hdf5',str(probe/'raw/data'),'--output',str(replay),'--instruction',spec['instruction'],'--task-scope','left_grasp_lift_hold']
  with (OUT/'logs'/f'{probe.name}.log').open('x') as f:
   result=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  assert result.returncode==0,('replay_exit',result.returncode)
  rr=json.loads((replay/'results.json').read_text())[0]
  bad=[c['pair'] for c in json.loads((replay/f'seed_{seed}_contacts.json').read_text()) if lab_forbidden_contact(c['pair'])]
  row.update(replay_grasp_success=rr['grasp_success'],replay_failure=rr['failure_reason'],forbidden_contacts=bad,accepted=bool(rr['grasp_success'] and not rr['failure_reason'] and not bad),stage='checked')
 except Exception as e:row.update(stage='rejected',error=repr(e))
 save(OUT/f'{probe.name}.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)
 return row
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for row in pool.map(verify,sorted(p.parent for p in (B/'episodes').glob('*/result.json'))):
  rows.append(row);save(OUT/'progress.json',{'completed':False,'attempts':rows,'accepted':sum(r['accepted'] for r in rows)})
accepted=[r for r in rows if r['accepted']]
summary={'completed':True,'attempts':rows,'accepted':len(accepted),'data_frames':sum(r.get('frames',0) for r in accepted),'scope':'Expert RGB/data + fixed servo acceptance only, not learned performance','split_assignment':{'train':list(range(4000,4010)),'note':'Collection accidentally labelled split=heldout. These 4000-series scenes were created expressly for expert training on 2026-11-18 and never used as a model holdout. Original source files retained.'}}
if accepted:
 dest=ROOT/'data/raw/lab_expert_expansion_20261118'
 cmd=[sys.executable,str(ROOT/'scripts/assemble_grasp_curriculum.py'),'--probes',*[r['probe'] for r in accepted],'--replays',*[r['replay'] for r in accepted],'--output',str(dest)]
 with (OUT/'logs/assemble.log').open('x') as f:subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
 summary['raw_dataset']=str(dest)
 # Correct the dataset's intended training split without modifying original evidence.
 p=dest/'robot_identity.json';identity=json.loads(p.read_text())
 for spec in identity['scenes_by_episode'].values():spec['split']='train'
 identity['split_assignment_provenance']=summary['split_assignment'];save(p,identity)
 save(dest/'split_assignment.json',summary['split_assignment'])
save(OUT/'progress.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k!='attempts'},ensure_ascii=False),flush=True)
