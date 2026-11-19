"""Small bounded recovery batch from recorded nominal-policy deviations."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/data/jaka-vla');sys.path.insert(0,str(R))
from jaka_vla.contact_acceptance import lab_forbidden_contact
B=R/'results/lab_recovery_20261118';rows=[]
end=datetime.datetime.fromisoformat('2026-11-18T19:25:00+08:00').timestamp()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
for case in json.loads((B/'candidates.json').read_text()):
 if end-time.time()<240:break
 key=case['key'];probe=B/(key+'_probe');replay=B/(key+'_replay');row={**case,'probe':str(probe),'replay':str(replay),'accepted':False}
 if probe.exists():raise FileExistsError(probe)
 def run(cmd,name,seconds):
  with (B/f'{key}_{name}.log').open('x') as f:
   subprocess.run([sys.executable,'-u',*map(str,cmd)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=min(seconds,max(1,end-time.time())),check=True)
 try:
  run([R/'scripts/probe_zu7_short_grasp.py','--output',probe,'--seed',case['seed'],'--start-trace',case['source_trace'],'--start-step',case['step'],'--collect','--no-review-video','--rig-config',R/'configs/robots/dual_zu7_lab.json','--scene-spec',case['scene']],'collect',360)
  d=json.loads((probe/'result.json').read_text());row['expert_success']=d['grasp_success']
  assert d['grasp_success'] and not d['failure']
  assert not any(lab_forbidden_contact(c['pair']) for c in json.loads((probe/'contacts.json').read_text()))
  run([R/'scripts/eval_v2.py','--robot-id','dual_jaka_zu7_lab_archive','--rig-config',R/'configs/robots/dual_zu7_lab.json','--scene-spec',case['scene'],'--seed-start',case['seed'],'--episodes','1','--expert-hdf5',probe/'raw/data','--output',replay,'--instruction',d['instruction'],'--task-scope','left_grasp_lift_hold'],'replay',90)
  rr=json.loads((replay/'results.json').read_text())[0];bad=[c['pair'] for c in json.loads((replay/f"seed_{case['seed']}_contacts.json").read_text()) if lab_forbidden_contact(c['pair'])]
  row.update(replay_success=rr['grasp_success'],replay_failure=rr['failure_reason'],forbidden_contacts=bad,accepted=bool(rr['grasp_success'] and not rr['failure_reason'] and not bad))
 except Exception as e:row['error']=repr(e)
 rows.append(row);(B/'progress.json').write_text(json.dumps({'completed':False,'attempts':rows},indent=2)+'\n');print(json.dumps(row),flush=True)
(B/'progress.json').write_text(json.dumps({'completed':True,'attempts':rows,'accepted':sum(r['accepted'] for r in rows),'scope':'Recorded learner configuration reset then expert replanning; no executed policy prefix; layout 2020 no longer future holdout'},indent=2)+'\n')
