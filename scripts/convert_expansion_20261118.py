"""Convert only the final accepted expansion dataset, bounded by today's shutdown."""
import datetime,json,os,subprocess,time
from pathlib import Path
R=Path('/data/jaka-vla');name='lab_expert_expansion_20261118';out=R/'results'/f'{name}_conversion';out.mkdir(exist_ok=False)
stop=datetime.datetime.fromisoformat('2026-11-18T19:27:00+08:00').timestamp()
report={'completed':False,'stages':[],'name':name,'scope':'separate accepted expansion, not yet mixed into training'}
def save():
 p=out/'progress.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(p)
save()
try:
 p=R/'results/expert_multilayout_20261118/acceptance_1913/progress.json'
 while time.time()<stop-120:
  if p.exists():
   acceptance=json.loads(p.read_text())
   if acceptance.get('completed') and acceptance.get('raw_dataset'):break
  time.sleep(5)
 else:raise TimeoutError('Accepted raw dataset not ready before conversion cutoff')
 raw=Path(acceptance['raw_dataset']);v3=R/'data/lerobot'/name;compatible=R/'data/openpi'/name
 report.update(accepted=acceptance['accepted'],frames=acceptance['data_frames'])
 env=dict(os.environ,PYTHONPATH=str(R)+':'+str(R/'repos/openpi/src')+':'+str(R/'repos/openpi/packages/openpi-client/src'),HF_LEROBOT_HOME=str(R/'data/openpi'),OPENPI_DATA_HOME='/data/pi05-cache/models',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cpu',WANDB_MODE='disabled',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
 stages=[('lerobot3',R/'envs/vla/bin/python',['scripts/convert_dataset_v2.py','--source',raw/'data','--output',v3,'--repo-id',f'local/{name}']),('openpi',Path('/data/pi05-env/bin/python'),['scripts/convert_zu7_for_openpi.py','--source',v3,'--output',compatible]),('statistics',Path('/data/pi05-env/bin/python'),['scripts/prepare_pi05_stats.py','--dataset',name]),('audit',Path('/data/pi05-env/bin/python'),['scripts/audit_pi05_data.py','--dataset',name,'--output',out/'input_audit.json'])]
 for stage,python,argv in stages:
  if time.time()>=stop:raise TimeoutError('Shutdown buffer reached')
  report['stage']=stage;save()
  with (out/f'{stage}.log').open('x') as f:subprocess.run([str(python),'-u',*map(str,argv)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,stop-time.time()),check=True)
  report['stages'].append(stage);save()
 report.update(completed=True,stage='ready',openpi=str(compatible))
except Exception as e:report.update(error=repr(e),stage='incomplete')
finally:save();print(json.dumps(report),flush=True)
