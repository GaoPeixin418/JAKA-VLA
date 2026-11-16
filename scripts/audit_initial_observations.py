"""Verify same-scene target switching and record historical reload differences."""
from pathlib import Path
import hashlib,json
import numpy as np
import argparse
p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--historical-blue',type=Path);args=p.parse_args()
out=args.output
def array_hash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
checks=[]
for base in args.sources:
 name=base.name
 if not (base/'protocol.json').exists():continue
 groups={}
 for w in sorted(base.glob('worker_*.json')):
  worker=json.loads(w.read_text())
  if any(r['status']!='completed' for r in worker['rollouts']):
   raise ValueError('Connection mapping needs manual review after an operational error')
  for i,row in enumerate(worker['rollouts']):
   path=base/f"policy_observations_{worker['worker']}"/f'connection_{i:02d}_prediction_0000.npz'
   with np.load(path) as x:
    hashes={k:array_hash(x[k]) for k in ['state','image','left_wrist_image','right_wrist_image','noise']}
    entry={'target':row['target'],'hashes':hashes,'prompt':str(x['prompt']),'first_actions_sha256':array_hash(x['actions'])}
   groups.setdefault(row['layout'],[]).append(entry)
 for layout,entries in groups.items():
  if len(entries)==3:
   equal=all(e['hashes']==entries[0]['hashes'] for e in entries)
   assert len({e['prompt'] for e in entries})==3, 'Target instructions should differ'
   checks.append({'batch':name,'layout':layout,'identical_initial_state_rgb_noise':equal,'entries':entries})
comparison={}
new=args.sources[0]/'policy_observations_1/connection_00_prediction_0000.npz'
old=args.historical_blue
if new.exists() and old is not None and old.exists():
 with np.load(new) as a,np.load(old) as b:
  for key in ['state','image','left_wrist_image','right_wrist_image','noise','actions']:
   delta=np.abs(np.asarray(a[key],dtype=np.float64)-np.asarray(b[key],dtype=np.float64))
   comparison[key]={'max_abs_difference':float(delta.max()),'mean_abs_difference':float(delta.mean())}
report={'scope':'Offline evidence only. Identical initial images/state across target prompts checks target leakage; it does not prove semantic generalization.',
 'checked_layouts':len(checks),'initial_observation_matches':sum(c['identical_initial_state_rgb_noise'] for c in checks),
 'historical_development_1000_blue_first_prediction':comparison,
 'historical_reproduction_note':'2026-11-12 layout0 1/3, layout1000 2/3; 2026-11-16 layout0 1/3, layout1000 1/3. Same first observation and noise, but small first action differences across process reloads; cause not isolated. Historical trials not pooled into fresh heldout rate.',
 'layouts':checks}
out.mkdir(parents=True,exist_ok=True);(out/'initial_observation_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['layouts','scope']},ensure_ascii=False))
