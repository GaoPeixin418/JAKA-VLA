"""Create immutable-batch file manifests and verify the checkpoint remains frozen."""
import argparse,hashlib,json
from datetime import datetime
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert (a.run/'execution_finished.json').exists(),'Batch still active'
 protocol=json.loads((a.run/'protocol.json').read_text());checkpoint=Path(protocol['arguments']['checkpoint'])
 expected=protocol['model_sha256'];actual={name:sha(checkpoint/name) for name in expected}
 assert actual==expected,'Checkpoint changed during evaluation'
 records=[]
 for path in sorted(a.run.rglob('*')):
  if path.is_file():records.append({'path':str(path.relative_to(a.run)),'bytes':path.stat().st_size,'sha256':sha(path)})
 workers=[json.loads(x.read_text()) for x in a.run.glob('worker_*.json')]
 report={'created_at':datetime.now().astimezone().isoformat(),'run':str(a.run),'checkpoint_frozen_sha256_verified':True,'model_sha256':actual,
 'planned':len(protocol['cases']),'attempted':sum(len(w['rollouts']) for w in workers),'all_workers_completed':all(w['completed'] for w in workers),
 'file_count':len(records),'bytes':sum(r['bytes'] for r in records),'files':records}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['files','model_sha256']}))
if __name__=='__main__':main()
