"""Decode every completed source video and retain SHA256 verification records."""
import argparse,concurrent.futures,hashlib,json,subprocess
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 previous=json.loads(a.output.read_text()).get('files',[]) if a.output.exists() else []
 cache={x['path']:x for x in previous};pending=[];verified=[]
 for run in a.runs:
  for worker in run.glob('worker_*.json'):
   d=json.loads(worker.read_text())
   for row in d['rollouts']:
    if row['status']!='completed':continue
    for video in sorted((run/row['key']).glob('*.mp4')):
     digest=sha(video);old=cache.get(str(video))
     if old and old['sha256']==digest and old['full_decode_passed']:
      verified.append(old)
     else:pending.append((video,digest))
 def check(item):
  video,digest=item
  decoded=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=90)
  if decoded.returncode or decoded.stderr.strip():raise RuntimeError(f'{video}: {decoded.stderr}')
  meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate,codec_name:format=duration','-of','json',str(video)],text=True))
  return {'path':str(video),'bytes':video.stat().st_size,'sha256':digest,'full_decode_passed':True,'metadata':meta}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for row in pool.map(check,pending):verified.append(row)
 report={'verified_count':len(verified),'newly_decoded':len(pending),'files':sorted(verified,key=lambda r:r['path'])}
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
 print(json.dumps({'verified_videos':len(verified),'newly_decoded':len(pending)}))
if __name__=='__main__':main()
