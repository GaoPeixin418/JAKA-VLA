"""Build an explicitly illustrative montage, keeping full individual episodes."""
import argparse,json,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);a=p.parse_args()
s=json.loads((a.review/'summary.json').read_text())
rows=[r for r in s['rows'] if r['split'].startswith('heldout') and r['review_video']]
selected=[]
for color in ['blue','red','yellow']:
 for success in [True,False]:
  candidates=[r for r in rows if r['color']==color and r['qualified_success']==success]
  if candidates:selected.append(candidates[0])
if len(selected)<4:raise ValueError('Need success and failure illustrations for multiple colors')
a.review.mkdir(parents=True,exist_ok=True);temp=a.review/'demo_parts';temp.mkdir(exist_ok=True)
segments=[]
for i,r in enumerate(selected):
 original=a.review/r['video']
 if not original.exists():raise FileNotFoundError(original)
 verdict='QUALIFIED GRASP' if r['qualified_success'] else 'FAILURE - '+str(r['failure'])
 label=f"ILLUSTRATIVE CASE {i+1}/{len(selected)} | Layout {r['layout']} | {r['color'].upper()} | Slot {r['target']} | {verdict}"
 txt=temp/f'label_{i}.txt';txt.write_text(label)
 output=temp/f'part_{i}.mp4'
 vf=f"drawbox=x=0:y=40:w=iw:h=38:color=black@0.8:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile={txt}:fontsize=19:fontcolor=white:x=12:y=49,setsar=1"
 subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-i',str(original),'-vf',vf,'-r','25','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p','-an',str(output)],check=True,timeout=120)
 segments.append(output)
(temp/'concat.txt').write_text(''.join("file '"+str(p.resolve())+"'\n" for p in segments))
final=a.review/'demo_success_and_failure.mp4'
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(temp/'concat.txt'),'-c','copy','-movflags','+faststart',str(final)],check=True,timeout=120)
subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(final),'-f','null','-'],check=True,timeout=120)
(a.review/'demo_selection.json').write_text(json.dumps({'scope':'Illustrative full-episode montage, not a representative sample or success-rate estimate. First available HD success/failure for each color among heldout cases; full videos retained in viewer.','selection':selected},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'file':str(final),'cases':len(selected),'bytes':final.stat().st_size}))
