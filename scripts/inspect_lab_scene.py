"""Render and audit the archive rig before collecting any data."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import cv2
ROOT=Path('/data/jaka-vla');ROBO=ROOT/'repos/RoboTwin';sys.path[:0]=[str(ROOT),str(ROBO)];os.chdir(ROBO)
from scripts.eval_policy_xpolicylab import load_task_args,class_decorator
from jaka_vla.review_video import ReviewVideo
from eval_v2 import robot_vector

out=ROOT/'results/lab_scene_20261111_v3';out.mkdir(parents=True,exist_ok=False)
cfg,_=load_task_args({'task_name':'handover_block','task_config':'zu7_lab_smoke','policy_name':'SmolVLA'})
cfg.update(save_data=False,need_plan=False,eval_mode=True,render_freq=0,eval_video_save_dir=None)
task=class_decorator('handover_block');task.setup_demo(now_ep_num=0,seed=0,**cfg)
review=ReviewVideo(task,out/'scene.mp4',25,'ARCHIVE RIG / THREE-FINGER PROXY',0)
obs=task.get_obs();frame=review.capture(obs);review.write(frame,0,'STATIC GEOMETRY CHECK',25);review.close()
cv2.imwrite(str(out/'scene.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
r=task.robot
report={'measured_state':robot_vector(r).tolist(),'left_tcp':r.get_left_tcp_pose(),'right_tcp':r.get_right_tcp_pose(),
        'left_base':list(r.left_entity.get_pose().p)+list(r.left_entity.get_pose().q),
        'right_base':list(r.right_entity.get_pose().p)+list(r.right_entity.get_pose().q),
        'contacts':[]}
arm_names={l.entity.global_id:a for a in ('left','right') for l in getattr(r,a+'_entity').get_links()}
for c in task.scene.get_contacts():
    impulse=sum(float(np.linalg.norm(p.impulse)) for p in c.points)
    if impulse>1e-5:report['contacts'].append({'pair':[arm_names.get(b.entity.global_id,'scene')+':'+b.entity.name for b in c.bodies],'impulse':impulse})
(out/'report.json').write_text(json.dumps(report,indent=2,default=lambda x:float(x))+'\n')
print(json.dumps(report,default=lambda x:float(x)),flush=True);task.close_env(clear_cache=True)
