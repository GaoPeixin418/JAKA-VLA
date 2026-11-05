from pathlib import Path
import os,sys,json,argparse
import numpy as np
ROOT=Path('/data/jaka-vla');REPO=ROOT/'repos/RoboTwin'
sys.path.insert(0,str(REPO));os.chdir(REPO)
from scripts.eval_policy_xpolicylab import load_task_args,class_decorator
from envs.utils.images_to_video import images_to_video
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0);args=p.parse_args()
cfg,_=load_task_args({'task_name':'handover_block','task_config':'zu7_smoke','policy_name':'SmolVLA'})
cfg.update(save_data=False,need_plan=True,eval_mode=False,render_freq=0)
task=class_decorator('handover_block');task.setup_demo(now_ep_num=args.seed,seed=args.seed,**cfg)
frames=[];rows=[];step=task._jaka_step;move=task.move
arm_ids={link.entity.global_id:arm for arm in ('left','right') for link in getattr(task.robot,f'{arm}_entity').get_links()}
cross_contacts=[]
out=args.output;out.mkdir(exist_ok=False)
(out/'rig.json').write_text((ROOT/'configs/robots/dual_zu7.json').read_text())
def record_step():
 step()
 for contact in task.scene.get_contacts():
  arms={arm_ids.get(body.entity.global_id) for body in contact.bodies}
  if arms=={'left','right'}:
   impulse=sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
   if impulse>1e-5:cross_contacts.append({'time_s':task._jaka_sim_steps*.004,'impulse_ns':impulse,'bodies':[arm_ids[b.entity.global_id]+':'+b.entity.name for b in contact.bodies]})
 if task._jaka_sim_steps%10==0:frames.append(task.get_obs()['observation']['head_camera']['rgb'])
task._jaka_step=record_step
for name in ['left_planner','right_planner']:
 planner=getattr(task.robot,name);original=planner.plan_path
 def wrapper(*args,_orig=original,_name=name,**kwargs):
  result=_orig(*args,**kwargs)
  print(_name,'plan',result['status'], 'target',args[1] if len(args)>1 else None,flush=True)
  return result
 planner.plan_path=wrapper

def record_move(*args,**kwargs):
 result=move(*args,**kwargs)
 row={'stage':len(rows),'plan_success':bool(task.plan_success),'box':task.box.get_pose().p.tolist(),
 'actual_link_quat':task.robot.left_ee.child_link.get_pose().q.tolist(),'joint_quat':task.robot.left_ee.global_pose.q.tolist(),'tcp_left':task.robot.get_left_tcp_pose(),'tcp_right':task.robot.get_right_tcp_pose(),
 'gripper':[task.robot.get_left_gripper_val(),task.robot.get_right_gripper_val()]}
 rows.append(row);print(json.dumps(row),flush=True)
 return result
task.move=record_move
try:task.play_once()
finally:
 (out/'stages.json').write_text(json.dumps(rows,indent=2))
 (out/'interarm_contacts.json').write_text(json.dumps(cross_contacts,indent=2))
 if frames:images_to_video(np.stack(frames),str(out/f'seed_{args.seed}.mp4'),fps=25)
 (out/'result.json').write_text(json.dumps({'seed':args.seed,'success':bool(task.check_success()),'frames':len(frames),'interarm_contact_samples':len(cross_contacts)},indent=2))
 print('success',task.check_success(),'frames',len(frames),flush=True)
 task.close_env(clear_cache=True)
