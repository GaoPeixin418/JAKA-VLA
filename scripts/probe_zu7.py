from pathlib import Path
import os,sys,json
import numpy as np
from PIL import Image
ROOT=Path('/data/jaka-vla');REPO=ROOT/'repos/RoboTwin'
sys.path.insert(0,str(REPO));os.chdir(REPO)
from scripts.eval_policy_xpolicylab import load_task_args,class_decorator
from envs.utils import ArmTag
cfg,_=load_task_args({'task_name':'handover_block','task_config':'zu7_smoke','policy_name':'SmolVLA'})
cfg.update(save_data=False,need_plan=True,eval_mode=False,render_freq=0)
task=class_decorator('handover_block');task.setup_demo(now_ep_num=0,seed=0,**cfg)
obs=task.get_obs();out=ROOT/'results/zu7_probe';out.mkdir(exist_ok=True)
Image.fromarray(obs['observation']['head_camera']['rgb']).save(out/'initial.png')
r=task.robot
print('qpos',r.left_entity.get_qpos(),r.right_entity.get_qpos(),flush=True)
print('left tool',r.get_left_ee_pose(),'root',r.left_entity.get_root_pose(),flush=True)
print('box',task.box.get_pose(),flush=True)
planner=r.left_planner
from curobo.types.robot import JointState
import torch
indices=[r.left_entity.get_active_joints().index(j) for j in r.left_arm_joints]
js=JointState.from_position(torch.tensor(r.left_entity.get_qpos()[indices],device='cuda').reshape(1,-1),joint_names=r.left_arm_joints_name)
print('start_valid',planner.motion_gen.check_start_state(js),flush=True)
print('joint_names',[j.get_name() for j in r.left_entity.get_active_joints()],flush=True)

for target in [r.left_ee.global_pose,task.box.get_pose()]:
 result=planner.plan_path(r.left_entity.get_qpos(),target,arms_tag='left')
 print('direct target',target,'result',result['status'],flush=True)
for i in range(4):
 print('contact',i,task.box.get_contact_point(i,'list'),flush=True)
 pose=task.get_grasp_pose(task.box,ArmTag('left'),contact_point_id=i,pre_dis=.07)
 print('pose',pose,flush=True)
print('done',flush=True)
task.close_env(clear_cache=True)
