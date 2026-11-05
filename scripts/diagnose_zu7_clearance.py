"""Read-only FK/IK clearance diagnosis; no collision-disabled trajectories executed."""
import json,os,sys
from pathlib import Path
import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation
import yaml
ROOT=Path('/data/jaka-vla');REPO=ROOT/'repos/RoboTwin'
sys.path.insert(0,str(REPO));os.chdir(REPO)
from scripts.eval_policy_xpolicylab import load_task_args,class_decorator
cfg,_=load_task_args({'task_name':'handover_block','task_config':'zu7_smoke','policy_name':'SmolVLA'})
cfg.update(save_data=False,need_plan=False,eval_mode=False,render_freq=0)
task=class_decorator('handover_block');task.setup_demo(now_ep_num=0,seed=0,**cfg)
r=task.robot;entity=r.right_entity;joints=entity.get_active_joints()
indices=[joints.index(j) for j in r.right_arm_joints];original=entity.get_qpos().copy()
target_p=np.array([.185424,.0459838,.77611]);target_r=Rotation.from_quat([0,0,.70710678,.70710678])
def residual(q):
    values=original.copy();values[indices]=q;entity.set_qpos(values)
    pose=r.right_ee.child_link.get_pose();rot=Rotation.from_quat(np.roll(pose.q,-1))
    return np.r_[(pose.p-target_p)*5,(target_r.inv()*rot).as_rotvec()]
limits=np.array([j.get_limits()[0] for j in r.right_arm_joints])
def jacobian(q):
    delta=np.eye(len(q))*.001
    return np.stack([(residual(q+d)-residual(q-d))/.002 for d in delta],axis=1)
result=scipy.optimize.least_squares(residual,original[indices],jac=jacobian,bounds=(limits[:,0],limits[:,1]),max_nfev=300)
if np.linalg.norm(result.fun)>.01:raise RuntimeError(f'IK did not converge: {result.fun}; no clearance conclusion is valid')
residual(result.x)
print('residual',result.fun,'q',result.x,flush=True)
spheres=yaml.safe_load((REPO/'assets/embodiments/jaka-zu7-proxy/collision.yml').read_text())['collision_spheres']
report=[]
for link in entity.get_links():
    name=link.get_name()
    if name not in spheres:continue
    pose=link.get_pose();rot=Rotation.from_quat(np.roll(pose.q,-1))
    for sphere in spheres[name]:
        p=pose.p+rot.apply(sphere['center']);radius=sphere['radius']+.002
        if abs(p[0])<.6+radius and abs(p[1])<.35+radius:
            report.append({'link':name,'sphere_bottom_z':float(p[2]-radius),'center':p.tolist(),'radius':radius})
print(json.dumps(sorted(report,key=lambda x:x['sphere_bottom_z'])[:15],indent=2),flush=True)
entity.set_qpos(original);task.close_env(clear_cache=True)
