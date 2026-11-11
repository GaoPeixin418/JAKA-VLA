"""Audit start-pose collision proxies without executing a trajectory."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import sapien
import yaml
import trimesh
import transforms3d as t3d
import argparse

ROOT = Path('/data/jaka-vla')
ROBO = ROOT/'repos/RoboTwin'
sys.path[:0] = [str(ROOT), str(ROBO)]
os.chdir(ROBO)
from scripts.eval_policy_xpolicylab import load_task_args, class_decorator

parser=argparse.ArgumentParser();parser.add_argument('--planner',action='store_true');parser.add_argument('--ik-report',type=Path);args=parser.parse_args()
cfg, _ = load_task_args({'task_name':'handover_block','task_config':'zu7_lab_smoke','policy_name':'SmolVLA'})
cfg.update(save_data=False,need_plan=args.planner,eval_mode=True,render_freq=0,eval_video_save_dir=None)
task=class_decorator('handover_block')
task.setup_demo(now_ep_num=0,seed=0,**cfg)
report={}
for side in ('left','right'):
    asset=ROBO/f'assets/embodiments/jaka-zu7-lab-{side}'
    spheres=yaml.safe_load((asset/'collision.yml').read_text())['collision_spheres']
    config=yaml.safe_load((asset/'curobo.yml').read_text())
    table=config['planner']['world_table']
    pose=sapien.Pose(table['position'],table['quaternion'])
    rows=[]
    for link in getattr(task.robot,side+'_entity').get_links():
        name=link.get_name()
        for index,s in enumerate(spheres.get(name,[])):
            center=(pose.inv()*link.get_pose()*sapien.Pose(s['center'])).p
            q=np.abs(center)-np.array(table['dims'])/2
            signed=float(np.linalg.norm(np.maximum(q,0))+min(max(q),0))
            clearance=signed-s['radius']-config['robot_cfg']['kinematics']['collision_sphere_buffer']
            if clearance<.02:
                rows.append({'link':name,'index':index,'clearance_m':clearance,'table_local_center':center.tolist(),'radius_m':s['radius']})
    report[side]={'table':table,'closest_spheres':sorted(rows,key=lambda r:r['clearance_m'])[:30]}
    if args.planner:
        import torch
        from curobo.types.robot import JointState
        from curobo.geom.sdf.world import CollisionQueryBuffer
        planner=getattr(task.robot,side+'_planner');planner.refresh_other_arm_world()
        mg=planner.motion_gen
        entity=getattr(task.robot,side+'_entity');active=entity.get_active_joints()
        q=[float(entity.get_qpos()[active.index(j)]) for j in getattr(task.robot,side+'_arm_joints')]
        js=JointState.from_position(mg.tensor_args.to_device([q]),joint_names=planner.active_joints_name)
        state=mg.compute_kinematics(js)
        spheres_tensor=state.robot_spheres.reshape(1,1,-1,4).contiguous()
        checker=mg.world_coll_checker
        buffer=CollisionQueryBuffer()
        buffer.update_buffer_shape(spheres_tensor.shape,mg.tensor_args,checker.collision_types)
        names=[n for n in checker.get_obstacle_names() if n]
        kin=mg.kinematics.kinematics_config
        inverse={i:n for n,i in kin.link_name_to_idx_map.items()}
        indices=kin.link_sphere_idx_map.cpu().tolist()
        found=[]
        try:
            for name in names:checker.enable_obstacle(name,False)
            for name in names:
                checker.enable_obstacle(name,True)
                distances=checker.get_sphere_distance(spheres_tensor,buffer,mg.tensor_args.to_device([1.]),mg.tensor_args.to_device([0.])).detach().cpu().numpy().reshape(-1)
                bad=np.flatnonzero(distances>0)
                if len(bad):found.append({'obstacle':name,'collisions':[{'sphere':int(i),'link':inverse[indices[i]],'penetration_m':float(distances[i])} for i in bad]})
                checker.enable_obstacle(name,False)
        finally:
            for name in names:checker.enable_obstacle(name,True)
        report[side]['planner_collisions']=found
        mg.rollout_fn.robot_self_collision_constraint.enable_cost()
        valid,status=mg.check_start_state(js)
        report[side]['home_valid']={'valid':valid,'status':str(status)}
        mg.rollout_fn.robot_self_collision_constraint.enable_cost()
        if side=='left' and args.ik_report:
            ik=json.loads(args.ik_report.read_text());checked=[]
            for candidate in ik['candidates']:
                for solution in candidate['solutions']:
                    state=JointState.from_position(mg.tensor_args.to_device([solution['q']]),joint_names=planner.active_joints_name)
                    mg.rollout_fn.robot_self_collision_constraint.enable_cost()
                    computed=mg.compute_kinematics(state)
                    self_cost=float(mg.rollout_fn.robot_self_collision_constraint(computed.robot_spheres.reshape(1,1,-1,4)).item())
                    valid,status=mg.check_start_state(state)
                    checked.append({'yaw':candidate['yaw'],'q':solution['q'],'valid':valid,'status':str(status),'self_cost':self_cost})
                    if len(checked)==1:
                        np.savez(ROOT/'results/lab_actual_curobo_spheres.npz',spheres=computed.robot_spheres.detach().cpu().numpy(),q=solution['q'],
                                 self_matrix=mg.rollout_fn.robot_self_collision_constraint.self_collision_kin_config.collision_matrix.cpu().numpy(),
                                 self_offset=mg.rollout_fn.robot_self_collision_constraint.self_collision_kin_config.offset.cpu().numpy())
            report[side]['ik_candidates']=checked
print(json.dumps(report,indent=2),flush=True)
task.close_env(clear_cache=True)
