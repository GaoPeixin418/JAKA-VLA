"""Short Cartesian insertion for the archive expert; checked before execution."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import torch
from curobo.types.math import Pose
from jaka_vla.lab_retiming import retime_and_validate


def plan_cartesian(planner, start, target_world):
    xml=ET.parse(Path(planner.yml_path).parent/'robot.urdf').getroot()
    names=planner.active_joints_name
    by_child={j.find('child').get('link'):j for j in xml.findall('joint')}
    chain=[];child='tool'
    while child in by_child:
        joint=by_child[child];chain.insert(0,joint);child=joint.find('parent').get('link')
    operations=[];lower=[];upper=[]
    for name in names:
        limit=xml.find(f"joint[@name='{name}']/limit")
        lower.append(float(limit.get('lower')));upper.append(float(limit.get('upper')))
    for joint in chain:
        origin=joint.find('origin');m=np.eye(4)
        if origin is not None:
            m[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            m[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        index=names.index(joint.get('name')) if joint.get('name') in names else None
        axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ') if index is not None else None
        operations.append((m,index,axis))
    def fk(q):
        m=np.eye(4)
        for origin,index,axis in operations:
            m=m@origin
            if index is not None:
                rot=np.eye(4);rot[:3,:3]=Rotation.from_rotvec(axis*q[index]).as_matrix();m=m@rot
        return m
    target=planner.robot_origion_pose.inv()*target_world
    goal_p=np.asarray(target.p,dtype=float)
    goal_r=Rotation.from_quat(np.roll(target.q,-1))
    begin=fk(start);initial_r=Rotation.from_matrix(begin[:3,:3])
    if (initial_r.inv()*goal_r).magnitude()>.05:raise ValueError('Cartesian insertion start orientation mismatch')
    length=np.linalg.norm(goal_p-begin[:3,3]);count=max(2,int(np.ceil(length/.005))+1)
    rotations=Slerp([0,1],Rotation.from_matrix(np.stack([begin[:3,:3],goal_r.as_matrix()])))
    q=np.asarray(start,dtype=float).copy();points=[q.copy()];worst=0.
    for fraction in np.linspace(0,1,count)[1:]:
        position=begin[:3,3]*(1-fraction)+goal_p*fraction;rotation=rotations(fraction)
        def error(value):
            pose=fk(value)
            return np.r_[3*(pose[:3,3]-position),(rotation.inv()*Rotation.from_matrix(pose[:3,:3])).as_rotvec()]
        result=least_squares(error,q,bounds=(lower,upper),max_nfev=60,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        residual=float(np.linalg.norm(result.fun));worst=max(worst,residual)
        if residual>1e-4 or np.max(np.abs(result.x-q))>.15:raise ValueError(f'Cartesian IK at {fraction:.3f}: residual={residual:.6g}, max_step={np.max(np.abs(result.x-q)):.5f}, bounds={result.active_mask.tolist()}')
        q=result.x;points.append(q.copy())
    mg=planner.motion_gen
    geometric=SimpleNamespace(interpolated_plan=SimpleNamespace(position=torch.as_tensor(np.array(points),device='cuda')))
    result=retime_and_validate(planner,geometric,goal=Pose.from_list([*goal_p,*target.q]),
                               start=mg.tensor_args.to_device(np.asarray(start)),hold_orientation=True)
    result['cartesian']={'waypoints':count,'max_weighted_ik_residual':worst,'translation_m':float(length)}
    return result


def plan_grasp_pair(planner, initial, pre_world, grasp_world, keepout):
    """Choose the grasp IK branch first, then connect home to its vertical approach."""
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    mg=planner.motion_gen
    planner.refresh_other_arm_world()
    target=planner.robot_origion_pose.inv()*grasp_world
    goal=Pose.from_list([*target.p,*target.q])
    solved=mg.ik_solver.solve_single(goal,return_seeds=8)
    candidates=solved.solution.detach().cpu().numpy().reshape(-1,6)
    success=solved.success.detach().cpu().numpy().reshape(-1)
    candidates=candidates[success]
    candidates=sorted(candidates,key=lambda q:np.linalg.norm(q-initial))
    reasons=[];seen=[]
    for q in candidates:
        if any(np.linalg.norm(q-prev)<.02 for prev in seen):continue
        seen.append(q)
        try:
            reverse=plan_cartesian(planner,q,pre_world)
            qpre=reverse['position'][-1]
            planner.project_world['cuboid']['target_keepout']=keepout
            try:
                planner.refresh_other_arm_world()
                start_js=JointState.from_position(mg.tensor_args.to_device(np.asarray(initial)[None]),joint_names=planner.active_joints_name)
                goal_js=JointState.from_position(mg.tensor_args.to_device(qpre[None]),joint_names=planner.active_joints_name)
                result=mg.plan_single_js(start_js,goal_js,MotionGenPlanConfig(max_attempts=5,enable_graph=True,timeout=15.,time_dilation_factor=.5))
                if not bool(result.success.item()):raise RuntimeError(f'Approach joint planning: {result.status}')
                pre=planner.robot_origion_pose.inv()*pre_world
                geometry=SimpleNamespace(interpolated_plan=result.optimized_plan)
                first=retime_and_validate(planner,geometry,goal=Pose.from_list([*pre.p,*pre.q]),start=start_js.position)
            finally:
                planner.project_world['cuboid'].pop('target_keepout',None);planner.refresh_other_arm_world()
            second={**reverse,'position':reverse['position'][::-1].copy(),'velocity':-reverse['velocity'][::-1].copy()}
            second['cartesian']={**reverse['cartesian'],'computed_from_grasp_backwards':True}
            return first,second
        except (ValueError,RuntimeError) as error:
            reasons.append(str(error))
        if len(seen)>=4:break
    raise RuntimeError(f'No grasp branch with continuous descent: valid_ik={len(candidates)}, rejected={reasons}')
