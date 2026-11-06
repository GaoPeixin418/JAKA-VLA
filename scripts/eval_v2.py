"""Robot-parametric, instrumented evaluation with bounded command dynamics."""
import argparse,hashlib,json,os,sys,time
from multiprocessing.connection import Client
from pathlib import Path
import h5py
import numpy as np
import sapien
ROOT=Path('/data/jaka-vla');REPO=ROOT/'repos/RoboTwin'
sys.path[:0]=[str(ROOT),str(REPO)];os.chdir(REPO)
from scripts.eval_policy_xpolicylab import load_task_args,class_decorator
from envs.utils.images_to_video import images_to_video
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from jaka_vla.control import ActionLayout,BoundedJointServo,ChunkContinuity
from jaka_vla.grasp_monitor import EmptyGraspMonitor
from jaka_vla.identity import physical_rig_hash
from jaka_vla.review_video import ReviewVideo
from jaka_vla.initial_conditions import apply_left_offset, sample_left_offset


def robot_vector(robot,velocity=False):
    parts=[]
    for arm in ('left','right'):
        entity=getattr(robot,f'{arm}_entity');active=entity.get_active_joints()
        values=entity.get_qvel() if velocity else entity.get_qpos()
        parts.extend([values[active.index(j)] for j in getattr(robot,f'{arm}_arm_joints')])
        joint,multiplier,offset=getattr(robot,f'{arm}_gripper')[0]
        low,high=getattr(robot,f'{arm}_gripper_scale')
        value=values[active.index(joint)]/multiplier
        parts.append(value/(high-low) if velocity else np.clip((value-offset/multiplier-low)/(high-low),0,1))
    return np.array(parts)


def limits_for(robot):
    rows=[]
    for arm in ('left','right'):
        rows.extend([j.get_limits()[0] for j in getattr(robot,f'{arm}_arm_joints')]);rows.append([0,1])
    return np.asarray(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--robot-id',default='dual_jaka_zu7');p.add_argument('--task-config',default='zu7_smoke')
    p.add_argument('--k',type=int,default=8);p.add_argument('--seed-start',type=int,default=1000);p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--max-steps',type=int,default=1000);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--socket',type=Path,default=ROOT/'run/policy-v2.sock');p.add_argument('--expert-hdf5',type=Path)
    p.add_argument('--blend-steps',type=int,default=4)
    p.add_argument('--review-video',action='store_true',help='Add an HD overview, overhead and close-up recording.')
    p.add_argument('--instruction', default='Use the left arm to grab the red block, hand it to the right arm, and place it on the blue pad.')
    p.add_argument('--task-scope', choices=['full_handover','left_grasp_lift_hold'], default='full_handover')
    p.add_argument('--initial-offset',type=float,default=0.)
    args=p.parse_args()
    args.output=ROOT/args.output
    if args.expert_hdf5:args.expert_hdf5=ROOT/args.expert_hdf5
    if not 1<=args.k<=50:raise ValueError('Invalid K')
    if args.expert_hdf5 and args.expert_hdf5.is_file() and args.episodes!=1:
        raise ValueError('A single expert trajectory can only be replayed in its own episode; pass the data directory for multiple seeds')
    args.output.mkdir(parents=True,exist_ok=False)
    dims=get_robot_action_dim_info(args.robot_id);layout=ActionLayout(tuple(dims['arm_dim']),tuple(dims['ee_dim']))
    rig=json.loads((ROOT/'configs/robots/dual_zu7.json').read_text())
    urdf_sha=hashlib.sha256((REPO/'assets/embodiments/jaka-zu7-proxy/robot.urdf').read_bytes()).hexdigest()
    if args.robot_id!=rig['robot_id']:raise ValueError('Rig identity mismatch')
    cfg,_=load_task_args({'task_name':'handover_block','task_config':args.task_config,'policy_name':'SmolVLA'})
    cfg.update(save_data=False,need_plan=False,eval_mode=True,render_freq=0,eval_video_save_dir=None)
    task=class_decorator('handover_block')
    conn=None if args.expert_hdf5 else Client(str(args.socket),family='AF_UNIX')
    reports=[];review=None
    try:
        for seed in range(args.seed_start,args.seed_start+args.episodes):
            task.setup_demo(now_ep_num=seed,seed=seed,**cfg)
            offset=sample_left_offset(seed,args.initial_offset)
            if args.expert_hdf5:
                expert_root=args.expert_hdf5.parent if args.expert_hdf5.is_dir() else args.expert_hdf5.parent.parent
                source_metadata=json.loads((expert_root/'robot_identity.json').read_text())
                condition=source_metadata.get('initial_conditions_by_seed',{}).get(str(seed),source_metadata.get('initial_condition'))
                if condition is not None:offset=np.asarray(condition['left_joint_offset_rad'])
            initial_condition=apply_left_offset(task,offset)
            dt=task.scene.get_timestep();substeps=round(1/rig['control_hz']/dt)
            assert abs(substeps*dt-1/rig['control_hz'])<1e-7
            r=task.robot;measured=robot_vector(r)
            arm_links={link.entity.global_id:arm for arm in ('left','right') for link in getattr(r,f'{arm}_entity').get_links()}
            contacts={}
            servo=BoundedJointServo(measured,limits_for(r),layout,dt=dt,vmax=rig['joint_velocity_limit_rad_s'],
                amax=rig['joint_acceleration_limit_rad_s2'],gripper_rate=rig['gripper_rate_per_s'],omega=rig['tracking_frequency_rad_s'])
            continuity=ChunkContinuity(layout,args.blend_steps)
            empty_monitor=EmptyGraspMonitor(layout.gripper_indices,dt=1/rig['control_hz'])
            if conn:
                conn.send({'op':'reset','seed':seed,'robot_id':args.robot_id,'urdf_sha256':urdf_sha,'rig':rig});mode=conn.recv()
            else:mode={'noise_mode':'expert_replay'}
            expert=None
            if args.expert_hdf5:
                source=args.expert_hdf5
                if source.is_dir():
                    seeds=[int(x) for x in (source.parent/'seed.txt').read_text().split()]
                    source=source/f'episode_{seeds.index(seed):07d}.hdf5'
                expert_identity=json.loads((source.parent.parent/'robot_identity.json').read_text())
                if physical_rig_hash(expert_identity['rig'])!=physical_rig_hash(rig):raise ValueError('Expert physical rig/task differs from current scene')
                with h5py.File(source) as f:
                    if f.attrs.get('robot_id')!=args.robot_id or f.attrs.get('rig_urdf_sha256')!=urdf_sha:
                        raise ValueError('Expert replay robot/rig mismatch')
                    expert=np.concatenate([f['action'][key][:] for key in ['left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states']],axis=1)
            frames=[];ticks=[];control=[];chunks=[];calls=[];normalization_checks=[];success=False;failure_reason=None;stable_goal_steps=0;tracking_error_time=0.
            initial_height=float(task.box.get_pose().p[2]);max_height=initial_height
            object_body=task.box.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            initial_ticks=task._jaka_sim_steps;start=time.perf_counter()
            count=len(expert)+25 if expert is not None else args.max_steps
            grasp_ticks=0;longest_grasp_ticks=0;grasp_success_time=None;grasp_evidence=[]
            if args.review_video:
                review=ReviewVideo(task,args.output/f'seed_{seed}_multiview.mp4',rig['control_hz'],
                    'EXPERT REPLAY' if expert is not None else 'LEARNED SmolVLA',seed)
            for step in range(count):
                obs=task.get_obs();frames.append(obs['observation']['head_camera']['rgb'])
                if review:
                    frame=review.capture(obs)
                    now=(task._jaka_sim_steps-initial_ticks)*dt
                    if step==0:review.write(frame,now,'INTRO / frozen initial frame; simulation has not advanced',round(review.intro_s*review.fps))
                    status='RUNNING / '+args.task_scope
                    if grasp_success_time is not None:status='LEFT GRASP verified / '+args.task_scope
                    review.write(frame,now,status)
                if expert is not None:target=expert[min(step,len(expert)-1)]
                else:
                    if step%args.k==0:
                        conn.send({'op':'predict','observation':{'state':obs['joint_measured']['vector'].astype(np.float32),
                            'images':[obs['observation'][c]['rgb'] for c in ['head_camera','left_camera','right_camera']],
                            'instruction':args.instruction}})
                        reply=conn.recv();calls.append(reply['inference_seconds'])
                        if 'normalized_state_abs_max' in reply:normalization_checks.append({'step':step,'normalized_state_abs_max':reply['normalized_state_abs_max']})
                        raw,blended,n=continuity.replace(reply['actions']);chunks.append((step,raw,blended))
                    target=continuity.pop()
                raw,clipped=servo.set_target(target)
                control.append(np.r_[step,robot_vector(r),robot_vector(r,True),raw,clipped,task.box.get_pose().p,
                    r.get_left_tcp_pose(),r.get_right_tcp_pose()])
                for tick in range(substeps):
                    position,velocity=servo.step()
                    for arm,sl,gi in zip(('left','right'),layout.arm_slices,layout.gripper_indices):
                        r.set_arm_joints(position[sl],velocity[sl],arm);r.set_gripper(position[gi],arm,gripper_eps=0)
                    task._jaka_step()
                    measured_tick=robot_vector(r)
                    ticks.append(np.r_[(task._jaka_sim_steps-initial_ticks)*dt,position,velocity,measured_tick,robot_vector(r,True)])
                    left_fingers=set();right_object_contact=False
                    for contact in task.scene.get_contacts():
                        if not any(body.entity.global_id in arm_links for body in contact.bodies):continue
                        impulse=sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
                        if impulse<1e-5:continue
                        labels=sorted(f"{arm_links.get(body.entity.global_id,'scene')}:{body.entity.name}" for body in contact.bodies)
                        if 'scene:box' in labels:
                            left_fingers.update(s for s in labels if s in ('left:finger_left','left:finger_right'))
                            right_object_contact=right_object_contact or any(s.startswith('right:') for s in labels)
                        key=' / '.join(labels);now=(task._jaka_sim_steps-initial_ticks)*dt
                        entry=contacts.setdefault(key,{'pair':labels,'ticks':0,'peak_impulse_ns':0.,'peak_time_s':now,'first_time_s':now})
                        entry['ticks']+=1;entry['last_time_s']=now
                        if impulse>entry['peak_impulse_ns']:entry.update(peak_impulse_ns=impulse,peak_time_s=now)
                    # Simulator truth is evaluation-only: it never changes the commanded action.
                    lift=float(task.box.get_pose().p[2])-initial_height
                    aperture=float(measured_tick[layout.gripper_indices[0]])
                    held=lift>.05 and .15<aperture<.85 and len(left_fingers)==2 and not right_object_contact
                    grasp_ticks=grasp_ticks+1 if held else 0
                    longest_grasp_ticks=max(longest_grasp_ticks,grasp_ticks)
                    if grasp_ticks*dt>=.6 and grasp_success_time is None:
                        grasp_success_time=(task._jaka_sim_steps-initial_ticks)*dt
                    grasp_evidence.append([(task._jaka_sim_steps-initial_ticks)*dt,lift,aperture,
                                           len(left_fingers)==2,right_object_contact,held])
                    tracking_error=float(np.abs(position-measured_tick)[layout.joint_indices].max())
                    tracking_error_time=tracking_error_time+dt if tracking_error>.12 else 0.
                    if tracking_error_time>=.04:
                        failure_reason={'type':'persistent_tracking_error','step':step,'max_error_rad':tracking_error,'threshold_rad':.12,'confirmation_s':.04}
                        break
                if failure_reason:break
                max_height=max(max_height,float(task.box.get_pose().p[2]))
                if args.task_scope=='left_grasp_lift_hold' and grasp_success_time is not None:break
                # Require measured release and 0.4 s of stable placement, not only a command to open.
                measured_now=robot_vector(r)
                object_velocity=object_body.linear_velocity
                stable=task.check_success() and measured_now[layout.gripper_indices[-1]]>.85 and np.linalg.norm(object_velocity)<.02
                stable_goal_steps=stable_goal_steps+1 if stable else 0
                if stable_goal_steps>=round(.4*rig['control_hz']):success=True;break
                empty_arms=empty_monitor.update(clipped,robot_vector(r))
                if empty_arms:
                    failure_reason={'type':'empty_grasp','arms':empty_arms,'step':step,
                        'evidence':'Commanded closed and measured aperture <8% for 0.16 s; proxy/block-specific threshold.'}
                    break
            duration=(task._jaka_sim_steps-initial_ticks)*dt
            subtask_complete=args.task_scope=='left_grasp_lift_hold' and grasp_success_time is not None and failure_reason is None
            if not success and not subtask_complete and failure_reason is None:failure_reason={'type':'step_budget_exhausted','budget':count}
            final_obs=task.get_obs();frames.append(final_obs['observation']['head_camera']['rgb'])
            video_metadata=None
            if review:
                last=review.capture(final_obs)
                verdict='SUCCESS / full handover' if success else ('GRASP VERIFIED / subtask only' if subtask_complete else 'STOP / '+failure_reason['type'])
                if grasp_success_time is not None:verdict+=' / LEFT GRASP achieved'
                review.write(last,duration,verdict)
                review.write(last,duration,verdict+' / FROZEN final frame',round(review.outro_s*review.fps))
                review.close();video_metadata=review.metadata();review=None
            arrays={'physics_trace':np.asarray(ticks),'control_trace':np.asarray(control),'grasp_evidence':np.asarray(grasp_evidence)}
            if chunks: arrays.update(chunk_steps=np.array([c[0] for c in chunks]),raw_chunks=np.stack([c[1] for c in chunks]),blended_chunks=np.stack([c[2] for c in chunks]))
            np.savez_compressed(args.output/f'seed_{seed}_trace.npz',**arrays)
            (args.output/f'seed_{seed}_contacts.json').write_text(json.dumps(list(contacts.values()),indent=2)+'\n')
            physical=np.asarray(ticks);offset=1+2*layout.size
            tracking=np.abs(physical[:,1:1+layout.size]-physical[:,offset:offset+layout.size])[:,layout.joint_indices]
            report={'seed':seed,'success':bool(success),'steps':step+1,'sim_seconds':duration,'wall_seconds':time.perf_counter()-start,
                'instruction':args.instruction,'success_scope':'full_handover','grasp_success_scope':'left_grasp_lift_hold',
                'task_scope':args.task_scope,'complete_handover_tested':args.task_scope=='full_handover',
                'initial_condition':initial_condition,
                'k':args.k,'robot_id':args.robot_id,'rig_urdf_sha256':urdf_sha,'controller':'bounded_position_velocity_v2','sampling':mode,
                'controller_limits':{key:rig[key] for key in ['joint_velocity_limit_rad_s','joint_acceleration_limit_rad_s2','gripper_rate_per_s','tracking_frequency_rad_s']},
                'planning_calls':len(calls),'inference_seconds':calls,'tracking_joint_mae_rad':float(tracking.mean()),
                'normalization_checks':normalization_checks,
                'tracking_joint_p95_rad':float(np.quantile(tracking,.95)),'hard_limit_events':servo.hard_limit_events,
                'failure_reason':failure_reason,'empty_grasp_monitor':'proprioception_only_v1',
                'grasp_success':grasp_success_time is not None,'grasp_success_time_s':grasp_success_time,
                'longest_verified_left_grasp_s':longest_grasp_ticks*dt,
                'grasp_definition':'Lift >0.05m, measured left aperture in (0.15,0.85), both left fingers contact box with positive impulse >1e-5Ns, no right-arm box contact, all continuously for 0.6s; evaluation only',
                'success_definition':'Task pose tolerance, measured right aperture >85%, object speed <0.02 m/s, all for 0.4 s',
                'oracle_max_object_lift_m':max_height-initial_height,'oracle_note':'Object height is evaluation-only, never policy input or controller gate.',
                'mode':'expert_replay' if expert is not None else 'synchronous_zero_simulated_inference_latency',
                'review_video':video_metadata}
            reports.append(report);(args.output/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
            images_to_video(np.stack(frames),str(args.output/f'seed_{seed}.mp4'),fps=rig['control_hz'])
            print(json.dumps(report),flush=True);task.close_env(clear_cache=True)
        (args.output/'trace_schema.json').write_text(json.dumps({'physics_trace':['time_s',f'command_position[{layout.size}]',f'command_velocity[{layout.size}]',f'measured_position[{layout.size}]',f'measured_velocity[{layout.size}]'],
            'control_trace':['step',f'measured_position[{layout.size}]',f'measured_velocity[{layout.size}]',f'target_raw[{layout.size}]',f'target_clipped[{layout.size}]','object_xyz[3]','left_tcp_pose_wxyz[7]','right_tcp_pose_wxyz[7]'],
            'grasp_evidence':['time_s','lift_m','measured_left_aperture','bilateral_left_finger_contact','right_arm_object_contact','valid_grasp_tick'],
            'rig':rig},indent=2)+'\n')
    finally:
        if review:review.close()
        if conn:
            try:conn.send({'op':'close'});conn.recv()
            finally:conn.close()


if __name__=='__main__':main()
