"""Bounded initial-grasp expert experiment; records failures and full-view video."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/data/jaka-vla')
REPO = ROOT / 'repos/RoboTwin'
sys.path[:0] = [str(ROOT), str(REPO)]
os.chdir(REPO)
from scripts.eval_policy_xpolicylab import load_task_args, class_decorator
from envs.utils import ArmTag
from eval_v2 import robot_vector
from jaka_vla.review_video import ReviewVideo
from jaka_vla.short_grasp import install_short_grasp
from jaka_vla.initial_conditions import apply_left_offset, apply_left_configuration, sample_left_offset
from jaka_vla.lab_scene import install_scene, install_lab_grasp, rig_urdf_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--original-selector', action='store_true')
    p.add_argument('--no-review-video',action='store_true',help='Keep policy RGB data and numeric evidence; omit extra HD observer rendering')
    p.add_argument('--collect', action='store_true', help='Save identified, uniformly sampled grasp-only supervision')
    p.add_argument('--initial-offset', type=float, default=0., help='Uniform initial left joint perturbation amplitude in rad')
    p.add_argument('--start-trace',type=Path)
    p.add_argument('--start-step',type=int)
    p.add_argument('--rig-config',type=Path,default=ROOT/'configs/robots/dual_zu7.json')
    p.add_argument('--scene-spec',type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rig=json.loads((ROOT/args.rig_config).read_text())
    spec=json.loads(args.scene_spec.read_text()) if args.scene_spec else None
    instruction=spec['instruction'] if spec else 'Use the left arm to grasp the red block, lift it, and hold it.'
    cfg, _ = load_task_args({'task_name': 'handover_block', 'task_config': rig.get('task_config','zu7_smoke'), 'policy_name': 'SmolVLA'})
    cfg.update(save_data=False, need_plan=True, eval_mode=False, render_freq=0,
               save_freq=10 if args.collect else None, save_path=str(args.output/'raw'))
    task = class_decorator('handover_block')
    if spec:install_scene(task,spec)
    task.setup_demo(now_ep_num=0, seed=args.seed, **cfg)
    if args.start_trace:
        if args.start_step is None or args.initial_offset:
            raise ValueError('Recovery requires a trace/step and no offset')
        rows=np.load(args.start_trace)['control_trace']
        if not 0<=args.start_step<len(rows):raise ValueError('Invalid recovery start step')
        row=rows[args.start_step]
        seed_reports=json.loads((args.start_trace.parent/'results.json').read_text())
        if args.seed not in [r['seed'] for r in seed_reports]:raise ValueError('Trace seed mismatch')
        if np.linalg.norm(row[57:60]-rows[0,57:60])>.005:
            raise ValueError('Target already displaced; cannot recreate only robot state')
        initial_condition=apply_left_configuration(task,row[1:7],float(row[7]))
        initial_condition.update(source_trace=str(args.start_trace),source_step=args.start_step,
                                 source_trace_sha256=hashlib.sha256(args.start_trace.read_bytes()).hexdigest())
    else:
        initial_condition = apply_left_offset(task, sample_left_offset(args.seed, args.initial_offset))
    if args.collect:
        task.save_data = True
        task._jaka_capture_active = True
        task._take_picture(_jaka_force=True)
    audit = {} if args.original_selector else (install_lab_grasp(task) if spec else install_short_grasp(task))
    video = None if args.no_review_video else ReviewVideo(task, args.output/f'seed_{args.seed}_multiview.mp4', 25, 'EXPERT GRASP PATH PROBE', args.seed)
    r = task.robot
    arms = {link.entity.global_id: arm for arm in ('left', 'right') for link in getattr(r, f'{arm}_entity').get_links()}
    t0 = task._jaka_sim_steps
    initial_height = float(task.box.get_pose().p[2])
    step = task._jaka_step
    traces, contacts = [], []
    phase = 'approach'
    held_ticks = longest = 0
    track_ticks = 0
    failure = None
    start_wall = time.perf_counter()
    arm_indices = [*range(6), *range(7, 13)]
    initial_tcp = np.asarray(r.get_left_tcp_pose()[:3])
    if video:
        first = video.capture(task.get_obs())
        video.write(first, 0, 'INTRO / frozen frame', 50)

    def observe():
        nonlocal held_ticks, longest, track_ticks
        step()
        now = (task._jaka_sim_steps-t0)*.004
        measured = robot_vector(r)
        velocity = robot_vector(r, True)
        commanded = np.r_[r.get_left_arm_jointState(), r.get_right_arm_jointState()]
        fingers = set()
        right_contact = False
        cross = False
        for contact in task.scene.get_contacts():
            impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
            if impulse <= 1e-5:
                continue
            labels = sorted(f'{arms.get(b.entity.global_id, "scene")}:{b.entity.name}' for b in contact.bodies)
            if not any(arms.get(b.entity.global_id) for b in contact.bodies):
                continue
            contacts.append({'time_s': now, 'pair': labels, 'impulse_ns': impulse})
            if 'scene:box' in labels:
                fingers.update(s for s in labels if s in ('left:finger_left', 'left:finger_right','left:finger_third'))
                right_contact |= any(s.startswith('right:') for s in labels)
            cross |= {arms.get(b.entity.global_id) for b in contact.bodies} == {'left', 'right'}
        lift = float(task.box.get_pose().p[2])-initial_height
        held = lift > .05 and .08 < measured[6] < .90 and len(fingers) >= 2 and not right_contact if spec else lift > .05 and .15 < measured[6] < .85 and len(fingers) == 2 and not right_contact
        held_ticks = held_ticks+1 if held else 0
        longest = max(longest, held_ticks)
        error = float(np.abs(commanded-measured)[arm_indices].max())
        track_ticks = track_ticks+1 if error > .12 else 0
        traces.append(np.r_[now, measured, velocity, commanded, r.get_left_tcp_pose(), task.box.get_pose().p,
                               lift, held, error, len(fingers)])
        if video and (task._jaka_sim_steps-t0) % 10 == 0:
            frame = video.capture(task.get_obs())
            video.write(frame, now, phase+' / expert only')
        if cross:
            raise RuntimeError('interarm_contact')
        if track_ticks >= 10:
            raise RuntimeError('persistent_tracking_error')
    task._jaka_step = observe
    try:
        task.move(task.grasp_actor(task.box, arm_tag=ArmTag('left'), pre_grasp_dis=.07,
                                  grasp_dis=0., contact_point_id=[0, 1, 2, 3]))
        if not task.plan_success:
            raise RuntimeError('grasp_plan_failed')
        phase = 'lift'
        task.move(task.move_by_displacement(ArmTag('left'), z=.1))
        if not task.plan_success:
            raise RuntimeError('lift_plan_failed')
        phase = 'hold 1 second'
        for _ in range(250):
            observe()
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        print('PROBE FAILURE', failure, flush=True)
    finally:
        array = np.asarray(traces)
        duration = (task._jaka_sim_steps-t0)*.004
        if video:
            frame = video.capture(task.get_obs())
            status = 'GRASP VERIFIED / expert only' if longest*.004 >= .6 else 'GRASP NOT VERIFIED'
            if failure:
                status += ' / STOP '+failure
            video.write(frame, duration, status)
            video.write(frame, duration, status+' / FROZEN final frame', 75)
            video.close()
        report = {'seed': args.seed, 'mode': 'expert_grasp_subtask_only','scene_spec':spec,'instruction':instruction, 'selector': 'original' if args.original_selector else 'short_grasp_v1',
                  'grasp_success': longest*.004 >= .6, 'complete_handover_tested': False,
                  'initial_condition': initial_condition,
                  'longest_verified_hold_s': longest*.004, 'failure': failure,
                  'sim_seconds': duration, 'wall_seconds': time.perf_counter()-start_wall, 'video': video.metadata() if video else {'extra_observer_video':False,'policy_rgb_in_raw_dataset':bool(args.collect)},
                  'grasp_definition': 'lift >5cm, measured aperture 15-85%, bilateral finger impulses >1e-5Ns, no right-arm contact, continuous 0.6s'}
        report['selector'] = 'original' if args.original_selector else audit['method']
        if spec:
            report['grasp_definition']='lift >5cm, simulated aperture 8-90%, at least 2 of 3 fingers contact target with impulse >1e-5Ns, no right-arm target contact, continuous 0.6s; evaluation only'
        if len(array):
            # time + measured14 + velocity14 + commanded14 + left_tcp7 + object3 + lift/held/error/fingers
            closed = np.flatnonzero(array[:, 29+6] < .95)
            end = int(closed[0]) if len(closed) else len(array)-1
            tcp = np.vstack([initial_tcp, array[:end+1, 43:46]])
            length = float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum())
            direct = float(np.linalg.norm(tcp[-1]-tcp[0]))
            report.update(preclose_tcp_path_m=length, preclose_endpoint_distance_m=direct,
                          preclose_path_ratio=length/max(direct, 1e-9),
                          max_lift_m=float(array[:,53].max()),
                          joint_speed_peak_rad_s=float(np.abs(array[:,15:29][:,arm_indices]).max()),
                          tracking_error_peak_rad=float(array[:,55].max()),
                          interarm_contact_samples=sum(any(s.startswith('left:') for s in c['pair']) and any(s.startswith('right:') for s in c['pair']) for c in contacts))
        np.savez_compressed(args.output/'trace.npz', trace=array)
        (args.output/'trace_schema.json').write_text(json.dumps(['time_s','measured_position[14]','measured_velocity[14]','commanded_position[14]','left_tcp[7]','object_xyz[3]','lift_m','held','tracking_error_rad','finger_count'], indent=2))
        (args.output/'selection.json').write_text(json.dumps(audit, indent=2)+'\n')
        (args.output/'contacts.json').write_text(json.dumps(contacts, indent=2)+'\n')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        if args.collect and duration>=.04:
            task.merge_pkl_to_hdf5_video(instructions=[instruction])
            raw = args.output/'raw'
            identity = {'robot_id': rig['robot_id'],
                        'rig':rig,'scene_spec':spec,
                        'urdf_sha256':rig_urdf_hash(rig),
                        'task_scope': 'left_grasp_lift_hold', 'seed': args.seed,
                        'initial_condition': initial_condition,
                        'expert_probe_passed': bool(report['grasp_success'] and failure is None),
                        'requires_bounded_servo_replay_acceptance': True}
            (raw/'robot_identity.json').write_text(json.dumps(identity, indent=2)+'\n')
            (raw/'seed.txt').write_text(str(args.seed)+'\n')
            import h5py
            with h5py.File(raw/'data/episode_0000000.hdf5', 'r+') as hf:
                hf.attrs['robot_id'] = identity['robot_id']
                hf.attrs['rig_urdf_sha256'] = identity['urdf_sha256']
                hf.attrs['task_scope'] = identity['task_scope']
            task.remove_data_cache()
        elif args.collect:
            report['dataset_saved']=False
            report['dataset_skip_reason']='Fewer than two observation frames; failed probe retained without training pairs'
            (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            task.remove_data_cache()
        task.close_env(clear_cache=True)


if __name__ == '__main__':
    main()
