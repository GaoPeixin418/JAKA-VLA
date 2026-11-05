"""25 Hz fixed-horizon baseline and expert replay control check.

Initial baseline uses synchronous zero simulated inference latency; wall inference
time is reported separately. This script is not yet the asynchronous delay study.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/data/jaka-vla')
REPO = ROOT / 'repos/RoboTwin'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
os.chdir(REPO)
from scripts.eval_policy_xpolicylab import load_task_args, class_decorator
from envs.utils.images_to_video import images_to_video


def apply_action(task, target):
    """Interpolate command targets for exactly 10 physics ticks = 40 ms."""
    robot = task.robot
    previous = np.array(robot.get_left_arm_jointState() + robot.get_right_arm_jointState())
    target = np.asarray(target, dtype=np.float64).copy()
    target[[6, 13]] = np.clip(target[[6, 13]], 0, 1)
    for part, joints in [(slice(0, 6), robot.left_arm_joints), (slice(7, 13), robot.right_arm_joints)]:
        limits = np.array([joint.get_limits()[0] for joint in joints])
        target[part] = np.clip(target[part], limits[:, 0], limits[:, 1])
    substeps = 10
    for i in range(1, substeps + 1):
        command = previous + (target - previous) * (i / substeps)
        robot.set_arm_joints(command[:6], np.zeros(6), 'left')
        robot.set_arm_joints(command[7:13], np.zeros(6), 'right')
        robot.set_gripper(command[6], 'left', gripper_eps=0)
        robot.set_gripper(command[13], 'right', gripper_eps=0)
        task._jaka_step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--seed-start', type=int, default=1000)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=750)
    p.add_argument('--expert-hdf5', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.k not in range(1, 51):
        raise ValueError('K must be within the 50-step action chunk')
    cfg, _ = load_task_args({'task_name': 'handover_block', 'task_config': 'jaka_smoke', 'policy_name': 'SmolVLA'})
    cfg.update(save_data=False, need_plan=False, eval_mode=True, render_freq=0, eval_video_save_dir=None)
    task = class_decorator('handover_block')
    conn = None if args.expert_hdf5 else Client(str(ROOT / 'run/policy.sock'), family='AF_UNIX')
    expert = None
    if args.expert_hdf5:
        with h5py.File(args.expert_hdf5, 'r') as f:
            keys = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']
            expert = np.concatenate([f['action'][key][:] for key in keys], axis=1)
    reports = []
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            task.setup_demo(now_ep_num=seed, seed=seed, **cfg)
            if conn:
                conn.send({'op': 'reset', 'seed': seed})
                conn.recv()
            queued = []
            frames = []
            calls = []
            success = False
            start = time.perf_counter()
            initial_ticks = task._jaka_sim_steps
            max_steps = len(expert) if expert is not None else args.max_steps
            for step in range(max_steps):
                obs = task.get_obs()
                frames.append(obs['observation']['head_camera']['rgb'])
                if expert is not None:
                    action = expert[step]
                else:
                    if not queued:
                        conn.send({'op': 'predict', 'observation': {
                            'state': obs['joint_measured']['vector'].astype(np.float32),
                            'images': [obs['observation'][c]['rgb'] for c in ['head_camera', 'left_camera', 'right_camera']],
                            'instruction': 'Grasp the block in your left arm, pass it into the right, and place it onto the blue pad.'}})
                        result = conn.recv()
                        calls.append(result['inference_seconds'])
                        queued = list(result['actions'][:args.k])
                    action = queued.pop(0)
                apply_action(task, action)
                if task.check_success():
                    success = True
                    break
            frames.append(task.get_obs()['observation']['head_camera']['rgb'])
            duration = (task._jaka_sim_steps - initial_ticks) * task.scene.get_timestep()
            np.testing.assert_allclose(duration, (step + 1) / 25, atol=1e-5)
            report = {'seed': seed, 'success': bool(success), 'steps': step + 1, 'sim_seconds': duration,
                      'wall_seconds': time.perf_counter() - start, 'k': args.k,
                      'planning_calls': len(calls), 'inference_seconds': calls,
                      'mode': 'expert_replay' if expert is not None else 'synchronous_zero_simulated_inference_latency',
                      'controller': 'linear_command_interpolation_25hz', 'fps': 25}
            reports.append(report)
            (args.output / 'results.json').write_text(json.dumps(reports, indent=2))
            images_to_video(np.stack(frames), str(args.output / f'seed_{seed}.mp4'), fps=25)
            print(json.dumps(report), flush=True)
            task.close_env(clear_cache=True)
    finally:
        if conn:
            conn.send({'op': 'close'})
            conn.recv()
            conn.close()


if __name__ == '__main__':
    main()
