"""Offline FK of saved raw targets versus measured TCP; never executes targets."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import transforms3d as t3d

ROOT = Path('/data/jaka-vla')
REPO = ROOT/'repos/RoboTwin'
sys.path[:0] = [str(ROOT), str(REPO)]
os.chdir(REPO)
from scripts.eval_policy_xpolicylab import load_task_args, class_decorator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--traces', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg, _ = load_task_args({'task_name': 'handover_block', 'task_config': 'zu7_smoke', 'policy_name': 'SmolVLA'})
    cfg.update(save_data=False, need_plan=False, eval_mode=True, render_freq=0)
    task = class_decorator('handover_block')
    task.setup_demo(now_ep_num=0, seed=0, **cfg)
    r = task.robot
    entity = r.left_entity
    active = entity.get_active_joints()
    indices = [active.index(j) for j in r.left_arm_joints]
    tool = r.left_ee.child_link
    tool_index = entity.get_links().index(tool)
    pin = entity.create_pinocchio_model()
    qbase = entity.get_qpos().copy()
    base_pose = entity.get_root_pose()

    def fk(q):
        full = qbase.copy()
        full[indices] = q
        pin.compute_forward_kinematics(full)
        pose = base_pose * pin.get_link_pose(tool_index)
        rotation = t3d.quaternions.quat2mat(pose.q) @ r.left_global_trans_matrix @ r.left_delta_matrix
        return pose.p + rotation @ np.array([r.left_gripper_bias, 0, 0])

    reports = []
    try:
        for path in a.traces:
            data = np.load(path)
            trace = data['control_trace']
            measured_fk = np.stack([fk(row[1:7]) for row in trace])
            errors = np.linalg.norm(measured_fk-trace[:,60:63], axis=1)
            if errors.max() > .0001:
                raise ValueError(f'FK validation failed: {path}: {errors.max()} m')
            raw_fk = np.stack([fk(row[29:35]) for row in trace])
            target_error = np.linalg.norm(raw_fk[:,:2]-trace[:,57:59], axis=1)
            measured_error = np.linalg.norm(measured_fk[:,:2]-trace[:,57:59], axis=1)
            requested = np.flatnonzero(trace[:,35] < .15)
            events = []
            for index in sorted(set([*requested[:1], *requested[::10], len(trace)-1])):
                row = trace[index]
                events.append({'step': int(row[0]), 'time_s': float(row[0]*.04),
                               'requested_aperture': float(row[35]), 'measured_aperture': float(row[7]),
                               'raw_target_horizontal_error_mm': float(1000*target_error[index]),
                               'measured_horizontal_error_mm': float(1000*measured_error[index]),
                               'raw_to_measured_left_joint_max_rad': float(abs(row[29:35]-row[1:7]).max()),
                               'raw_target_tcp': raw_fk[index].tolist(), 'measured_tcp': measured_fk[index].tolist()})
            report = {'trace': str(path), 'fk_measured_max_error_m': float(errors.max()), 'close_events': events}
            reports.append(report)
            print(json.dumps({'trace': str(path), 'fk_max_error_m': float(errors.max()), 'first_close': events[0]}), flush=True)
            np.savez_compressed(a.output/(path.parent.name+'_'+path.stem+'_geometry.npz'),
                                raw_target_tcp=raw_fk, measured_tcp=measured_fk,
                                raw_horizontal_error_m=target_error, measured_horizontal_error_m=measured_error)
        (a.output/'report.json').write_text(json.dumps({'scope': 'Offline FK; object truth is diagnostic only', 'reports': reports}, indent=2)+'\n')
    finally:
        task.close_env(clear_cache=True)


if __name__ == '__main__':
    main()
