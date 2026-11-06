"""Reproducible initial joint offsets, applied before observation/data collection.

This sets an episode initial condition, never a corrective action during a policy
rollout. Nominal rig home, geometry and runtime controller are unchanged.
"""
import numpy as np


def sample_left_offset(seed, amplitude):
    if not 0 <= amplitude <= .08:
        raise ValueError('Initial offset amplitude must be between 0 and 0.08 rad')
    return np.random.default_rng(int(seed)+17837).uniform(-amplitude, amplitude, 6)


def apply_left_offset(task, offset):
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (6,) or not np.isfinite(offset).all() or abs(offset).max() > .08:
        raise ValueError('Invalid initial left joint offset')
    if task._jaka_capture_active:
        raise RuntimeError('Initial condition must be set before sampling starts')
    if not np.any(offset):
        return {'left_joint_offset_rad': offset.tolist(), 'settle_s': 0.}
    robot = task.robot
    entity = robot.left_entity
    active = entity.get_active_joints()
    indices = [active.index(j) for j in robot.left_arm_joints]
    target = np.array([j.get_drive_target()[0] for j in robot.left_arm_joints])+offset
    qpos = entity.get_qpos().copy()
    qpos[indices] = target
    entity.set_qpos(qpos)
    entity.set_qvel(np.zeros_like(qpos))
    right = np.array([j.get_drive_target()[0] for j in robot.right_arm_joints])
    arm_ids = {link.entity.global_id: arm for arm in ('left', 'right')
               for link in getattr(robot, f'{arm}_entity').get_links()}
    for _ in range(125):
        robot.set_arm_joints(target, np.zeros(6), 'left')
        robot.set_arm_joints(right, np.zeros(6), 'right')
        task._jaka_step()
        for contact in task.scene.get_contacts():
            labels = [arm_ids.get(b.entity.global_id) for b in contact.bodies]
            impulse = sum(float(np.linalg.norm(p.impulse)) for p in contact.points)
            if impulse > 1e-5 and ('left' in labels) and ('right' in labels or None in labels):
                raise RuntimeError('Initial perturbed configuration contacts another arm or scene')
    measured = entity.get_qpos()[indices]
    if abs(measured-target).max() > .05:
        raise RuntimeError('Initial perturbed configuration did not settle')
    task._jaka_sim_steps = 0
    return {'left_joint_offset_rad': offset.tolist(), 'measured_initial_left_joint_rad': measured.tolist(),
            'settle_s': .5, 'initial_state_reset_only': True}
