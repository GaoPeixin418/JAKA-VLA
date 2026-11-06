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


def apply_left_configuration(task, joint_position, aperture=1.):
    """Set an explicitly labelled recovery initial state, before recording starts."""
    if task._jaka_capture_active:
        raise RuntimeError('Recovery state must precede all sampled observations')
    target = np.asarray(joint_position, dtype=np.float64)
    if target.shape != (6,) or not np.isfinite(target).all() or not .85 <= aperture <= 1.:
        raise ValueError('Recovery requires six finite joints and an open gripper')
    r = task.robot
    entity = r.left_entity
    active = entity.get_active_joints()
    indices = [active.index(j) for j in r.left_arm_joints]
    bounds = np.array([j.get_limits()[0] for j in r.left_arm_joints])
    if np.any(target < bounds[:,0]) or np.any(target > bounds[:,1]):
        raise ValueError('Recovery initial joints outside URDF limits')
    qpos = entity.get_qpos().copy()
    qpos[indices] = target
    for joint, multiplier, offset in r.left_gripper:
        low, high = r.left_gripper_scale
        qpos[active.index(joint)] = (low+aperture*(high-low))*multiplier+offset
    entity.set_qpos(qpos)
    entity.set_qvel(np.zeros_like(qpos))
    right = np.array([j.get_drive_target()[0] for j in r.right_arm_joints])
    ids = {link.entity.global_id: arm for arm in ('left', 'right') for link in getattr(r,f'{arm}_entity').get_links()}
    for _ in range(125):
        r.set_arm_joints(target, np.zeros(6), 'left')
        r.set_arm_joints(right, np.zeros(6), 'right')
        r.set_gripper(aperture, 'left', gripper_eps=0)
        task._jaka_step()
        for c in task.scene.get_contacts():
            labels = [ids.get(b.entity.global_id) for b in c.bodies]
            impulse = sum(float(np.linalg.norm(p.impulse)) for p in c.points)
            if impulse > 1e-5 and 'left' in labels and ('right' in labels or None in labels):
                raise RuntimeError('Recovery initial state contacts other arm or scene')
    measured = entity.get_qpos()[indices]
    if abs(measured-target).max() > .05:
        raise RuntimeError('Recovery initial state did not settle')
    task._jaka_sim_steps = 0
    return {'left_joint_position_rad': target.tolist(), 'left_aperture': aperture,
            'measured_initial_left_joint_rad': measured.tolist(), 'settle_s': .5,
            'initial_state_reset_only': True, 'scope': 'intervened_recovery_start_not_executed_prefix'}


def apply_recorded_condition(task, condition):
    if 'left_joint_position_rad' in condition:
        return apply_left_configuration(task, condition['left_joint_position_rad'], condition.get('left_aperture',1.))
    return apply_left_offset(task, condition['left_joint_offset_rad'])
