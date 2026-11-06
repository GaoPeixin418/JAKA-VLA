"""Opt-in, bounded geometric expert selection for the initial left Zu7 grasp.

Simulator object poses are expert supervision only. No learned policy calls this.
Candidates retain the existing cuRobo collision world and Cartesian approach
constraint. Cached paths prevent execution from silently choosing another IK branch.
"""
from copy import deepcopy
from types import MethodType

import numpy as np
import transforms3d as t3d


def joint_arc(position, start):
    points = np.vstack([np.asarray(start), np.asarray(position)])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def install_short_grasp(task):
    original_choose = task.choose_grasp_pose
    original_move = task.left_move_to_pose
    robot = task.robot
    entity = robot.left_entity
    active = entity.get_active_joints()
    indices = [active.index(j) for j in robot.left_arm_joints]
    queue = []
    task.short_grasp_audit = {'method': 'minimum_joint_arc_8_candidates_target_keepout_v2',
                             'target_keepout_dimensions_m': [.07, .07, .21],
                             'candidates': [], 'executed_cached_paths': 0}

    def choose(self, actor, arm_tag, pre_dis=.1, target_dis=0, contact_point_id=None):
        if str(arm_tag) != 'left' or self.short_grasp_audit['candidates']:
            return original_choose(actor, arm_tag, pre_dis, target_dis, contact_point_id)
        if not self.need_plan:
            raise RuntimeError('Short-grasp selector is a planning-only expert adapter')
        ids = [0, 1, 2, 3] if contact_point_id is None else list(contact_point_id)
        if len(ids) > 4:
            raise ValueError('At most four contact faces are supported')
        initial = entity.get_qpos().copy()
        best = None
        contact_to_gripper = np.array([[0, 0, 1, 0], [-1, 0, 0, 0],
                                       [0, -1, 0, 0], [0, 0, 0, 1]])
        for cid in ids:
            matrix = actor.get_contact_point(cid, 'matrix') @ contact_to_gripper
            for roll in (0, 1):
                rotation = matrix[:3, :3] @ np.diag([1, -1 if roll else 1, -1 if roll else 1])
                quat = t3d.quaternions.mat2quat(rotation)
                pre = np.r_[matrix[:3, 3] + rotation @ [-.12-pre_dis, 0, 0], quat].tolist()
                grasp = np.r_[matrix[:3, 3] + rotation @ [-.12-target_dis, 0, 0], quat].tolist()
                row = {'contact_id': cid, 'roll_pi': bool(roll), 'pre_pose': pre, 'grasp_pose': grasp}
                self.short_grasp_audit['candidates'].append(row)
                # Before reaching the pregrasp, the fingers must not sweep through
                # the target. The final constrained insertion intentionally permits
                # target contact, while retaining table and opposite-arm geometry.
                planner = robot.left_planner
                local_target = planner.robot_origion_pose.inv() * actor.get_pose()
                obstacles = planner.project_world['cuboid']
                key = 'short_grasp_target_keepout'
                if key in obstacles:
                    raise RuntimeError('Unexpected existing target keepout')
                obstacles[key] = {'dims': [.07, .07, .21],
                                  'pose': list(local_target.p) + list(local_target.q)}
                try:
                    first = robot.left_plan_path(pre, last_qpos=initial)
                finally:
                    del obstacles[key]
                    planner.refresh_other_arm_world()
                row['pre_status'] = first['status']
                if first['status'] != 'Success':
                    print('CANDIDATE', row, flush=True)
                    continue
                after = initial.copy()
                after[indices] = first['position'][-1]
                second = robot.left_plan_path(grasp, constraint_pose=[1, 1, 1, 0, 0, 0], last_qpos=after)
                row['grasp_status'] = second['status']
                if second['status'] != 'Success':
                    print('CANDIDATE', row, flush=True)
                    continue
                cost = joint_arc(first['position'], initial[indices]) + joint_arc(second['position'], after[indices])
                row.update(joint_arc_rad=cost, planned_seconds=.004*(len(first['position'])+len(second['position'])))
                print('CANDIDATE', row, flush=True)
                if best is None or cost < best[0]:
                    best = (cost, pre, grasp, first, second, initial, after, len(self.short_grasp_audit['candidates'])-1)
        if best is None:
            raise RuntimeError('No candidate passed both pregrasp and constrained approach planning')
        _, pre, grasp, first, second, initial, after, selected = best
        self.short_grasp_audit['selected_index'] = selected
        queue.extend([(pre, None, first, initial), (grasp, [1, 1, 1, 0, 0, 0], second, after)])
        return pre, grasp

    def move(self, pose, constraint_pose=None, **kwargs):
        if not queue:
            return original_move(pose, constraint_pose=constraint_pose, **kwargs)
        expected, constraint, result, start = queue.pop(0)
        error = float(np.abs(entity.get_qpos()[indices] - start[indices]).max())
        if not np.allclose(pose, expected, atol=1e-7) or constraint_pose != constraint or error > .05:
            raise RuntimeError(f'Cached grasp path no longer matches execution state: {error:.5f} rad')
        self.left_joint_path.append(deepcopy(result))
        self.short_grasp_audit['executed_cached_paths'] += 1
        return deepcopy(result)

    task.choose_grasp_pose = MethodType(choose, task)
    task.left_move_to_pose = MethodType(move, task)
    return task.short_grasp_audit
