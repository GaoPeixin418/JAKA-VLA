"""Separate physical rig/task compatibility from recorded controller ablations."""
import hashlib,json


def physical_rig_hash(rig):
    keys=('robot_id','arms','source_revision','mounting','home_joint_rad','gripper',
          'receiving_platform_height_m','expert_handover_grasp_heights_m',
          'physics_hz','control_hz','drive_stiffness','drive_damping')
    physical={key:rig[key] for key in keys}
    return hashlib.sha256(json.dumps(physical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
