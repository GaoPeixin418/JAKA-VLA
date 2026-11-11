"""Separate physical rig/task compatibility from recorded controller ablations."""
import hashlib,json
from pathlib import Path


def rig_urdf_hash(rig,root=Path('/data/jaka-vla')):
    if 'generated' not in rig:
        return hashlib.sha256((root/'repos/RoboTwin/assets/embodiments/jaka-zu7-proxy/robot.urdf').read_bytes()).hexdigest()
    content={side:hashlib.sha256((Path(rig['generated'][side]['asset'])/'robot.urdf').read_bytes()).hexdigest()
             for side in ('left','right')}
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()


def physical_rig_hash(rig):
    keys=('robot_id','arms','source_revision','mounting','home_joint_rad','gripper',
          'receiving_platform_height_m','expert_handover_grasp_heights_m',
          'physics_hz','control_hz','drive_stiffness','drive_damping')
    physical={key:rig[key] for key in keys}
    if 'cameras' in rig:physical['cameras']=rig['cameras']
    return hashlib.sha256(json.dumps(physical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
