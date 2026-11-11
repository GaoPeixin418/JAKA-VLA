"""Independent CPU URDF FK/IK diagnosis. It never executes a candidate."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import yaml

ROOT=Path('/data/jaka-vla')
rig=json.loads((ROOT/'configs/robots/dual_zu7_lab.json').read_text())
asset=Path(rig['generated']['left']['asset'])
xml=ET.parse(asset/'robot.urdf').getroot()
base=np.array(rig['mounting']['world_bases']['left'])
joints=list(xml.findall('joint'));names=[f'joint_{i}' for i in range(1,7)]
lower=[];upper=[]
for name in names:
    limit=xml.find(f"joint[@name='{name}']/limit")
    lower.append(float(limit.get('lower')));upper.append(float(limit.get('upper')))
def fk(q):
    poses={'Link_0':base};values=dict(zip(names,q))
    for joint in joints:
        origin=joint.find('origin');m=np.eye(4)
        if origin is not None:
            m[:3,3]=[float(v) for v in origin.get('xyz','0 0 0').split()]
            m[:3,:3]=Rotation.from_euler('xyz',[float(v) for v in origin.get('rpy','0 0 0').split()]).as_matrix()
        if joint.get('type')=='revolute':
            motion=np.eye(4);axis=np.array([float(v) for v in joint.find('axis').get('xyz').split()])
            motion[:3,:3]=Rotation.from_rotvec(axis*values[joint.get('name')]).as_matrix();m=m@motion
        elif joint.get('type')=='prismatic':
            motion=np.eye(4);motion[:3,3]=np.array([float(v) for v in joint.find('axis').get('xyz').split()])*.05;m=m@motion
        poses[joint.find('child').get('link')]=poses[joint.find('parent').get('link')]@m
    return poses

collision=yaml.safe_load((asset/'collision.yml').read_text())['collision_spheres']
table=yaml.safe_load((asset/'curobo.yml').read_text())['planner']['world_table']
def clearance(q):
    poses=fk(q);rows=[]
    for name,spheres in collision.items():
        m=poses[name]
        for sphere in spheres:
            p=m[:3,:3]@sphere['center']+m[:3,3]-table['position']
            delta=np.abs(p)-np.array(table['dims'])/2
            dist=np.linalg.norm(np.maximum(delta,0))+min(max(delta),0)-sphere['radius']-.002
            rows.append((float(dist),name))
    return min(rows)

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--selection',type=Path,default=ROOT/'results/lab_probe_20261111_0_v8/selection.json')
parser.add_argument('--pose-key',default='pre_pose');args=parser.parse_args()
home=np.array(rig['home_joint_rad']['left']);rng=np.random.default_rng(42)
selection=json.loads(args.selection.read_text())
report={'home_tool_pose':fk(home)['tool'].tolist(),'home_clearance':clearance(home),'candidates':[]}
for row in selection['candidates']:
    goal=row[args.pose_key];goal_rotation=Rotation.from_quat(np.roll(goal[3:],-1))
    def error(q):
        pose=fk(q)['tool']
        return np.r_[(pose[:3,3]-goal[:3])*3,(goal_rotation.inv()*Rotation.from_matrix(pose[:3,:3])).as_rotvec()]
    found=[];best=None
    for start in [home,*rng.uniform(lower,upper,(15,6))]:
        result=least_squares(error,start,bounds=(lower,upper),max_nfev=180)
        if best is None or np.linalg.norm(result.fun)<best['error']:best={'error':float(np.linalg.norm(result.fun)),'q':result.x.tolist(),'active_bounds':result.active_mask.tolist()}
        if np.linalg.norm(result.fun)<.005 and all(np.linalg.norm(result.x-x['q'])>.05 for x in found):
            found.append({'q':result.x.tolist(),'error':float(np.linalg.norm(result.fun)),'table_clearance':clearance(result.x)})
    report['candidates'].append({'yaw':row['yaw'],'solutions':found,'best':best})
print(json.dumps(report,indent=2),flush=True)
