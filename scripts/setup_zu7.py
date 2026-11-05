"""Fetch pinned JAKA geometry and register a configurable two-Zu7 simulation rig."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import yaml

ROOT = Path('/data/jaka-vla')
ROBO = ROOT/'repos/RoboTwin'
CFG = json.loads((ROOT/'configs/robots/dual_zu7.json').read_text())
ASSET = ROBO/'assets/embodiments/jaka-zu7-proxy'
SOURCE = ROOT/'repos/jaka_zu7_description'


def fetch(rel):
    path = SOURCE/rel
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        url=f"https://raw.githubusercontent.com/JakaCobot/jaka_ros2/{CFG['source_revision']}/{rel}"
        data=urllib.request.urlopen(url,timeout=40).read()
        path.write_bytes(data)
    return path


def element(parent, tag, **attrs):
    return ET.SubElement(parent,tag,{k:str(v) for k,v in attrs.items()})


def box_link(robot,name,xyz,size,mass,color):
    link=element(robot,'link',name=name)
    inertia=element(link,'inertial'); element(inertia,'origin',xyz=xyz,rpy='0 0 0'); element(inertia,'mass',value=mass)
    element(inertia,'inertia',ixx=.0002,iyy=.0002,izz=.0002,ixy=0,ixz=0,iyz=0)
    for kind in ('visual','collision'):
        visual=element(link,kind); element(visual,'origin',xyz=xyz,rpy='0 0 0')
        element(element(visual,'geometry'),'box',size=size)
        if kind=='visual': element(element(visual,'material',name=name+'_color'),'color',rgba=color)
    return link


def joint(robot,name,typ,parent,child,xyz,rpy='0 0 0',axis=None):
    j=element(robot,'joint',name=name,type=typ)
    element(j,'parent',link=parent); element(j,'child',link=child); element(j,'origin',xyz=xyz,rpy=rpy)
    if axis: element(j,'axis',xyz=axis)
    return j


def sphere_chain(mesh):
    points=np.asarray(mesh.vertices)
    center=points.mean(axis=0)
    _,_,basis=np.linalg.svd(points-center,full_matrices=False)
    axis=basis[0]
    along=(points-center)@axis
    lo,hi=along.min(),along.max()
    count=max(2,int(math.ceil((hi-lo)/.045)))
    edges=np.linspace(lo,hi,count+1)
    spheres=[]
    for i in range(count):
        mask=(along>=edges[i])&(along<=edges[i+1])
        if not mask.any(): continue
        c=center+axis*((edges[i]+edges[i+1])/2)
        radius=float(np.linalg.norm(points[mask]-c,axis=1).max()+.003)
        spheres.append({'center':[float(x) for x in c],'radius':radius})
    return spheres


def box_spheres(center, size, cell=0.018):
    """Cover the entire collision box with circumspheres of small voxels."""
    size = np.asarray(size)
    count = np.ceil(size / cell).astype(int)
    width = size / count
    radius = float(np.linalg.norm(width / 2))
    return [{'center': (np.asarray(center) - size/2 + (np.asarray(index)+.5)*width).tolist(),
             'radius': radius} for index in np.ndindex(*count)]


def main():
    ASSET.mkdir(parents=True,exist_ok=True)
    source=fetch('src/jaka_description/urdf/jaka_zu7.urdf')
    source_files=[source,fetch('README.md'),fetch('src/jaka_description/package.xml'),fetch('src/jaka_zu7_moveit_config/config/jaka_zu7.srdf')]
    robot=ET.parse(source).getroot()
    for child in list(robot):
        if child.tag not in ('link','joint'): robot.remove(child)
        elif child.tag=='link' and child.get('name')=='world': robot.remove(child)
        elif child.tag=='joint' and child.get('name')=='fixed': robot.remove(child)
    spheres={}
    for i in range(7):
        mesh=fetch(f'src/jaka_description/meshes/jaka_zu7_meshes/Link_{i}.STL')
        source_files.append(mesh)
        spheres[f'Link_{i}']=sphere_chain(trimesh.load(mesh,force='mesh'))
    for mesh in robot.iter('mesh'):
        mesh.set('filename',str(SOURCE/('src/'+mesh.get('filename').replace('package://',''))))
    g=CFG['gripper']
    box_link(robot,'tool','0.035 0 0','0.07 0.12 0.045',.4,'0.15 0.17 0.2 1')
    joint(robot,'tool_mount','fixed','Link_6','tool','0 0 0.02',rpy=f'0 {-math.pi/2} 0')
    half=g['max_opening_m']/2; thick=g['finger_thickness_m']
    for name,sign in [('finger_left',1),('finger_right',-1)]:
        box_link(robot,name,f"{g['tcp_offset_m']-.015} 0 0",f"{g['finger_length_m']} {thick} {g['finger_height_m']}",.06,'0.1 0.1 0.12 1')
        j=joint(robot,name+'_joint','prismatic','tool',name,f'0 {sign*thick/2} 0',axis=f'0 {sign} 0')
        element(j,'limit',lower=0,upper=half,effort=100,velocity=.15)
        element(j,'dynamics',damping=1,friction=.1)
    box_link(robot,'camera','0 0 0','0.025 0.07 0.025',.08,'0.1 0.1 0.1 1')
    joint(robot,'camera_joint','fixed','tool','camera','0.07 0 0.06',rpy='0 0.4 0')
    ET.indent(robot)
    ET.ElementTree(robot).write(ASSET/'robot.urdf',encoding='utf-8',xml_declaration=True)
    # Deliberately named proxy; none of these tool dimensions claim to be real hardware.
    spheres['tool']=box_spheres([.035,0,0],[.07,.12,.045])
    spheres['finger_left']=[{'center':[.12,0,0],'radius':.023},{'center':[.15,0,0],'radius':.023}]
    spheres['finger_right']=deepcopy(spheres['finger_left'])
    spheres['camera']=[{'center':[0,0,0],'radius':.04}]
    (ASSET/'collision.yml').write_text(yaml.safe_dump({'collision_spheres':spheres}))
    arm_joints=[f'joint_{i}' for i in range(1,7)]
    links=list(spheres)
    ignore={f'Link_{i}':[f'Link_{j}' for j in range(i+1,min(7,i+3))] for i in range(7)}
    for i in (4,5,6): ignore[f'Link_{i}']+=['tool','camera','finger_left','finger_right']
    ignore['tool']=['camera','finger_left','finger_right']; ignore['camera']=['finger_left','finger_right'];ignore['finger_left']=['finger_right']
    kinematics={'use_usd_kinematics':False,'urdf_path':str(ASSET/'robot.urdf'),'asset_root_path':None,
        'base_link':'Link_0','ee_link':'tool','collision_link_names':links,'collision_spheres':str(ASSET/'collision.yml'),
        'collision_sphere_buffer':0.002,'extra_collision_spheres':{},'use_global_cumul':True,'self_collision_ignore':ignore,
        'self_collision_buffer':{name:0.0 for name in links},'mesh_link_names':links,
        'lock_joints':{'finger_left_joint':half,'finger_right_joint':half},'extra_links':None,
        'cspace':{'joint_names':arm_joints+['finger_left_joint','finger_right_joint'],
          'retract_config':CFG['home_joint_rad']+[half,half],'null_space_weight':[1.]*8,'cspace_distance_weight':[1.]*8,
          'max_jerk':100.,'max_acceleration':3.,'velocity_scale':CFG['expert_velocity_scale']}}
    # RoboTwin's table pose is its TOP; the collision box center is 2.5 cm lower.
    (ASSET/'curobo.yml').write_text(yaml.safe_dump({'robot_cfg':{'kinematics':kinematics},'planner':{'other_arm_collision':CFG['interarm_collision_planning'],'frame_bias':[0,0,0], 'world_table':{'dims':[1.2,.7,.05],'position':[0,0,.715],'quaternion':[1,0,0,0]}}}))
    mount=CFG['mounting']; yaw=mount['base_yaw_rad']
    embodiment={'use_tool_link_pose':True,'urdf_path':'robot.urdf','joint_stiffness':CFG['drive_stiffness'],'joint_damping':CFG['drive_damping'],
        'gripper_stiffness':1000,'gripper_damping':100,'move_group':['tool','tool'],'ee_joints':['tool_mount','tool_mount'],
        'arm_joints_name':[arm_joints,arm_joints],'gripper_name':[{'base':'finger_left_joint','mimic':[['finger_right_joint',1.,0.]]}]*2,
        'gripper_bias':g['tcp_offset_m'],'gripper_scale':[0,half],'homestate':[CFG['home_joint_rad']]*2,
        'delta_matrix':np.eye(3).tolist(),'global_trans_matrix':np.eye(3).tolist(),
        'robot_pose':[[0,mount['base_y_m'],mount['base_z_m'],math.cos(yaw/2),0,0,math.sin(yaw/2)]],
        'planner':'curobo','dual_arm':False,'rotate_lim':[0,1],'grasp_perfect_direction':['front_right','front_left'],
        'static_camera_list':[{'name':'head_camera','type':'D435','position':[0,-.7,1.6],'forward':[0,.65,-.75],'left':[-1,0,0]}]}
    (ASSET/'config.yml').write_text(yaml.safe_dump(embodiment,sort_keys=False))
    registry=ROBO/'env_cfg/task_config/_embodiment_config.yml'
    existing=yaml.safe_load(registry.read_text());existing['jaka-zu7-proxy']={'file_path':str(ASSET)}
    registry.write_text(yaml.safe_dump(existing,sort_keys=False))
    for path,key in [(ROBO/'env_cfg/robot/_robot_info.json','dual_jaka_zu7'),(ROBO/'XPolicyLab/utils/robot/_robot_info.json','dual_jaka_zu7')]:
        data=json.loads(path.read_text());data[key]={'arm_dim':[6,6],'ee_dim':[1,1]};path.write_text(json.dumps(data,indent=2)+'\n')
    (ROBO/'env_cfg/dual_jaka_zu7.yml').write_text(yaml.safe_dump({'config':{'robot':'dual_jaka_zu7'},'observation':{'collect_freq':25}}))
    task=yaml.safe_load((ROOT/'configs/jaka_pilot.yml').read_text())
    task.update(embodiment=['jaka-zu7-proxy','jaka-zu7-proxy',mount['base_separation_m']],episode_num=3,
                zu7_grasp_heights_m=CFG['expert_handover_grasp_heights_m'],
                zu7_receiving_platform_height_m=CFG['receiving_platform_height_m'],zu7_sequential_return=CFG['sequential_return'])
    for name,count in [('zu7_smoke',3),('zu7_collision_smoke',3),('zu7_pilot',50)]:
        task['episode_num']=count
        task['max_seed_attempts']=12 if count==3 else 100
        for path in [ROOT/f'configs/{name}.yml',ROBO/f'env_cfg/task_config/{name}.yml']:
            path.write_text(yaml.safe_dump(task,sort_keys=False))
    manifest={'repository':CFG['source_repository'],'revision':CFG['source_revision'],
      'license_note':'Upstream package.xml currently declares TODO; retained as local fetched dependency, no assets bundled for redistribution.',
      'files':{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
      'assumptions':CFG,'generated_urdf':str(ASSET/'robot.urdf')}
    (ROOT/'manifests/jaka_zu7_source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'asset':str(ASSET),'registered_robot':'dual_jaka_zu7','source':CFG['source_revision']},indent=2))


if __name__=='__main__': main()
