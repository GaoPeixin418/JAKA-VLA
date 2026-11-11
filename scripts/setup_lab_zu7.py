"""Create separate left/right archive-based embodiments with a declared 3-finger proxy.

The source static whole-gripper mesh is preserved separately. Its moving joints and
stroke are absent, so the articulated fingers below are explicitly an approximation.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import transforms3d as t3d
import trimesh
import yaml

from setup_zu7 import box_link, box_spheres, element, joint, sphere_chain

ROOT=Path('/data/jaka-vla');ROBO=ROOT/'repos/RoboTwin'
PROFILE=ROOT/'configs/robots/lab_archive_20261111.json'
SOURCE=ROOT/'data/hardware/lab_archive_20261111'


def transform(xyz,rpy):
    m=np.eye(4);m[:3,:3]=t3d.euler.euler2mat(*rpy);m[:3,3]=xyz;return m


def main():
    profile=json.loads(PROFILE.read_text());source=ET.parse(SOURCE/'qjaka_gui/urdf/jaka_dual.urdf').getroot()
    limits=yaml.safe_load((SOURCE/'dual_moveit_config/config/joint_limits.yaml').read_text())['joint_limits']
    mount=profile['mounting']['right_base_to_left_base']
    # Rigid world transform retains the complete measured six-DOF base relationship.
    platform=transform([.44311231,.4,.74],[0,0,math.pi])
    right=platform@transform([.2,.4,0],[0,0,0])
    left=right@transform(mount['xyz_m'],mount['rpy_rad'])
    table_center=(platform@np.array([.6,.3,-.025,1]))[:3].tolist()
    generated={};registry_path=ROBO/'env_cfg/task_config/_embodiment_config.yml'
    registry=yaml.safe_load(registry_path.read_text())
    for side,pose in [('left',left),('right',right)]:
        asset=ROBO/f'assets/embodiments/jaka-zu7-lab-{side}'
        if asset.exists():raise FileExistsError(asset)
        asset.mkdir(parents=True)
        robot=ET.Element('robot',name=f'jaka_zu7_lab_{side}')
        names={f'{side}_base_link':'Link_0',**{f'{side}_link_{i}':f'Link_{i}' for i in range(1,7)}}
        spheres={}
        for old,new in names.items():
            link=deepcopy(source.find(f"link[@name='{old}']"));link.set('name',new)
            for mesh in link.iter('mesh'):
                path=SOURCE/mesh.get('filename').removeprefix('package://');mesh.set('filename',str(path))
            robot.append(link)
            collision=link.find('collision');mesh_path=collision.find('geometry/mesh').get('filename')
            mesh=trimesh.load(mesh_path,force='mesh');origin=collision.find('origin')
            if origin is not None:
                mesh.apply_transform(transform([float(x) for x in origin.get('xyz','0 0 0').split()],
                                               [float(x) for x in origin.get('rpy','0 0 0').split()]))
            spheres[new]=sphere_chain(mesh)
        # Conservative intersection of archive URDF, MoveIt and the separate note.
        note_lo=np.deg2rad([-270,-50,-155,-85,-270,-270]);note_hi=np.deg2rad([270,230,155,265,270,270])
        for i in range(1,7):
            j=deepcopy(source.find(f"joint[@name='{side}_joint_{i}']"));j.set('name',f'joint_{i}')
            for key in ['parent','child']:j.find(key).set('link',names[j.find(key).get('link')])
            limit=j.find('limit');ml=limits[f'{side}_joint_{i}']
            limit.set('lower',str(max(float(limit.get('lower')),ml['min_position'],note_lo[i-1])))
            limit.set('upper',str(min(float(limit.get('upper')),ml['max_position'],note_hi[i-1])))
            limit.set('effort','100');limit.set('velocity','1.5');robot.append(j)
        offset=profile['arms'][side]['flange_to_gripper_base']['xyz_m']
        box_link(robot,'tool','.04 0 0','.08 .08 .08',.4,'.2 .25 .3 1')
        joint(robot,'tool_mount','fixed','Link_6','tool',' '.join(map(str,offset)),rpy=f'0 {-math.pi/2} 0')
        # Archive-derived body visualization, cut before the unarticulated fingers.
        mesh=trimesh.load(SOURCE/profile['gripper']['mesh'],force='mesh')
        body=mesh.slice_plane([0,0,.078],[0,0,-1],cap=False)
        body.apply_transform(np.linalg.inv(transform([0,0,0],[0,-math.pi/2,0])))
        body.export(asset/'gripper_body.stl')
        tool=robot.find("link[@name='tool']");visual=tool.find('visual');tool.remove(visual)
        visual=element(tool,'visual');element(element(visual,'geometry'),'mesh',filename=str(asset/'gripper_body.stl'))
        element(element(visual,'material',name='lab_body'),'color',rgba='.2 .25 .3 1')
        spheres['tool']=box_spheres([.04,0,0],[.08,.08,.08],cell=.025)
        finger_names=['finger_left','finger_right','finger_third'];angles=[0,2*math.pi/3,4*math.pi/3]
        for name,angle in zip(finger_names,angles):
            # One radial slider per finger. Aperture is simulated, not measured hardware feedback.
            cy,cz=math.cos(angle),math.sin(angle)
            box_link(robot,name,'.13 0 0','.10 .012 .018',.06,'.18 .28 .35 1')
            j=joint(robot,name+'_joint','prismatic','tool',name,f'0 {.01*cy} {.01*cz}',
                    rpy=f'{angle} 0 0',axis='0 1 0')
            element(j,'limit',lower=0,upper=.05,effort=50,velocity=.1)
            element(j,'dynamics',damping=1,friction=.1)
            spheres[name]=box_spheres([.13,0,0],[.10,.012,.018],cell=.025)
        box_link(robot,'camera','0 0 0','.025 .07 .025',.08,'.1 .1 .1 1')
        joint(robot,'camera_joint','fixed','tool','camera','.07 0 .07',rpy='0 .4 0')
        spheres['camera']=[{'center':[0,0,0],'radius':.04}]
        ET.indent(robot);ET.ElementTree(robot).write(asset/'robot.urdf',encoding='utf-8',xml_declaration=True)
        (asset/'collision.yml').write_text(yaml.safe_dump({'collision_spheres':spheres}))
        joints=[f'joint_{i}' for i in range(1,7)];gripper_joints=[n+'_joint' for n in finger_names]
        home=np.deg2rad(profile['arms'][side]['home_joint_deg']).tolist()
        ignore={f'Link_{i}':[f'Link_{j}' for j in range(i+1,min(7,i+3))] for i in range(7)}
        for i in (4,5,6):ignore[f'Link_{i}']+=['tool','camera',*finger_names]
        ignore['tool']=['camera',*finger_names];ignore['camera']=finger_names
        for n in finger_names:ignore[n]=[f for f in finger_names if f!=n]
        kin={'use_usd_kinematics':False,'urdf_path':str(asset/'robot.urdf'),'asset_root_path':None,
             'base_link':'Link_0','ee_link':'tool','collision_link_names':list(spheres),
             'collision_spheres':str(asset/'collision.yml'),'collision_sphere_buffer':.002,
             'extra_collision_spheres':{},'use_global_cumul':True,'self_collision_ignore':ignore,
             'self_collision_buffer':{k:0. for k in spheres},'mesh_link_names':list(spheres),
             'lock_joints':{k:.05 for k in gripper_joints},'extra_links':None,
             'cspace':{'joint_names':joints+gripper_joints,'retract_config':home+[.05]*3,
                       'null_space_weight':[1.]*9,'cspace_distance_weight':[1.]*9,
                       'max_jerk':50.,'max_acceleration':2.,'velocity_scale':.5}}
        planner={'other_arm_collision':True,'frame_bias':[0,0,0],
                 'world_table':{'dims':[1.2,.6,.05],'position':table_center,'quaternion':[1,0,0,0]}}
        (asset/'curobo.yml').write_text(yaml.safe_dump({'robot_cfg':{'kinematics':kin},'planner':planner}))
        gripper={'base':gripper_joints[0],'mimic':[[n,1.,0.] for n in gripper_joints[1:]]}
        embodiment={'project_inprocess_planners':True,'use_tool_link_pose':True,'urdf_path':'robot.urdf','joint_stiffness':10000,'joint_damping':500,
            'gripper_stiffness':1000,'gripper_damping':100,'move_group':['tool']*2,'ee_joints':['tool_mount']*2,
            'arm_joints_name':[joints]*2,'gripper_name':[gripper]*2,'gripper_bias':.12,'gripper_scale':[0,.05],
            'homestate':[home]*2,'delta_matrix':np.eye(3).tolist(),'global_trans_matrix':np.eye(3).tolist(),
            'robot_pose':[[*pose[:3,3].tolist(),*t3d.quaternions.mat2quat(pose[:3,:3]).tolist()]],
            'planner':'curobo','dual_arm':False,'rotate_lim':[0,1],'grasp_perfect_direction':['front_right','front_left'],
            'static_camera_list':[{'name':'head_camera','type':'D435','position':[0,-.7,1.6],
                                   'forward':[0,.65,-.75],'left':[-1,0,0]}]}
        (asset/'config.yml').write_text(yaml.safe_dump(embodiment,sort_keys=False))
        registry[f'jaka-zu7-lab-{side}']={'file_path':str(asset)}
        generated[side]={'asset':str(asset),'urdf_sha256':hashlib.sha256((asset/'robot.urdf').read_bytes()).hexdigest(),
                         'world_base_matrix':pose.tolist(),'home_joint_rad':home}
    registry_path.write_text(yaml.safe_dump(registry,sort_keys=False))
    task=yaml.safe_load((ROOT/'configs/zu7_smoke.yml').read_text())
    task.update(embodiment=['jaka-zu7-lab-left','jaka-zu7-lab-right',0.],episode_num=3,max_seed_attempts=12,
                zu7_grasp_heights_m=None)
    for path in [ROOT/'configs/zu7_lab_smoke.yml',ROBO/'env_cfg/task_config/zu7_lab_smoke.yml']:
        path.write_text(yaml.safe_dump(task,sort_keys=False))
    rig=json.loads((ROOT/'configs/robots/dual_zu7.json').read_text())
    rig.update(robot_id='dual_jaka_zu7_lab_archive',source_repository='lab_archive:JAKA_Dual',source_revision=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        mounting={'archive_profile':profile['profile_id'],'world_bases':{k:v['world_base_matrix'] for k,v in generated.items()}},
        home_joint_rad={k:v['home_joint_rad'] for k,v in generated.items()},
        gripper={'name':'lab_three_finger_radial_proxy','physical_model_confirmed':False,'tcp_offset_m':.12,
                 'max_radial_stroke_m':.05,'source_mesh':profile['gripper']['mesh'],'binary_real_control':True,
                 'simulation_aperture_feedback_only':True},
        joint_velocity_limit_rad_s=1.5,joint_acceleration_limit_rad_s2=3.,expert_velocity_scale=.5,
        generated=generated,task_config='zu7_lab_smoke',
        notes='Archive arm kinematics and base transform; approximate articulated gripper and simulation-only cameras; no real hardware validation.')
    (ROOT/'configs/robots/dual_zu7_lab.json').write_text(json.dumps(rig,indent=2)+'\n')
    for path in [ROBO/'env_cfg/robot/_robot_info.json',ROBO/'XPolicyLab/utils/robot/_robot_info.json']:
        data=json.loads(path.read_text());data[rig['robot_id']]={'arm_dim':[6,6],'ee_dim':[1,1]};path.write_text(json.dumps(data,indent=2)+'\n')
    (ROBO/f"env_cfg/{rig['robot_id']}.yml").write_text(yaml.safe_dump({'config':{'robot':rig['robot_id']},'observation':{'collect_freq':25}}))
    manifest={'source_profile_sha256':hashlib.sha256(PROFILE.read_bytes()).hexdigest(),'generated':generated,
              'assumptions':rig['notes'],'gripper_proxy':rig['gripper'],'world_table':planner['world_table']}
    (ROOT/'manifests/lab_hardware_20261111.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
