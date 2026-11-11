"""Extract laboratory evidence without executing archived ROS/SDK programs.

The generated profile records what the archive says, not a new real-robot calibration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    args=p.parse_args();source=args.archive_root/'打包temp';dest=args.output
    if dest.exists() or args.profile.exists():raise FileExistsError('Refusing to overwrite evidence')
    dest.mkdir(parents=True)
    relative=['qjaka_gui/urdf/jaka_dual.urdf','qjaka_gui/urdf/jaka_dual.xacro',
              'qjaka_gui/README.md','qjaka_gui/src/robot.hpp','qjaka_gui/src/subwindow.cpp',
              'qjaka_gui/src/mainwindow.cpp','qjaka_gui/srv/GraspService.srv',
              'zu7/urdf/jaka_macro.xacro','jaka_zu7_v2/urdf/jaka_zu7_v2.xacro',
              'pickingv2/urdf/jaka_dual.xacro','pickingv2/scripts/base/config.py',
              'pickingv2/scripts/base/ik.py','pickingv2/scripts/center.py',
              'pickingv2/launch/cam_to_right.launch','dual_moveit_config/config/joint_limits.yaml',
              'dual_moveit_config/config/dual.srdf','dual_moveit_config/config/kinematics.yaml']
    relative += [str(x.relative_to(source)) for x in (source/'Kinect标定文件').rglob('*.yaml')]
    robot=ET.parse(source/relative[0]).getroot()
    for mesh in robot.iter('mesh'):
        name=mesh.get('filename')
        if not name.startswith('package://'):raise ValueError(name)
        relative.append(name.removeprefix('package://'))
    records={}
    for rel in sorted(set(relative)):
        path=source/rel
        if not path.is_file():raise FileNotFoundError(path)
        out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,out)
        records[rel]={'sha256':sha(path),'bytes':path.stat().st_size}
    shutil.copyfile(args.archive_root/'关节限位参数.txt',dest/'joint_limits_notes.txt')
    records['joint_limits_notes.txt']={'sha256':sha(dest/'joint_limits_notes.txt'),'bytes':(dest/'joint_limits_notes.txt').stat().st_size}
    joints={j.get('name'):j for j in robot.findall('joint')}
    def joint_record(j):
        origin=j.find('origin');limit=j.find('limit');axis=j.find('axis')
        result={'name':j.get('name'),'type':j.get('type'),'parent':j.find('parent').get('link'),
                'child':j.find('child').get('link'),
                'xyz_m':[float(v) for v in origin.get('xyz','0 0 0').split()],
                'rpy_rad':[float(v) for v in origin.get('rpy','0 0 0').split()]}
        if axis is not None:result['axis']=[float(v) for v in axis.get('xyz').split()]
        if limit is not None:result['limit']={k:float(v) for k,v in limit.attrib.items()}
        return result
    arms={a:{'source_model':'zu7_v1' if a=='left' else 'zu7_v2',
               'joints':[joint_record(joints[f'{a}_joint_{i}']) for i in range(1,7)],
               'flange_to_gripper_base':joint_record(joints[f'{a}_ee_joint']),
               'home_joint_deg':[-90,90,-50,200,170,0] if a=='left' else [90,90,60,-30,-170,0]}
          for a in ('left','right')}
    mesh=dest/'jaka_zu7_v2/meshes/gripper-raw.STL';b=mesh.read_bytes();n=struct.unpack_from('<I',b,80)[0]
    vertices=[struct.unpack_from('<3f',b,84+i*50+12+j*12) for i in range(n) for j in range(3)]
    bounds=[[f(v[k] for v in vertices) for k in range(3)] for f in (min,max)]
    relative_mount=joint_record(joints['left_to_right'])
    profile={'profile_id':'dual_zu7_lab_archive_20261111','robot_id':'dual_jaka_zu7_lab_archive',
        'evidence_status':'archive-derived; current real installation unconfirmed by user',
        'source_root_runtime':'/data/jaka-vla/data/hardware/lab_archive_20261111',
        'source_manifest':records,'arms':arms,
        'mounting':{'right_base_to_left_base':relative_mount,
                    'base_separation_m':math.sqrt(sum(x*x for x in relative_mount['xyz_m'])),
                    'platform_to_right_base':joint_record(joints['right_to_platform']),
                    'platform_size_m':[1.2,.6,.6]},
        'gripper':{'mesh':'jaka_zu7_v2/meshes/gripper-raw.STL','mesh_bounds_m':bounds,
                   'geometry':'three-finger fixed whole-tool mesh; no articulated fingers in archive',
                   'tcp_from_gripper_base_m':[0,0,.12],
                   'tcp_evidence':'pickingv2/scripts/base/config.py; IK d6 includes flange-to-gripper mount',
                   'tcp_from_flange_m':{'left':[0,0,.1555],'right':[0,0,.159]},
                   'control':{'controller':'right_arm_cabinet','left_channel':0,'right_channel':1,
                              'open_level':False,'close_level':True,'interface':'JAKA SDK set_digital_output(IO_CABINET, index, value)'},
                   'grasp_service_semantics':'sixth arm joint rotation; not gripper opening/closing',
                   'stroke_m':None,'measured_aperture_available':None,'force_feedback_available':None,
                   'movable_simulation_model':'requires explicitly labelled approximation and replay validation'},
        'camera':{'name':'Kinect2','serial_from_calibration':'004878164947',
                  'right_base_to_kinect2_link':{'xyz_m':[.151197,-.0294089,.144123],
                                               'quaternion_xyzw':[.025197,-.514666,.856983,-.00806656]},
                  'calibration_dir':'Kinect标定文件/004878164947',
                  'wrist_cameras_evidenced':False,
                  'note':'calib_pose.yaml is RGB/depth sensor calibration, not robot hand-eye calibration; optical-frame transform still needs validation'},
        'conflicts':['URDF limits differ from MoveIt YAML and joint_limits_notes.txt; use conservative intersection for simulation, not hardware-certified limits',
                     'Right URDF effort=0 is a CAD export placeholder; simulation drive effort must be declared separately',
                     'Kinematic models differ by side; old shared-URDF training data is incompatible',
                     'Source GUI reports commanded digital output success without measured aperture confirmation'],
        'real_robot_connected':False}
    args.profile.parent.mkdir(parents=True,exist_ok=True)
    args.profile.write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
    (dest/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records.values()),
                      'profile':str(args.profile),'base_separation_m':profile['mounting']['base_separation_m']},ensure_ascii=False))

if __name__=='__main__':main()
