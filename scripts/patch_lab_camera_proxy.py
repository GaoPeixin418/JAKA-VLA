"""Remove unsupported simulated camera collision shells that overlap the fingers."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import yaml
root=Path('/data/jaka-vla');rig_path=root/'configs/robots/dual_zu7_lab.json'
rig=json.loads(rig_path.read_text());manifest=json.loads((root/'manifests/lab_hardware_20261111.json').read_text())
for side in ('left','right'):
    asset=Path(rig['generated'][side]['asset']);p=asset/'robot.urdf';xml=ET.parse(p)
    camera=xml.getroot().find("link[@name='camera']")
    for collision in list(camera.findall('collision')):camera.remove(collision)
    ET.indent(xml);xml.write(p,encoding='utf-8',xml_declaration=True)
    p=asset/'collision.yml';data=yaml.safe_load(p.read_text());data['collision_spheres'].pop('camera',None);p.write_text(yaml.safe_dump(data))
    p=asset/'curobo.yml';data=yaml.safe_load(p.read_text());kin=data['robot_cfg']['kinematics']
    kin['collision_link_names']=[x for x in kin['collision_link_names'] if x!='camera']
    kin['self_collision_buffer'].pop('camera',None);kin['self_collision_ignore'].pop('camera',None)
    for name,values in kin['self_collision_ignore'].items():kin['self_collision_ignore'][name]=[x for x in values if x!='camera']
    p.write_text(yaml.safe_dump(data))
    digest=hashlib.sha256((asset/'robot.urdf').read_bytes()).hexdigest()
    rig['generated'][side]['urdf_sha256']=digest;manifest['generated'][side]['urdf_sha256']=digest
rig['source_repository']='lab_archive:JAKA_Dual'
rig['simulated_camera_collision']='disabled: unsupported proxy shell overlapped gripper fingers; optical sensors remain simulated'
manifest['camera_collision']=rig['simulated_camera_collision']
rig_path.write_text(json.dumps(rig,indent=2)+'\n');(root/'manifests/lab_hardware_20261111.json').write_text(json.dumps(manifest,indent=2)+'\n')
p=root/'manifests/lab_collision_20261111.json';collision_manifest=json.loads(p.read_text())
for side in ('left','right'):
    data=yaml.safe_load((Path(rig['generated'][side]['asset'])/'collision.yml').read_text())
    collision_manifest[side]['total_spheres']=sum(len(v) for v in data['collision_spheres'].values())
    collision_manifest[side]['camera_proxy_collision_removed']=True
p.write_text(json.dumps(collision_manifest,indent=2)+'\n')
print(json.dumps({'camera_collision':rig['simulated_camera_collision'],'hashes':{k:v['urdf_sha256'] for k,v in rig['generated'].items()}}))
