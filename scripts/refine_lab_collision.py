"""Replace loose axial spheres with conservative filled voxel spheres for lab CAD."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import transforms3d as t3d
import yaml
from scipy.cluster.vq import kmeans2
ROOT=Path('/data/jaka-vla');report={}
for side in ('left','right'):
    asset=ROOT/f'repos/RoboTwin/assets/embodiments/jaka-zu7-lab-{side}'
    root=ET.parse(asset/'robot.urdf').getroot();data=yaml.safe_load((asset/'collision.yml').read_text())
    counts={}
    for i in range(1,7):
        name=f'Link_{i}';collision=root.find(f"link[@name='{name}']/collision")
        mesh=trimesh.load(collision.find('geometry/mesh').get('filename'),force='mesh')
        o=collision.find('origin')
        if o is not None:
            m=np.eye(4);m[:3,:3]=t3d.euler.euler2mat(*[float(x) for x in o.get('rpy','0 0 0').split()])
            m[:3,3]=[float(x) for x in o.get('xyz','0 0 0').split()];mesh.apply_transform(m)
        pitch=.005 if i==1 else .01;voxels=mesh.voxelized(pitch).fill();points=voxels.points
        # Each filled voxel is wholly covered by its cluster sphere. This bounds
        # volume as well as the surface while keeping the kernel sphere count low.
        # Link_1 has the smallest table clearance in the tilted archive base.
        # Finer clusters preserve the CAD volume without bridging free space
        # down into the table (24 clusters overreached by 4.77 mm at home).
        count={1:256,2:48,3:32,4:12,5:12,6:8}[i]
        centers,labels=kmeans2(points,min(count,len(points)),iter=20,minit='++',seed=42)
        fitted=[]
        for k,center in enumerate(centers):
            members=points[labels==k]
            if len(members):fitted.append({'center':center.tolist(),'radius':float(np.linalg.norm(members-center,axis=1).max()+np.sqrt(3)*pitch/2+.001)})
        data['collision_spheres'][name]=fitted
        counts[name]=len(fitted)
    # The fixed base is bolted to the table; it is not a moving collision body.
    data['collision_spheres'].pop('Link_0',None)
    total=sum(len(v) for v in data['collision_spheres'].values())
    if total>=1024:raise ValueError(f'cuRobo kernel supports fewer than 1024 spheres, got {total}')
    (asset/'collision.yml').write_text(yaml.safe_dump(data))
    config=yaml.safe_load((asset/'curobo.yml').read_text());kin=config['robot_cfg']['kinematics']
    kin['collision_link_names']=[n for n in kin['collision_link_names'] if n!='Link_0']
    kin['self_collision_ignore'].pop('Link_0',None);kin['self_collision_buffer'].pop('Link_0',None)
    (asset/'curobo.yml').write_text(yaml.safe_dump(config))
    report[side]={'voxel_pitch_m_by_link':{f'Link_{i}':.005 if i==1 else .01 for i in range(1,7)},'fit':'deterministic clusters enclosing every filled voxel','counts':counts,'total_spheres':total,
                  'fixed_base_excluded':'anchored table contact; opposite-arm base mesh remains an obstacle'}
(ROOT/'manifests/lab_collision_20261111.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
