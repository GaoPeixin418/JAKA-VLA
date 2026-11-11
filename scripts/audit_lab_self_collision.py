"""Identify conservative sphere pairs behind read-only IK collision rejection."""
from pathlib import Path
exec(Path('/data/jaka-vla/scripts/diagnose_lab_ik.py').read_text().split('import argparse')[0])
report=json.loads((ROOT/'logs/lab_ik_20261111.log').read_text())
ignore=yaml.safe_load((asset/'curobo.yml').read_text())['robot_cfg']['kinematics']['self_collision_ignore']
checked=[]
for row in report['candidates']:
    for solution in row['solutions']:
        poses=fk(solution['q']);spheres={}
        for name,values in collision.items():
            m=poses[name]
            spheres[name]=np.array([np.r_[m[:3,:3]@v['center']+m[:3,3],v['radius']+.002] for v in values])
        contacts=[]
        for i,(a,x) in enumerate(spheres.items()):
            for b,y in list(spheres.items())[i+1:]:
                if b in ignore.get(a,[]) or a in ignore.get(b,[]):continue
                gap=np.linalg.norm(x[:,None,:3]-y[None,:,:3],axis=-1)-x[:,None,3]-y[None,:,3]
                if gap.min()<0:contacts.append({'pair':[a,b],'sphere_gap_m':float(gap.min())})
        checked.append({'yaw':row['yaw'],'q':solution['q'],'contacts':sorted(contacts,key=lambda c:c['sphere_gap_m'])})
print(json.dumps(checked,indent=2),flush=True)
