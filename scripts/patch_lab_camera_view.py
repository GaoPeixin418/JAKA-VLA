"""Center the simulated external RGB view on the archive-table workspace."""
from pathlib import Path
import json,yaml
root=Path('/data/jaka-vla');p=root/'configs/robots/dual_zu7_lab.json';rig=json.loads(p.read_text())
head={'name':'head_camera','type':'D435','position':[-.25,.85,1.65],'forward':[0,-.65,-.85],'left':[1,0,0]}
for side in ('left','right'):
    path=Path(rig['generated'][side]['asset'])/'config.yml'
    config=yaml.safe_load(path.read_text());config['static_camera_list']=[head];path.write_text(yaml.safe_dump(config,sort_keys=False))
rig['cameras']={'external':head,'wrist':'RoboTwin simulated link-mounted RGB cameras',
                'evidence_status':'simulation configuration; not calibrated to archived Kinect or real wrist cameras'}
p.write_text(json.dumps(rig,indent=2)+'\n')
print(json.dumps(rig['cameras']))
