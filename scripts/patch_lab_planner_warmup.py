"""Skip unused batch-pose warmup for the single-pose lab expert."""
from pathlib import Path
import yaml
root=Path('/data/jaka-vla')
path=root/'repos/RoboTwin/envs/robot/planner.py'
source=path.read_text()
needle='            self.frame_bias = yml_data["planner"]["frame_bias"]'
if 'self.project_plan_options =' not in source:
    if source.count(needle)!=1:raise RuntimeError('Unexpected planner frame config')
    source=source.replace(needle,needle+'\n            self.project_plan_options = yml_data.get("planner", {}).get("plan_options", {})')
    source=source.replace('MotionGenPlanConfig(max_attempts=10)','MotionGenPlanConfig(max_attempts=10, **self.project_plan_options)')
needle='                interpolation_dt=1 / 250,'
if '**yml_data.get("planner", {}).get("motion_gen_options", {})' not in source:
    if source.count(needle)!=2:raise RuntimeError('Unexpected motion generation configuration')
    source=source.replace(needle,needle+'\n                **yml_data.get("planner", {}).get("motion_gen_options", {}),')
old='            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)'
new='''            if not yml_data.get("planner", {}).get("skip_batch_warmup", False):
                self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)'''
if new not in source:
    if source.count(old)!=1:raise RuntimeError('Unexpected planner warmup source')
    source=source.replace(old,new)
path.write_text(source)
for side in ('left','right'):
    path=root/f'repos/RoboTwin/assets/embodiments/jaka-zu7-lab-{side}/curobo.yml'
    config=yaml.safe_load(path.read_text());config['planner']['skip_batch_warmup']=True
    config['planner']['plan_options']={'finetune_dt_scale':1.1,'finetune_attempts':8,'finetune_dt_decay':1.1,'time_dilation_factor':.5,'timeout':20.}
    config['planner'].setdefault('motion_gen_options',{}).update(maximum_trajectory_dt=.5)
    config['robot_cfg']['kinematics']['cspace']['velocity_scale']=1.0
    path.write_text(yaml.safe_dump(config))
print('Lab batch-pose warmup skipped; batch planner remains available.')
