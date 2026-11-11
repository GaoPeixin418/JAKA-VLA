"""Enable a separately validated timing fallback only for the archive rig."""
from pathlib import Path
import yaml
root=Path('/data/jaka-vla');path=root/'repos/RoboTwin/envs/robot/planner.py';source=path.read_text()
needle='            self.project_plan_options = yml_data.get("planner", {}).get("plan_options", {})'
if 'self.project_retime_fallback =' not in source:
    if source.count(needle)!=1:raise RuntimeError('Planner config mismatch')
    source=source.replace(needle,needle+'\n            self.project_retime_fallback = yml_data.get("planner", {}).get("retime_fallback", False)')
old='''            result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)

            # output'''
new='''            result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)
            if self.project_retime_fallback and not result.success.item() and "FINETUNE_TRAJOPT_FAIL" in str(result.status):
                # Keep geometric collision/joint checks. Timing is solved separately
                # and every resampled 250 Hz state is checked again before execution.
                candidate = result.debug_info.get("project_geometric_candidate") if result.debug_info else None
                print("GEOMETRY_DEBUG", list(result.debug_info or {}), None if candidate is None else {k: v.detach().cpu().tolist() for k,v in candidate.items() if k != "position"}, flush=True)
                if candidate is not None:
                    from types import SimpleNamespace
                    index = 0
                    positions = candidate["position"]
                    knots = positions.reshape(-1, positions.shape[-2], 6)[index].clone()
                    geometric = SimpleNamespace(interpolated_plan=SimpleNamespace(position=knots))
                    from jaka_vla.lab_retiming import retime_and_validate
                    try:
                        retimed = retime_and_validate(self, geometric, goal=goal_pose_of_ee, start=start_joint_states.position, hold_orientation=constraint_pose is not None)
                        print("TOPPRA_RETIMED", retimed["retiming"], flush=True)
                        return retimed
                    except (ValueError, RuntimeError) as error:
                        print("TOPPRA_REJECTED", str(error), flush=True)

            # output'''
start=source.find('            if self.project_retime_fallback and not result.success.item()')
end=source.find('            # output',start)
if start>=0 and end>start:
    fallback=new.split('            if self.project_retime_fallback')[1].split('            # output')[0]
    source=source[:start]+'            if self.project_retime_fallback'+fallback+source[end:]
    path.write_text(source)
elif new not in source:
    if source.count(old)!=1:raise RuntimeError('Planner result mismatch')
    path.write_text(source.replace(old,new))
for side in ('left','right'):
    path=root/f'repos/RoboTwin/assets/embodiments/jaka-zu7-lab-{side}/curobo.yml'
    config=yaml.safe_load(path.read_text());config['planner']['retime_fallback']=True
    config['planner'].setdefault('motion_gen_options',{})['store_debug_in_result']=True
    path.write_text(yaml.safe_dump(config))
path=root/'repos/RoboTwin/envs/curobo/src/curobo/wrap/reacher/motion_gen.py'
source=path.read_text()
needle='                result.debug_info["trajopt_result"] = traj_result'
addition='''
                # Project fallback needs a snapshot before the finetune solver reuses buffers.
                result.debug_info["project_geometric_candidate"] = {
                    "position": traj_result.solution.position.detach().clone(),
                    "success": traj_result.success.detach().clone(),
                    "position_error": traj_result.position_error.detach().clone(),
                    "rotation_error": traj_result.rotation_error.detach().clone(),
                }'''
import re
source=re.sub(r'\n                # Project fallback needs a snapshot.*?\n                }', '', source, flags=re.S)
if source.count(needle)<1:raise RuntimeError('Unexpected debug trajectory source')
path.write_text(source.replace(needle,needle+addition,1))
print('Archive-rig fallback: collision-checked geometry, bounded TOPPRA, 250 Hz collision recheck.')
