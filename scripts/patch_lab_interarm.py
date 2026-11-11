"""Use the opposite arm's own URDF when the lab arms have different geometries."""
from pathlib import Path
root=Path('/data/jaka-vla/repos/RoboTwin')
p=root/'envs/robot/planner.py';s=p.read_text()
old='root = ET.parse(Path(self.yml_path).parent / "robot.urdf").getroot()'
new='root = ET.parse(getattr(self, "other_arm_urdf_path", Path(self.yml_path).parent / "robot.urdf")).getroot()'
if old in s:s=s.replace(old,new);p.write_text(s)
elif new not in s:raise RuntimeError('Unrecognized planner source')
needle_failure='                res_result["status"] = "Fail"\n                return res_result'
if 'res_result["planner_status"]' not in s:
    if needle_failure not in s:raise RuntimeError('Unrecognized planner failure path')
    s=s.replace(needle_failure,'                res_result["status"] = "Fail"\n                res_result["planner_status"] = str(result.status)\n                print("PLANNER_FAILURE", result.status, flush=True)\n                return res_result')
p.write_text(s)
p=root/'envs/robot/robot.py';s=p.read_text()
needle_init='        self.need_topp = need_topp'
if 'self.project_inprocess_planners =' not in s:
    if needle_init not in s:raise RuntimeError('Unrecognized robot initialization')
    s=s.replace(needle_init,needle_init+'\n        self.project_inprocess_planners = left_embodiment_args.get("project_inprocess_planners", False) and right_embodiment_args.get("project_inprocess_planners", False)')
    s=s.replace('self.communication_flag = (abs_left_curobo_yml_path != abs_right_curobo_yml_path)',
                'self.communication_flag = (abs_left_curobo_yml_path != abs_right_curobo_yml_path) and not self.project_inprocess_planners')
needle_home='    def move_to_homestate(self):\n'
if '# Lab initialization: reset before the first physics step' not in s:
    if needle_home not in s:raise RuntimeError('Unrecognized home initialization')
    s=s.replace(needle_home,needle_home+'''        # Lab initialization: reset before the first physics step, never during execution.
        if self.project_inprocess_planners:
            for arm in ("left", "right"):
                entity = getattr(self, arm + "_entity")
                active = entity.get_active_joints()
                values = entity.get_qpos().copy()
                for j, v in zip(getattr(self, arm + "_arm_joints"), getattr(self, arm + "_homestate")):
                    values[active.index(j)] = v
                for j, multiplier, offset in getattr(self, arm + "_gripper"):
                    values[active.index(j)] = .05 * multiplier + offset
                entity.set_qpos(values)
                entity.set_qvel(np.zeros_like(values))
''')
needle='            self.right_planner.other_arm_entity = self.left_entity'
if 'self.left_planner.other_arm_urdf_path' not in s:
    if needle not in s:raise RuntimeError('Unrecognized robot source')
    s=s.replace(needle,needle+'\n            self.left_planner.other_arm_urdf_path = self.right_urdf_path\n            self.right_planner.other_arm_urdf_path = self.left_urdf_path')
p.write_text(s)
print('Opposite-arm collision models now follow the correct side.')
