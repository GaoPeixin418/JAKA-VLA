"""Allow explicit world-frame table geometry for the new Zu7 embodiment only."""
from pathlib import Path
p=Path('/data/jaka-vla/repos/RoboTwin/envs/robot/planner.py')
s=p.read_text()
needle='            motion_gen_config = MotionGenConfig.load_from_robot_config(\n'
insert='''            # Project adapter: transform the configured world table into each arm base.
            if "world_table" in yml_data.get("planner", {}):
                import sapien
                table = yml_data["planner"]["world_table"]
                local = self.robot_origion_pose.inv() * sapien.Pose(table["position"], table["quaternion"])
                world_config = {"cuboid": {"table": {"dims": table["dims"], "pose": list(local.p) + list(local.q)}}}
'''
if '# Project adapter: transform the configured world table' not in s:
    assert needle in s
    s=s.replace(needle,insert+needle,1)
    p.write_text(s)
print('Explicit per-base table transform installed; default embodiments unchanged.')
