"""Avoid warming unused arm planners during collection and servo-only replay."""
from pathlib import Path
p=Path('/data/jaka-vla/repos/RoboTwin/envs/robot/robot.py');source=p.read_text()
needle='''        if not self.communication_flag:
            self.left_planner = CuroboPlanner('''
new='''        if not self.communication_flag:
            factory = CuroboPlanner
            if self.project_inprocess_planners:
                from jaka_vla.lazy_planner import LazyPlanner
                factory = lambda *a, **kw: LazyPlanner(CuroboPlanner, *a, **kw)
            self.left_planner = factory('''
if new not in source:
    if source.count(needle)!=1:raise RuntimeError('Unexpected planner constructor')
    source=source.replace(needle,new)
    source=source.replace('            self.right_planner = CuroboPlanner(', '            self.right_planner = factory(')
    p.write_text(source)
print('Lab arm planners now initialize only for a requested arm path.')
