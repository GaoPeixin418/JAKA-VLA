"""Reset reused outputs before the pinned cuRobo slow self-collision kernel.

That kernel writes distance/gradients only on positive penetration. Without a
reset a colliding query poisons later collision-free queries with the old cost.
"""
from pathlib import Path
path=Path('/data/jaka-vla/repos/RoboTwin/envs/curobo/src/curobo/rollout/cost/self_collision_cost.py')
old='''        self.update_batch_size(robot_spheres)

        dist = SelfCollisionDistance.apply('''
new='''        self.update_batch_size(robot_spheres)
        # Project regression: the pinned slow kernel does not write a zero result
        # for collision-free queries and does not clear old gradient entries.
        if not self.self_collision_kin_config.experimental_kernel:
            self._out_distance.zero_()
            self._out_vec.zero_()
            self._sparse_sphere_idx.zero_()

        dist = SelfCollisionDistance.apply('''
source=path.read_text()
if new in source:print('Already applied')
elif source.count(old)==1:path.write_text(source.replace(old,new));print('Applied slow-kernel buffer reset')
else:raise RuntimeError('Unexpected pinned cuRobo source; not modified')
