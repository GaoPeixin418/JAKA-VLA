import copy
from pathlib import Path
from types import SimpleNamespace, ModuleType
import tempfile
import unittest
from unittest.mock import patch
from jaka_vla import planner_cache


class PlannerCacheIsolation(unittest.TestCase):
    def test_reuse_clears_obstacles_and_entities_and_invalidates_changed_geometry(self):
        class Motion:
            def __init__(self):self.resets=0
            def reset(self,reset_seed=True):self.resets+=1;assert reset_seed
            def update_world(self,world):self.world=copy.deepcopy(world)
        class Planner:
            def __init__(self,**kwargs):
                self.project_world={'cuboid':{'table':{'dims':[1,2,3]}}}
                self.motion_gen=Motion();self.motion_gen_batch=Motion();self.other_arm_entity=None
        module=ModuleType('curobo.geom.types');module.WorldConfig=SimpleNamespace(from_dict=lambda value:value)
        planner_cache._CACHE.clear()
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'curobo.geom.types':module}):
            folder=Path(tmp)
            for name in ('curobo.yml','robot.urdf','collision.yml'):(folder/name).write_text(name)
            kw={'robot_origion_pose':SimpleNamespace(p=[0,0,0],q=[1,0,0,0]),'active_joints_name':['j1'],
                'all_joints':['j1'],'yml_path':str(folder/'curobo.yml')}
            first=planner_cache.get_planner(Planner,(),kw)
            first.project_world['cuboid']['old_target']={};first.other_arm_entity=object();first._other_arm_geometries=['stale']
            planner_cache.release_scene_entities()
            self.assertIsNone(first.other_arm_entity);self.assertFalse(hasattr(first,'_other_arm_geometries'))
            second=planner_cache.get_planner(Planner,(),kw)
            self.assertIs(first,second);self.assertNotIn('old_target',second.project_world['cuboid'])
            self.assertNotIn('old_target',second.motion_gen.world['cuboid']);self.assertEqual(second.motion_gen.resets,2)
            (folder/'robot.urdf').write_text('changed geometry')
            third=planner_cache.get_planner(Planner,(),kw)
            self.assertIsNot(first,third)
        planner_cache._CACHE.clear()


if __name__=='__main__':unittest.main()
