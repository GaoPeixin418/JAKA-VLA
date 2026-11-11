import copy,json,unittest
from pathlib import Path
from jaka_vla.identity import physical_rig_hash


class RigIdentityTest(unittest.TestCase):
    def test_mount_change_rejected_but_controller_ablation_allowed(self):
        rig=json.loads((Path(__file__).resolve().parents[1]/'configs/robots/dual_zu7.json').read_text())
        changed=copy.deepcopy(rig);changed['joint_velocity_limit_rad_s']=1.5
        self.assertEqual(physical_rig_hash(rig),physical_rig_hash(changed))
        changed['mounting']['base_separation_m']+=.05
        self.assertNotEqual(physical_rig_hash(rig),physical_rig_hash(changed))

    def test_camera_geometry_changes_identity(self):
        rig=json.loads((Path(__file__).resolve().parents[1]/'configs/robots/dual_zu7.json').read_text())
        changed=copy.deepcopy(rig);changed['cameras']={'external':{'position':[0,1,2]}}
        self.assertNotEqual(physical_rig_hash(rig),physical_rig_hash(changed))
        moved=copy.deepcopy(changed);moved['cameras']['external']['position'][0]=.1
        self.assertNotEqual(physical_rig_hash(changed),physical_rig_hash(moved))


if __name__=='__main__':unittest.main()
