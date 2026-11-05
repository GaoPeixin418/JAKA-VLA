"""Configure expert grasp heights for Zu7 wrist/table clearance, retaining the task goal."""
from pathlib import Path
p=Path('/data/jaka-vla/repos/RoboTwin/envs/handover_block.py')
s=p.read_text()
if 'zu7_grasp_heights_m' not in s:
    a='    def setup_demo(self, **kwags):\n'
    assert a in s
    s=s.replace(a,a+'        self.zu7_grasp_heights_m = kwags.get("zu7_grasp_heights_m")\n',1)
    a='        self.add_prohibit_area(self.box, padding=0.1)\n'
    b='''        # New rig only: keep the receiving wrist above the table when placing.
        # Expert annotation, not an oracle observation exposed to the learned policy.
        if self.zu7_grasp_heights_m is not None:
            for index, point in enumerate(self.box.config["contact_points_pose"]):
                point[2][3] = self.zu7_grasp_heights_m[index // 4] / self.box.config["scale"][2]

'''
    assert a in s;s=s.replace(a,b+a,1)
if 'zu7_receiving_platform_height_m' not in s:
    a='    def setup_demo(self, **kwags):\n'
    s=s.replace(a,a+'        self.zu7_receiving_platform_height_m = kwags.get("zu7_receiving_platform_height_m")\n',1)
    a='        self.target_box = create_box(\n'
    b='''        target_half_height = 0.005
        if self.zu7_receiving_platform_height_m is not None:
            target_half_height = self.zu7_receiving_platform_height_m / 2
            rand_pos = sapien.Pose([rand_pos.p[0], rand_pos.p[1], 0.74 + target_half_height], rand_pos.q)

'''
    assert a in s;s=s.replace(a,b+a,1)
    s=s.replace('half_size=(0.05, 0.05, 0.005),','half_size=(0.05, 0.05, target_half_height),',1)
if 'zu7_sequential_return' not in s:
    a='    def setup_demo(self, **kwags):\n'
    s=s.replace(a,a+'        self.zu7_sequential_return = kwags.get("zu7_sequential_return", False)\n',1)
    a='        self.move(\n            self.back_to_origin(grasp_arm_tag),\n'
    b='''        if self.zu7_sequential_return:
            self.move(self.back_to_origin(grasp_arm_tag))
        return_action = () if self.zu7_sequential_return else (self.back_to_origin(grasp_arm_tag),)
        self.move(
            *return_action,
'''
    assert a in s;s=s.replace(a,b,1)
p.write_text(s)
print('Optional Zu7 expert grasp heights installed.')
