"""Use the actual URDF tool-link frame for a fixed tool mount on Zu7."""
from pathlib import Path
p=Path('/data/jaka-vla/repos/RoboTwin/envs/robot/robot.py')
s=p.read_text()
if 'left_use_tool_link_pose' not in s:
    needle='        self.left_gripper_bias = left_embodiment_args["gripper_bias"]'
    s=s.replace(needle,'        self.left_use_tool_link_pose = left_embodiment_args.get("use_tool_link_pose", False)\n'+needle)
    needle='        self.right_gripper_bias = right_embodiment_args["gripper_bias"]'
    s=s.replace(needle,'        self.right_use_tool_link_pose = right_embodiment_args.get("use_tool_link_pose", False)\n'+needle)
    needle='        ee_pose = (self.left_ee.global_pose if arm_tag == "left" else self.right_ee.global_pose)'
    assert needle in s
    s=s.replace(needle,'''        joint = self.left_ee if arm_tag == "left" else self.right_ee
        use_link = self.left_use_tool_link_pose if arm_tag == "left" else self.right_use_tool_link_pose
        ee_pose = joint.child_link.get_pose() if use_link else joint.global_pose''')
    s=s.replace('        pose = self.left_ee.global_pose','        pose = self.left_ee.child_link.get_pose() if self.left_use_tool_link_pose else self.left_ee.global_pose')
    s=s.replace('        pose = self.right_ee.global_pose','        pose = self.right_ee.child_link.get_pose() if self.right_use_tool_link_pose else self.right_ee.global_pose')
    p.write_text(s)
print('Zu7 fixed tool mount uses its child-link frame; legacy frame semantics preserved.')
