"""Add stationary opposite-arm geometry to each Zu7 planning world."""
from pathlib import Path
ROOT=Path('/data/jaka-vla/repos/RoboTwin')
p=ROOT/'envs/robot/planner.py';s=p.read_text()
if 'def refresh_other_arm_world' not in s:
    needle='            motion_gen_config = MotionGenConfig.load_from_robot_config(\n'
    s=s.replace(needle,'            self.project_world = world_config\n            self.other_arm_entity = None\n            self.other_arm_enabled = yml_data.get("planner", {}).get("other_arm_collision", False)\n'+needle,1)
    s=s.replace('                interpolation_dt=1 / 250,','                collision_cache={"obb": 16, "mesh": 16} if self.other_arm_enabled else None,\n                interpolation_dt=1 / 250,',2)
    needle='        def plan_path(\n'
    method='''        def refresh_other_arm_world(self):
            if not self.other_arm_enabled or self.other_arm_entity is None:
                return
            import sapien
            import xml.etree.ElementTree as ET
            from pathlib import Path
            from copy import deepcopy
            from curobo.geom.types import WorldConfig
            if not hasattr(self, "_other_arm_geometries"):
                root = ET.parse(Path(self.yml_path).parent / "robot.urdf").getroot()
                self._other_arm_geometries = []
                for link in root.findall("link"):
                    for index, collision in enumerate(link.findall("collision")):
                        origin = collision.find("origin")
                        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0,0,0]
                        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0,0,0]
                        pose = sapien.Pose(xyz, t3d.euler.euler2quat(*rpy))
                        self._other_arm_geometries.append((link.get("name"), index, pose, collision.find("geometry")))
            links = {link.get_name(): link for link in self.other_arm_entity.get_links()}
            world = deepcopy(self.project_world)
            world["mesh"] = {}
            for name, index, origin, geometry in self._other_arm_geometries:
                pose = self.robot_origion_pose.inv() * links[name].get_pose() * origin
                value = {"pose": list(pose.p) + list(pose.q)}
                key = "other_" + name + "_" + str(index)
                mesh, box = geometry.find("mesh"), geometry.find("box")
                if mesh is not None:
                    value["file_path"] = mesh.get("filename")
                    value["scale"] = [float(x) for x in mesh.get("scale", "1 1 1").split()]
                    world["mesh"][key] = value
                elif box is not None:
                    value["dims"] = [float(x) for x in box.get("size").split()]
                    world["cuboid"][key] = value
                else:
                    raise ValueError("Unsupported opposite-arm collision geometry")
            config = WorldConfig.from_dict(world)
            self.motion_gen.update_world(config)
            self.motion_gen_batch.update_world(config)

'''
    assert needle in s;s=s.replace(needle,method+needle,1)
    s=s.replace('            world_base_pose = np.concatenate([','            self.refresh_other_arm_world()\n            world_base_pose = np.concatenate([',2)
    p.write_text(s)
p=ROOT/'envs/robot/robot.py';s=p.read_text()
if 'left_planner.other_arm_entity' not in s:
    needle='        else:\n            self.left_conn, left_child_conn = mp.Pipe()'
    insert='''            # Project adapter: each expert move treats the stationary other arm as an obstacle.
            self.left_planner.other_arm_entity = self.right_entity
            self.right_planner.other_arm_entity = self.left_entity
'''
    assert needle in s;s=s.replace(needle,insert+needle,1);p.write_text(s)
print('Opposite-arm collision geometry installed; opt-in per embodiment.')
