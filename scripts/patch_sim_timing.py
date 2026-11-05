"""Add optional uniform simulation-time capture and preserve actual timestamps.

Native sampling remains available when jaka_uniform_sampling is false. Our project
config enables 25 Hz capture across motion-primitive boundaries at 250 Hz physics.
"""
from pathlib import Path

root = Path('/data/jaka-vla/repos/RoboTwin')
p = root / 'envs/_base_task.py'
s = p.read_text()
if 'def _jaka_step(' not in s:
    s = s.replace('        self.FRAME_IDX = 0\n', '''        self.FRAME_IDX = 0
        self._jaka_sim_steps = 0
        self._jaka_capture_active = False
        self._jaka_uniform_sampling = kwags.get("jaka_uniform_sampling", False)
''', 1)
    s = s.replace('self.scene.step()', 'self._jaka_step()')
    s = s.replace('    def get_obs(self):', '''    def _jaka_step(self):
        self.scene.step()
        self._jaka_sim_steps += 1
        if self._jaka_capture_active and self._jaka_sim_steps % self.save_freq == 0:
            self._take_picture(_jaka_force=True)

    def get_obs(self):''', 1)
    s = s.replace('        self.stage_success_tag = False\n', '''        self.stage_success_tag = False
        if self._jaka_uniform_sampling:
            self._jaka_sim_steps = 0
            self._jaka_capture_active = bool(self.save_data)
            if self._jaka_capture_active:
                self._take_picture(_jaka_force=True)
''', 1)
    s = s.replace('    def _take_picture(self):  # save data\n', '''    def _take_picture(self, _jaka_force=False):  # save data
        if self._jaka_uniform_sampling and not _jaka_force:
            return
''', 1)
    s = s.replace('        self.now_obs = deepcopy(pkl_dic)', '''        pkl_dic["sim_time"] = float(self._jaka_sim_steps * self.scene.get_timestep())
        self.now_obs = deepcopy(pkl_dic)''', 1)
    s = s.replace('frequency=int(self.save_freq or 15),', 'frequency=int(round(1.0 / (self.scene.get_timestep() * (self.save_freq or 15)))),')
    p.write_text(s)
if 'def _jaka_measured_state(' not in s:
    s = s.replace('    def get_obs(self):', '''    def _jaka_measured_state(self, arm):
        entity = getattr(self.robot, f"{arm}_entity")
        joints = getattr(self.robot, f"{arm}_arm_joints")
        active = entity.get_active_joints()
        qpos = entity.get_qpos()
        measured = [float(qpos[active.index(joint)]) for joint in joints]
        grip_joint, multiplier, offset = getattr(self.robot, f"{arm}_gripper")[0]
        grip_position = (float(qpos[active.index(grip_joint)]) - offset) / multiplier
        lo, hi = getattr(self.robot, f"{arm}_gripper_scale")
        measured.append(float(np.clip((grip_position - lo) / (hi - lo), 0.0, 1.0)))
        return measured

    def get_obs(self):''', 1)
    s = s.replace('        self.now_obs = deepcopy(pkl_dic)', '''        measured_left = self._jaka_measured_state("left")
        measured_right = self._jaka_measured_state("right")
        pkl_dic["joint_measured"] = {
            "left_arm": measured_left[:-1], "left_gripper": measured_left[-1],
            "right_arm": measured_right[:-1], "right_gripper": measured_right[-1],
            "vector": np.array(measured_left + measured_right),
        }
        self.now_obs = deepcopy(pkl_dic)''', 1)
    p.write_text(s)
p = root / 'envs/utils/pkl2hdf5.py'
s = p.read_text()
if 'observation_timestamps' not in s:
    s = s.replace('        state = f.create_group("state")', '''        if "sim_time" in data:
            timestamps = np.asarray(data["sim_time"], dtype=np.float64)
            f.create_dataset("observation_timestamps", data=timestamps[:-1])
            f.create_dataset("action_timestamps", data=timestamps[1:])
            f.attrs["sampling_note"] = "Measured simulation timestamps; validate spacing before training."
        state = f.create_group("state")''', 1)
    p.write_text(s)
old_video = 'images_to_video(np.array(data_list["observation"]["head_camera"]["rgb"]), out_path=video_path)'
new_video = 'images_to_video(np.array(data_list["observation"]["head_camera"]["rgb"]), out_path=video_path, fps=frequency)'
s = p.read_text().replace(old_video, new_video)
if 'commanded_state' not in s:
    s = s.replace('        action = f.create_group("action")', '''        action = f.create_group("action")
        commanded_state = f.create_group("commanded_state")
        measured = data.get("joint_measured", joints)
        f.attrs["state_semantics"] = "measured_qpos" if "joint_measured" in data else "command_targets_legacy"
        f.attrs["action_semantics"] = "next_sample_drive_targets"''', 1)
    s = s.replace('            state.create_dataset(target_name, data=values[:-1])', '''            state_values = _ensure_2d(measured[source_name])
            state.create_dataset(target_name, data=state_values[:-1])
            commanded_state.create_dataset(target_name, data=values[:-1])''', 1)
p.write_text(s)
print('Uniform sampling, measured proprioception, and commanded targets installed.')
