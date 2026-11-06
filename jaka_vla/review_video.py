"""Observer-only cameras and streaming review recording; never policy inputs."""
import subprocess
from pathlib import Path

import cv2
import numpy as np
import sapien


class ReviewVideo:
    width, height = 1280, 800

    def __init__(self, task, path, fps, label, seed, intro_s=2., outro_s=3.):
        self.path = Path(path)
        self.fps, self.label, self.seed = fps, label, seed
        self.intro_s, self.outro_s = intro_s, outro_s
        self.frames = 0
        self.cameras = {}
        self.specs = {
            'overview': {'eye': [1.05, 1.15, 1.85], 'target': [0., -.12, 1.03],
                         'width': 960, 'height': 720, 'fovy_deg': 45.},
            'overhead': {'eye': [0., -.05, 2.05], 'target': [0., -.05, .74],
                         'width': 640, 'height': 480, 'fovy_deg': 50.},
        }
        for name, spec in self.specs.items():
            forward = np.asarray(spec['target']) - np.asarray(spec['eye'])
            forward /= np.linalg.norm(forward)
            up = np.array([0., 0., 1.]) if abs(forward[2]) < .99 else np.array([0., 1., 0.])
            left = np.cross(up, forward); left /= np.linalg.norm(left)
            transform = np.eye(4)
            transform[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
            transform[:3, 3] = spec['eye']
            camera = task.scene.add_camera('review_' + name, spec['width'], spec['height'],
                                           np.deg2rad(spec['fovy_deg']), .05, 20.)
            camera.entity.set_pose(sapien.Pose(transform))
            self.cameras[name] = camera
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{self.width}x{self.height}', '-r', str(fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '21',
            '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path)],
            stdin=subprocess.PIPE)
        self.closed = False

    def capture(self, obs):
        panels = {}
        for name, camera in self.cameras.items():
            camera.take_picture()
            panels[name] = (camera.get_picture('Color')[..., :3] * 255).clip(0, 255).astype(np.uint8)
        frame = np.full((self.height, self.width, 3), (17, 26, 40), dtype=np.uint8)
        frame[40:760, :960] = panels['overview']
        for i, (name, rgb) in enumerate([
            ('OVERHEAD', panels['overhead']),
            ('HEAD / policy input', obs['observation']['head_camera']['rgb']),
            ('LEFT WRIST / policy input', obs['observation']['left_camera']['rgb'])]):
            y = 40 + 240 * i
            frame[y:y+240, 960:] = cv2.resize(rgb, (320, 240))
            cv2.rectangle(frame, (960, y), (1279, y+24), (17, 26, 40), -1)
            cv2.putText(frame, name, (969, y+17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(frame, 'OVERVIEW / recording only', (14, 64), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 25, 30), 2)
        return frame

    def annotated(self, frame, sim_time, status):
        frame = frame.copy()
        frame[:40] = (17, 26, 40); frame[760:] = (17, 26, 40)
        cv2.putText(frame, f'{self.label} | seed {self.seed} | simulation {sim_time:.2f} s',
                    (14, 27), cv2.FONT_HERSHEY_SIMPLEX, .66, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, status, (14, 786), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 228, 161), 1, cv2.LINE_AA)
        return frame

    def write(self, frame, sim_time, status, repeats=1):
        data = self.annotated(frame, sim_time, status).tobytes()
        for _ in range(repeats):
            self.process.stdin.write(data)
            self.frames += 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.process.stdin.close()
            if self.process.wait(timeout=120):
                raise RuntimeError(f'Video encoder failed: {self.path}')

    def metadata(self):
        return {'file': self.path.name, 'fps': self.fps, 'frames': self.frames,
                'duration_s': self.frames / self.fps, 'resolution': [self.width, self.height],
                'intro_freeze_s': self.intro_s, 'outro_freeze_s': self.outro_s,
                'cameras': self.specs, 'observer_cameras_are_policy_inputs': False,
                'timing': '25 Hz simulation playback; inference pauses excluded; labelled intro/outro are frozen frames'}
