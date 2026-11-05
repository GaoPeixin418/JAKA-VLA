"""Check an actual headless SAPIEN RGB render, not only renderer creation."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import sapien.core as sapien

root = Path('/data/jaka-vla')
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.set_timestep(1 / 250)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[0.9, 0.1, 0.1])
actor = builder.build_static(name='red_box')
camera = scene.add_camera('check', 320, 240, 1.0, 0.01, 10)
camera.set_pose(sapien.Pose([-0.8, 0, 0]))
scene.step()
scene.update_render()
camera.take_picture()
rgb = np.clip(camera.get_picture('Color')[..., :3] * 255, 0, 255).astype(np.uint8)
assert rgb.shape == (240, 320, 3)
assert np.ptp(rgb) > 20, 'Rendered image is blank'
out = root / 'results/render_smoke.png'
Image.fromarray(rgb).save(out)
print(json.dumps({'render_ok': True, 'rgb_shape': list(rgb.shape), 'output': str(out)}))
