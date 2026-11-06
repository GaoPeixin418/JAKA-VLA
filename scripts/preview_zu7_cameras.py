"""Render a scene preview without moving either robot or calling a policy."""
import runpy
from pathlib import Path
import cv2

ROOT = Path('/data/jaka-vla')
api = runpy.run_path(str(ROOT / 'scripts/eval_v2.py'))
from jaka_vla.review_video import ReviewVideo

cfg, _ = api['load_task_args']({'task_name': 'handover_block', 'task_config': 'zu7_smoke', 'policy_name': 'SmolVLA'})
cfg.update(save_data=False, need_plan=False, eval_mode=True, render_freq=0, eval_video_save_dir=None)
task = api['class_decorator']('handover_block')
task.setup_demo(now_ep_num=2, seed=2, **cfg)
dest = ROOT / 'results/zu7_camera_preview_20261106'
dest.mkdir(exist_ok=False)
video = ReviewVideo(task, dest / 'preview.mp4', 25, 'CAMERA PREVIEW / no policy', 2)
obs = task.get_obs()
frame = video.capture(obs)
cv2.imwrite(str(dest / 'preview.png'), cv2.cvtColor(video.annotated(frame, 0., 'Camera framing check'), cv2.COLOR_RGB2BGR))
video.write(frame, 0., 'Camera framing check / no action', repeats=25)
video.close()
task.close_env(clear_cache=True)
print(dest, flush=True)
