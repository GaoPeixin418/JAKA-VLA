from pathlib import Path
import os,sys,json,cv2
root=Path('/data/jaka-vla');robo=root/'repos/RoboTwin';sys.path[:0]=[str(root),str(robo)];os.chdir(robo)
from scripts.eval_policy_xpolicylab import load_task_args,class_decorator
from jaka_vla.lab_scene import install_scene,scene_spec
from jaka_vla.review_video import ReviewVideo
cfg,_=load_task_args({'task_name':'handover_block','task_config':'zu7_lab_smoke','policy_name':'SmolVLA'})
cfg.update(save_data=False,need_plan=False,eval_mode=True,render_freq=0,eval_video_save_dir=None)
t=class_decorator('handover_block');install_scene(t,scene_spec(0,0));t.setup_demo(now_ep_num=0,seed=60000,**cfg)
obs=t.get_obs();cv2.imwrite(str(root/'results/lab_camera_v3_initial.png'),cv2.cvtColor(obs['observation']['head_camera']['rgb'],cv2.COLOR_RGB2BGR))
t.close_env(clear_cache=True)
