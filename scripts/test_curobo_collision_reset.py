"""Regression: a reused slow self-collision buffer must clear cost and gradients."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch
import yaml
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.rollout.cost.self_collision_cost import SelfCollisionCost,SelfCollisionCostConfig

root=Path('/data/jaka-vla')
model=CudaRobotModel(RobotConfig.from_dict(yaml.safe_load((root/'repos/RoboTwin/assets/embodiments/jaka-zu7-lab-left/curobo.yml').read_text())).kinematics)
cfg=replace(model.self_collision_config,experimental_kernel=False)
rows=json.loads((root/'logs/lab_self_collision_20261111.log').read_text())
bad=next(x for x in rows if len(x['contacts'])>0)['q']
good=next(x for x in rows if len(x['contacts'])==0)['q']
cost=SelfCollisionCost(SelfCollisionCostConfig(weight=1.,tensor_args=TensorDeviceType(),self_collision_kin_config=cfg))
results=[]
for label,q in [('collision',bad),('free',good),('free',good),('collision',bad),('free',good)]:
    state=model.get_state(torch.tensor([q],device='cuda',dtype=torch.float32))
    spheres=state.link_spheres_tensor.reshape(1,1,-1,4).detach().clone().requires_grad_(True)
    value=cost(spheres);value.sum().backward()
    number=float(value.item());gradient=float(spheres.grad.abs().max().item())
    assert np.isfinite(number) and np.isfinite(gradient)
    if label=='free':assert number==0 and gradient==0,(label,number,gradient)
    else:assert number>0 and gradient>0,(label,number,gradient)
    results.append({'query':label,'cost':number,'gradient_abs_max':gradient})
print(json.dumps({'passed':True,'queries':results}),flush=True)
