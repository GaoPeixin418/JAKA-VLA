"""Cross-check cuRobo sphere FK against independent URDF transforms."""
from pathlib import Path
exec(Path('/data/jaka-vla/scripts/diagnose_lab_ik.py').read_text().split('import argparse')[0])
import torch
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.rollout.cost.self_collision_cost import SelfCollisionCost,SelfCollisionCostConfig
from dataclasses import replace
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig
from curobo.types.robot import JointState
config=RobotConfig.from_dict(yaml.safe_load((asset/'curobo.yml').read_text()))
model=CudaRobotModel(config.kinematics);kin=model.kinematics_config
mg=MotionGen(MotionGenConfig.load_from_robot_config(str(asset/'curobo.yml'),{'cuboid':{'far':{'dims':[.01,.01,.01],'pose':[10,10,10,1,0,0,0]}}},interpolation_dt=.004,num_trajopt_seeds=1))
inv={i:n for n,i in kin.link_name_to_idx_map.items()}
cpu=json.loads((ROOT/'logs/lab_self_collision_20261111.log').read_text())
report=[]
actual=np.load(ROOT/'results/lab_actual_curobo_spheres.npz')
for row in [{'q':actual['q'].tolist()},*[x for x in cpu if not x['contacts']][:3]]:
    state=model.get_state(torch.tensor([row['q']],device='cuda',dtype=torch.float32))
    gpu=state.link_spheres_tensor[0].cpu().numpy()
    poses=fk(row['q']);reference=[]
    for name,values in collision.items():
        m=np.linalg.inv(base)@poses[name]
        reference.extend([np.r_[m[:3,:3]@v['center']+m[:3,3],v['radius']+.002] for v in values])
    reference=np.array(reference)
    nearest=np.linalg.norm(gpu[:,None,:3]-reference[None,:,:3],axis=-1).min(axis=1)
    idx=kin.link_sphere_idx_map.cpu().numpy()
    dist=np.linalg.norm(gpu[:,None,:3]-gpu[None,:,:3],axis=-1)-gpu[:,None,3]-gpu[None,:,3]
    self_config=model.self_collision_config
    matrix=self_config.collision_matrix.cpu().numpy()
    pairs=np.argwhere((matrix>0)&(dist<0))
    costs={}
    for experimental in [self_config.experimental_kernel]:
        cost=SelfCollisionCost(SelfCollisionCostConfig(weight=1.,tensor_args=TensorDeviceType(),self_collision_kin_config=replace(self_config,experimental_kernel=experimental)))
        costs[str(experimental)]=float(cost(state.link_spheres_tensor.reshape(1,1,-1,4)).item())
    js=JointState.from_position(torch.tensor([row['q']],device='cuda',dtype=torch.float32),joint_names=model.joint_names)
    ms=mg.compute_kinematics(js)
    motion_cost=float(mg.rollout_fn.robot_self_collision_constraint(ms.robot_spheres.reshape(1,1,-1,4)).item())
    motion_valid,motion_status=mg.check_start_state(js)
    report.append({'q':row['q'],'max_sphere_fk_error_m':float(nearest.max()),
                   'actual_scene_comparison':{'max_sphere_difference':float(np.abs(actual['spheres'].reshape(-1,4)-gpu).max()),
                                              'matrix_difference_count':int(np.sum(actual['self_matrix']!=matrix)),
                                              'max_offset_difference':float(np.abs(actual['self_offset']-self_config.offset.cpu().numpy()).max())} if len(report)==0 else None,
                   'self_costs_by_experimental_kernel':costs,
                   'motion_gen_cost':motion_cost,'motion_gen_valid':motion_valid,'motion_gen_status':str(motion_status),
                   'motion_sphere_difference':float(np.abs(ms.robot_spheres.reshape(-1,4).cpu().numpy()-gpu).max()),
                   'matrix_shape':matrix.shape,'collision_matrix_values':np.unique(matrix).tolist(),
                   'pairs':[{'a':inv[idx[i]],'b':inv[idx[j]],'i':int(i),'j':int(j),'gap':float(dist[i,j])} for i,j in pairs[:40]]})
print(json.dumps(report,indent=2),flush=True)
