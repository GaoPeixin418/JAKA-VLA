from pathlib import Path
exec(Path('/data/jaka-vla/scripts/diagnose_lab_ik.py').read_text().split('import argparse')[0])
import time
# Simulation workspace screening only; no motion is executed.
rng=np.random.default_rng(20261111);home=np.array(rig['home_joint_rad']['left']);out=[]
for px,py in [(-.6,.15),(-.6,.32),(-.42,.32),(-.28,.23),(-.1,.33),(.12,.3)]:
 goal_rotation=Rotation.from_matrix(np.array([[0,1,0],[0,0,-1],[-1,0,0]]))
 def error(q,z):
  pose=fk(q)['tool'];return np.r_[3*(pose[:3,3]-[px,py,z]),(goal_rotation.inv()*Rotation.from_matrix(pose[:3,:3])).as_rotvec()]
 paths=[]
 for start in [home,*rng.uniform(lower,upper,(31,6))]:
  result=least_squares(error,start,args=(.934,),bounds=(lower,upper),max_nfev=120)
  if np.linalg.norm(result.fun)>1e-4 or clearance(result.x)[0]<0:continue
  q=result.x;end=q.copy();valid=True
  for z in np.linspace(.934,1.054,25)[1:]:
   result=least_squares(error,q,args=(z,),bounds=(lower,upper),max_nfev=60);q=result.x
   if np.linalg.norm(result.fun)>1e-4 or clearance(q)[0]<0:valid=False;break
  if valid:paths.append({'grasp':end.tolist(),'pre':q.tolist()});break
 row={'xyz':[px,py,.934],'vertical_path_found':bool(paths),'paths':paths};out.append(row);print(json.dumps(row),flush=True)
Path('/data/jaka-vla/results/lab_workspace_screen_20261111.json').write_text(json.dumps(out,indent=2)+'\n')
