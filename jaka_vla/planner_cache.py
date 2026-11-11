"""Optional sequential planner reuse; no scene entities or obstacle state are reused."""
from copy import deepcopy
import hashlib
from pathlib import Path

_CACHE = {}


def get_planner(factory, args, kwargs):
    path=Path(kwargs.get('yml_path',args[3] if len(args)>3 else ''))
    signature=[]
    for name in (path.name,'robot.urdf','collision.yml'):
        signature.append(hashlib.sha256((path.parent/name).read_bytes()).hexdigest())
    pose=kwargs.get('robot_origion_pose',args[0] if len(args)>0 else None)
    names=kwargs.get('active_joints_name',args[1] if len(args)>1 else None)
    all_joints=kwargs.get('all_joints',args[2] if len(args)>2 else None)
    key=(str(path.resolve()),tuple(signature),tuple(map(float,[*pose.p,*pose.q])),tuple(names),tuple(all_joints))
    reused=key in _CACHE
    if not reused:
        instance=factory(*args,**kwargs)
        _CACHE[key]=(instance,deepcopy(instance.project_world))
    instance,world=_CACHE[key]
    instance.project_world=deepcopy(world)
    instance.other_arm_entity=None
    for mg in (instance.motion_gen,instance.motion_gen_batch):
        mg.reset(reset_seed=True)
        mg.update_world(__import__('curobo.geom.types',fromlist=['WorldConfig']).WorldConfig.from_dict(instance.project_world))
    print('PROJECT_PLANNER_CACHE',{'reused':reused,'entries':len(_CACHE)},flush=True)
    return instance


def release_scene_entities():
    # The current task can now be garbage collected after SAPIEN closes it.
    for instance,_ in _CACHE.values():
        instance.other_arm_entity=None
        if hasattr(instance,'_other_arm_geometries'):del instance._other_arm_geometries
