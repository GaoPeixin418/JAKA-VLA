"""Construct each arm's GPU planner when that arm first requests a path."""
class LazyPlanner:
    def __init__(self,factory,*args,**kwargs):
        self._factory=factory;self._args=args;self._kwargs=kwargs;self._instance=None

    def _get(self):
        if self._instance is None:
            import os
            if os.environ.get('JAKA_REUSE_PLANNER')=='1':
                from jaka_vla.planner_cache import get_planner
                self._instance=get_planner(self._factory,self._args,self._kwargs)
            else:
                self._instance=self._factory(*self._args,**self._kwargs)
            for name in ('other_arm_entity','other_arm_urdf_path'):
                if name in self.__dict__:setattr(self._instance,name,self.__dict__[name])
        return self._instance

    def __getattr__(self,name):
        return getattr(self._get(),name)

    def plan_grippers(self,now_val,target_val):
        # The upstream gripper interpolation has no dependence on GPU state.
        return self._factory.plan_grippers(self,now_val,target_val)

    def __setattr__(self,name,value):
        object.__setattr__(self,name,value)
        if name in ('other_arm_entity','other_arm_urdf_path') and self.__dict__.get('_instance') is not None:
            setattr(self._instance,name,value)
