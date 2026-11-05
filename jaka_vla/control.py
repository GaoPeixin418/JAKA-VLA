"""Action continuity and time-domain command limits, shared across embodiments."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ActionLayout:
    arm_dims: tuple
    ee_dims: tuple

    @property
    def size(self): return sum(self.arm_dims)+sum(self.ee_dims)

    @property
    def arm_slices(self):
        result=[]; offset=0
        for arm,ee in zip(self.arm_dims,self.ee_dims):
            result.append(slice(offset,offset+arm));offset+=arm+ee
        return result

    @property
    def gripper_indices(self):
        result=[];offset=0
        for arm,ee in zip(self.arm_dims,self.ee_dims):
            if ee!=1: raise ValueError('This controller requires one scalar aperture per gripper')
            result.append(offset+arm);offset+=arm+ee
        return result

    @property
    def joint_indices(self): return np.concatenate([np.arange(s.start,s.stop) for s in self.arm_slices])


def validate_vector(value, size, label='action'):
    vector=np.asarray(value,dtype=np.float64)
    if vector.shape!=(size,) or not np.isfinite(vector).all():
        raise ValueError(f'{label}: expected finite shape {(size,)}, got {vector.shape}')
    return vector.copy()


class ChunkContinuity:
    """Blend the old prediction for the SAME future timestamps into the new prefix.

    Old predictions have already been advanced by exactly the executed count.
    No averaging of actions that refer to different simulation times.
    """
    def __init__(self, layout, blend_steps=4):
        self.layout=layout; self.blend_steps=blend_steps; self.remaining=None

    def reset(self): self.remaining=None

    def replace(self, new):
        new=np.asarray(new,dtype=np.float64).copy()
        if new.ndim!=2 or new.shape[1]!=self.layout.size or not np.isfinite(new).all():
            raise ValueError('Invalid action chunk')
        raw=new.copy(); n=0
        if self.remaining is not None:
            n=min(len(self.remaining),len(new),self.blend_steps)
            if n:
                # New observations retain at least half the weight; grippers use new targets.
                weights=np.linspace(.5,1,n)[:,None]
                idx=self.layout.joint_indices
                new[:n,idx]=weights*new[:n,idx]+(1-weights)*self.remaining[:n,idx]
        self.remaining=new
        return raw,new.copy(),n

    def pop(self):
        if self.remaining is None or len(self.remaining)==0: raise RuntimeError('Action queue exhausted')
        value=self.remaining[0].copy();self.remaining=self.remaining[1:]
        return value


class BoundedJointServo:
    """Critically damped target filter at PHYSICS frequency with rate/acceleration caps.

    Outputs consistent position AND velocity drive targets. This is a command filter,
    not a guarantee on actual physical joint velocity; measured tracking is logged.
    """
    def __init__(self, initial, limits, layout, dt=.004, vmax=1.5, amax=8., gripper_rate=2., omega=28.):
        self.layout=layout; self.dt=float(dt);self.q=validate_vector(initial,layout.size,'initial')
        self.v=np.zeros(layout.size);self.target=self.q.copy()
        self.limits=np.asarray(limits,dtype=float)
        if self.limits.shape!=(layout.size,2) or np.any(self.limits[:,0]>=self.limits[:,1]): raise ValueError('Invalid limits')
        if min(dt,vmax,amax,gripper_rate,omega)<=0: raise ValueError('Controller limits must be positive')
        self.vmax=np.full(layout.size,vmax);self.amax=np.full(layout.size,amax)
        self.vmax[layout.gripper_indices]=gripper_rate
        self.amax[layout.gripper_indices]=gripper_rate*10
        self.omega=omega;self.hard_limit_events=0

    def set_target(self, target):
        raw=validate_vector(target,self.layout.size)
        self.target=np.clip(raw,self.limits[:,0],self.limits[:,1])
        return raw,self.target.copy()

    def step(self):
        acceleration=self.omega**2*(self.target-self.q)-2*self.omega*self.v
        acceleration=np.clip(acceleration,-self.amax,self.amax)
        new_v=np.clip(self.v+acceleration*self.dt,-self.vmax,self.vmax)
        new_q=self.q+new_v*self.dt
        clipped=np.clip(new_q,self.limits[:,0],self.limits[:,1])
        hits=clipped!=new_q
        self.hard_limit_events+=int(hits.sum())
        new_v[hits]=0
        self.q=clipped;self.v=new_v
        return self.q.copy(),self.v.copy()
