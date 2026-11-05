"""Proprioceptive failure detection. Aperture is evidence, not proof of a grasp."""
import numpy as np


class EmptyGraspMonitor:
    def __init__(self, gripper_indices, dt=.04, confirmation_s=.16):
        self.indices=list(gripper_indices);self.dt=dt;self.confirmation_s=confirmation_s
        self.empty_time=np.zeros(len(self.indices))

    def update(self, requested, measured):
        requested=np.asarray(requested)[self.indices]
        measured=np.asarray(measured)[self.indices]
        # Calibrated only for this proxy gripper and 6 cm block. No simulator object state.
        # An almost fully closed aperture cannot contain this object.
        empty=(requested<.15)&(measured<.08)
        self.empty_time=np.where(empty,self.empty_time+self.dt,0.)
        return [i for i,t in enumerate(self.empty_time) if t>=self.confirmation_s]
