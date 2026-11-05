import unittest
import numpy as np
from jaka_vla.control import ActionLayout, BoundedJointServo, ChunkContinuity


class ControlTest(unittest.TestCase):
    def setUp(self): self.layout=ActionLayout((6,6),(1,1))

    def test_reversal_rate_acceleration_and_convergence(self):
        limits=np.tile([-3.,3.],(14,1));limits[[6,13]]=[0,1]
        servo=BoundedJointServo(np.zeros(14),limits,self.layout)
        last_v=servo.v.copy(); last_q=servo.q.copy()
        for i in range(2000):
            target=np.zeros(14);target[0]=1. if i<500 else -.5;target[6]=1.
            servo.set_target(target);q,v=servo.step()
            self.assertLessEqual(np.max(np.abs(v)-servo.vmax),1e-9)
            self.assertLessEqual(np.max(np.abs(v-last_v)-servo.amax*servo.dt),1e-9)
            np.testing.assert_allclose(q-last_q,v*servo.dt,atol=1e-12)
            last_q=q;last_v=v
        self.assertAlmostEqual(q[0],-.5,places=5)
        self.assertAlmostEqual(q[6],1.,places=5)
        self.assertEqual(servo.hard_limit_events,0)

    def test_replanning_aligns_future_times(self):
        queue=ChunkContinuity(self.layout,blend_steps=4)
        old=np.repeat(np.arange(50)[:,None],14,axis=1)
        queue.replace(old)
        for _ in range(8): queue.pop()
        new=np.repeat((np.arange(50)+8+2)[:,None],14,axis=1)
        raw,blended,n=queue.replace(new)
        self.assertEqual(n,4)
        self.assertEqual(blended[0,0],9.) # old timestamp 8, new timestamp 8 target 10
        self.assertEqual(blended[0,6],10.) # gripper not delayed by blending
        self.assertEqual(blended[3,0],13.)
        self.assertEqual(queue.pop()[0],9.)

    def test_rejects_bad_commands(self):
        limits=np.tile([-3.,3.],(14,1))
        servo=BoundedJointServo(np.zeros(14),limits,self.layout)
        with self.assertRaises(ValueError): servo.set_target(np.full(14,np.nan))
        with self.assertRaises(ValueError): servo.set_target(np.zeros(6))


if __name__=='__main__': unittest.main()
