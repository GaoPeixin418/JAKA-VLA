import unittest
import numpy as np
from jaka_vla.grasp_monitor import EmptyGraspMonitor


class GraspMonitorTest(unittest.TestCase):
    def test_empty_requires_persistence_and_contact_aperture_is_not_empty(self):
        monitor=EmptyGraspMonitor([6,13])
        command=np.ones(14);measured=np.ones(14);command[6]=0;measured[6]=0
        for _ in range(3):self.assertEqual(monitor.update(command,measured),[])
        self.assertEqual(monitor.update(command,measured),[0])
        measured[6]=.6
        for _ in range(20):self.assertEqual(monitor.update(command,measured),[])
        command[6]=1;measured[6]=0
        self.assertEqual(monitor.update(command,measured),[])


if __name__=='__main__':unittest.main()
