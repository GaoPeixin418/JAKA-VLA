"""Regression: tiny physical drift must not become thousands after normalization."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


class StableStatisticsTest(unittest.TestCase):
    def test_constant_and_near_constant_joints_keep_bounded_noise(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            source, raw, output = d/'source', d/'raw', d/'stable'
            (source/'meta').mkdir(parents=True)
            raw.mkdir()
            x = np.tile(np.array([0, 1.4, -1.7, 1.87, 1.57, 0, 1]*2), (100,1))
            x[:,0] = np.linspace(-1,1,100)
            x[:,8] += np.linspace(-1e-5,1e-5,100)
            old = {key: {'mean': x.mean(0).tolist(), 'std': [0.]*14, 'min': x.min(0).tolist(),
                         'max': x.max(0).tolist(), 'count': [100]} for key in ('observation.state','action')}
            (source/'meta/stats.json').write_text(json.dumps(old))
            (source/'meta/project_identity.json').write_text(json.dumps({'robot_id': 'dual_jaka_zu7'}))
            (source/'untouched.txt').write_text('same data')
            pack = [('left_arm_joint_states',slice(0,6)),('left_ee_joint_states',slice(6,7)),
                    ('right_arm_joint_states',slice(7,13)),('right_ee_joint_states',slice(13,14))]
            with h5py.File(raw/'episode_0000000.hdf5','w') as f:
                for group in ('state','action'):
                    for key,sl in pack:f.create_dataset(group+'/'+key,data=x[:,sl].astype(np.float32))
            subprocess.run([sys.executable,str(root/'scripts/stabilize_zu7_statistics.py'),
                            '--source',str(source),'--raw',str(raw),'--output',str(output)],
                           stdout=subprocess.DEVNULL,check=True)
            s=json.loads((output/'meta/stats.json').read_text())['observation.state']
            scale=np.asarray(s['std'])
            self.assertTrue(np.all(scale>=.05))
            self.assertGreater(scale[0],.5)
            q=np.asarray(s['mean']);q[8]+=.0001
            self.assertLess(abs(((q-np.asarray(s['mean']))/scale)[8]),.003)
            self.assertEqual((source/'meta/stats.json').read_text(),json.dumps(old))
            self.assertEqual((output/'untouched.txt').read_text(),'same data')


if __name__=='__main__':unittest.main()
