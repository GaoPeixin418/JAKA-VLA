"""Round-trip multi-part archives, symlinks and fail closed on corruption."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class BackupTest(unittest.TestCase):
    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);project=base/'project';project.mkdir()
            (project/'AGENT.md').write_text('test')
            (project/'configs/robots').mkdir(parents=True)
            (project/'configs/robots/dual_zu7.json').write_text('{}')
            for rel in ('data','results','logs','backup-source','checkpoints/run/003000'):(project/rel).mkdir(parents=True)
            payload=os.urandom(2*1024**2+123);(project/'data/test.bin').write_bytes(payload)
            (project/'checkpoints/run/003000/config.json').write_text('{}')
            (project/'checkpoints/run/last').symlink_to('003000')
            backup=base/'backup';restored=base/'restored'
            subprocess.run([sys.executable,str(ROOT/'scripts/create_backup.py'),'--root',str(project),'--output',str(backup),'--part-mib','1'],check=True,stdout=subprocess.DEVNULL)
            self.assertGreater(len(list(backup.glob('datasets.tar.gz.part*'))),1)
            subprocess.run([sys.executable,str(ROOT/'scripts/restore_backup.py'),'--backup',str(backup),'--restore-to',str(restored)],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual((restored/'data/test.bin').read_bytes(),payload)
            self.assertTrue((restored/'checkpoints/run/last').is_symlink())
            self.assertEqual((restored/'checkpoints/run/last/config.json').read_text(),'{}')
            first=next(backup.glob('datasets.tar.gz.part*'))
            with first.open('r+b') as stream:stream.write(b'corrupt')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/restore_backup.py'),'--backup',str(backup)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('checksum mismatch',result.stderr)


if __name__=='__main__':unittest.main()
