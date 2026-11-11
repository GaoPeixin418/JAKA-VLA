"""Convert identified Zu7 trajectories using the shared RGB/timestamp validator."""
import argparse,json,sys
import h5py
from pathlib import Path
ROOT=Path('/data/jaka-vla')
sys.path.insert(0,str(ROOT/'scripts'))
from convert_dataset import main as convert


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repo-id',required=True)
    args=p.parse_args()
    identity=json.loads((args.source.parent/'robot_identity.json').read_text())
    if identity['robot_id'] not in ('dual_jaka_zu7','dual_jaka_zu7_lab_archive'):raise ValueError('Source robot mismatch')
    for path in args.source.glob('episode_*.hdf5'):
        with h5py.File(path) as hf:
            if hf.attrs.get('rig_urdf_sha256')!=identity['urdf_sha256']:raise ValueError(f'Rig mismatch: {path}')
    sys.argv=[sys.argv[0],'--source',str(args.source),'--output',str(args.output),'--repo-id',args.repo_id,'--robot-id',identity['robot_id']]
    convert()
    identity.update(source=str(args.source),state_semantics='measured_qpos',action_semantics='next_sample_drive_targets')
    (args.output/'meta/project_identity.json').write_text(json.dumps(identity,indent=2)+'\n')


if __name__=='__main__':main()
