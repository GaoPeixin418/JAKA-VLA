"""Collect bounded, identified Zu7 expert trajectories, preserving rig provenance."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import h5py,yaml
ROOT=Path('/data/jaka-vla');REPO=ROOT/'repos/RoboTwin'
sys.path.insert(0,str(REPO));os.chdir(REPO)
from scripts.collect_data import main as collect


def main():
    p=argparse.ArgumentParser();p.add_argument('--task-config',default='zu7_collision_smoke');args=p.parse_args()
    if not args.task_config.startswith('zu7_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.task_config):
        raise ValueError('Task config must be a safe zu7_* name')
    project_config=ROOT/f'configs/{args.task_config}.yml'
    task_config=REPO/'env_cfg/task_config'/f'{args.task_config}.yml'
    if not project_config.is_file():raise FileNotFoundError(project_config)
    if not task_config.is_file() or task_config.read_text()!=project_config.read_text():
        task_config.write_text(project_config.read_text())
    cfg=yaml.safe_load(project_config.read_text())
    assert cfg['embodiment'][:2]==['jaka-zu7-proxy','jaka-zu7-proxy']
    raw=ROOT/f'data/raw/{args.task_config}/handover_block/jaka_zu7_proxy_jaka_zu7_proxy'
    raw.mkdir(parents=True,exist_ok=True)
    identity={'robot_id':'dual_jaka_zu7','rig':json.loads((ROOT/'configs/robots/dual_zu7.json').read_text()),
      'urdf_sha256':hashlib.sha256((REPO/'assets/embodiments/jaka-zu7-proxy/robot.urdf').read_bytes()).hexdigest()}
    identity_path=raw/'robot_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise RuntimeError('Rig changed; use a new collection directory')
    if list((raw/'data').glob('*.hdf5')):raise FileExistsError('Completed/raw episodes already exist; use a new experiment')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    collect('handover_block',args.task_config)
    for path in (raw/'data').glob('*.hdf5'):
        with h5py.File(path,'r+') as hf:
            hf.attrs['robot_id']='dual_jaka_zu7';hf.attrs['rig_urdf_sha256']=identity['urdf_sha256']
    print('Identified Zu7 episodes:',len(list((raw/'data').glob('*.hdf5'))),flush=True)


if __name__=='__main__':main()
