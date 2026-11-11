"""Merge accepted data, convert formats and audit new-rig inputs before training."""
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path('/data/jaka-vla')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',type=Path,required=True);p.add_argument('--name',required=True)
    args=p.parse_args()
    if not args.name.startswith('lab_') or not all(c.isalnum() or c=='_' for c in args.name):raise ValueError('Invalid dataset name')
    run=ROOT/'results'/f'{args.name}_preparation';run.mkdir(parents=True,exist_ok=False)
    raw=ROOT/'data/raw'/args.name;v3=ROOT/'data/lerobot'/args.name;compatible=ROOT/'data/openpi'/args.name
    profile={'completed':False,'name':args.name,'source_batches':list(map(str,args.batches)),'stages':[]}
    env=os.environ.copy();pi_env={**env,'HF_LEROBOT_HOME':str(ROOT/'data/openpi'),'OPENPI_DATA_HOME':'/data/pi05-cache/models',
        'XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cpu','WANDB_MODE':'disabled',
        'PYTHONPATH':':'.join(map(str,[ROOT,ROOT/'repos/openpi/src',ROOT/'repos/openpi/packages/openpi-client/src']))}
    def execute(stage,python,argv,environment=env):
        with (run/f'{stage}.log').open('x') as log:
            subprocess.run([str(python),'-u',*map(str,argv)],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        profile['stages'].append(stage);(run/'progress.json').write_text(json.dumps(profile,indent=2)+'\n')
    try:
        execute('merge',ROOT/'envs/robotwin/bin/python',['scripts/merge_lab_batches.py','--batches',*args.batches,'--output',raw])
        coverage=json.loads((raw/'coverage.json').read_text());profile['coverage']=coverage
        if coverage['episodes']<9 or coverage['distinct_layouts']<3 or set(coverage['color_counts'])!={'red','blue','yellow'}:
            raise ValueError('Minimum new-rig training coverage not reached')
        execute('observation_controls',ROOT/'envs/robotwin/bin/python',['scripts/audit_lab_observation_controls.py',
            '--raw',raw,'--output',run/'initial_observation_controls.json'])
        execute('lerobot3',ROOT/'envs/vla/bin/python',['scripts/convert_dataset_v2.py','--source',raw/'data','--output',v3,'--repo-id',f'local/{args.name}'])
        execute('openpi',Path('/data/pi05-env/bin/python'),['scripts/convert_zu7_for_openpi.py','--source',v3,'--output',compatible],pi_env)
        execute('statistics',Path('/data/pi05-env/bin/python'),['scripts/prepare_pi05_stats.py','--dataset',args.name],pi_env)
        execute('audit',Path('/data/pi05-env/bin/python'),['scripts/audit_pi05_data.py','--dataset',args.name,'--output',run/'input_audit.json'],pi_env)
        profile.update(completed=True,raw=str(raw),lerobot3=str(v3),openpi=str(compatible))
    finally:
        (run/'progress.json').write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps(profile),flush=True)

if __name__=='__main__':main()
