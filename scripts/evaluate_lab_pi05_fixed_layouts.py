"""Matched bounded evaluation on a fixed cohort of entirely new layouts."""
import argparse,json,os,subprocess,time
from datetime import datetime
from pathlib import Path
ROOT=Path('/data/jaka-vla')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--deadline',required=True);p.add_argument('--max-steps',type=int,default=650)
    p.add_argument('--layouts',type=int,nargs='+',default=[5000,5001,5002])
    p.add_argument('--targets',type=int,choices=[0,1,2],nargs='+',default=[0,1,2])
    p.add_argument('--save-observations',action='store_true')
    p.add_argument('--k',type=int,choices=range(1,51),default=25)
    args=p.parse_args()
    if len(set(args.layouts))!=len(args.layouts) or len(set(args.targets))!=len(args.targets):raise ValueError('Duplicate evaluation case')
    if any(layout<5000 or layout>=6000 for layout in args.layouts):raise ValueError('Development layouts must be reserved in [5000,6000); keep final blind layouts separate')
    case_count=len(args.layouts)*len(args.targets)
    if not args.name.startswith('lab_pi05_') or not all(c.isalnum() or c=='_' for c in args.name):raise ValueError('Invalid run name')
    run=ROOT/'results'/args.name;run.mkdir(parents=True,exist_ok=False);(run/'logs').mkdir();(run/'scenes').mkdir()
    socket=ROOT/'run'/f'{args.name}.sock'
    if socket.exists():raise FileExistsError(socket)
    deadline=datetime.fromisoformat(args.deadline).timestamp()
    rig=json.loads((ROOT/'configs/robots/dual_zu7_lab.json').read_text())
    report={'checkpoint':str(args.checkpoint),'arguments':vars(args),'conditions':{'K':args.k,'H':50,'noise_seed':42,'max_steps':args.max_steps},
            'scope':'fixed unseen-layout development cohort; no final holdout; synchronous inference with simulation paused',
            'rollouts':[],'completed':False}
    import sys
    sys.path.insert(0,str(ROOT))
    from jaka_vla.lab_scene import scene_spec
    from jaka_vla.contact_acceptance import lab_forbidden_contact
    service_env=os.environ.copy();service_env.update(HF_LEROBOT_HOME=str(ROOT/'data/openpi'),OPENPI_DATA_HOME='/data/pi05-cache/models',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled',
        PYTHONPATH=':'.join(map(str,[ROOT,ROOT/'repos/openpi/src',ROOT/'repos/openpi/packages/openpi-client/src'])))
    with (run/'logs/server.log').open('w') as service_log:
        server=subprocess.Popen(['/data/pi05-env/bin/python','-u','scripts/policy_server_pi05.py','--checkpoint',str(args.checkpoint),
                '--robot-id',rig['robot_id'],'--socket',str(socket),'--noise-mode','fixed','--noise-seed','42','--connections',str(case_count),
                *(['--observation-dir',str(run/'policy_observations')] if args.save_observations else [])],
                cwd=ROOT,env=service_env,stdout=service_log,stderr=subprocess.STDOUT)
        try:
            startup=time.monotonic()+360
            while 'READY' not in (run/'logs/server.log').read_text():
                if server.poll() is not None:raise RuntimeError('Policy service exited')
                if time.monotonic()>startup:raise TimeoutError('Policy service startup exceeded 360 s')
                time.sleep(.25)
            for layout in args.layouts:
                for target in args.targets:
                    if time.time()>=deadline:break
                    spec=scene_spec(layout,target,'development');key=f'layout_{layout}_target_{target}'
                    path=run/'scenes'/f'{key}.json';path.write_text(json.dumps(spec,indent=2)+'\n')
                    seed=60000+layout*3+target;output=run/key
                    with (run/'logs'/f'{key}.log').open('w') as log:
                        subprocess.run([str(ROOT/'envs/robotwin/bin/python'),'-u','scripts/eval_v2.py','--robot-id',rig['robot_id'],
                            '--rig-config','configs/robots/dual_zu7_lab.json','--scene-spec',str(path),'--seed-start',str(seed),'--episodes','1',
                            '--socket',str(socket),'--output',str(output),'--k',str(args.k),'--max-steps',str(args.max_steps),'--review-video',
                            '--instruction',spec['instruction'],'--task-scope','left_grasp_lift_hold','--policy-label','pi0.5'],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=700,check=True)
                    result=json.loads((output/'results.json').read_text())[0]
                    contacts=json.loads((output/f'seed_{seed}_contacts.json').read_text())
                    forbidden=[c['pair'] for c in contacts if lab_forbidden_contact(c['pair'])]
                    row={'split':'development_unseen_layouts','layout':layout,'target':target,'color':spec['objects'][target]['color'],'seed':seed,
                         'grasp_success':result['grasp_success'],'failure_reason':result['failure_reason'],
                         'unintended_contacts':forbidden,'qualified_grasp_success':bool(result['grasp_success'] and not result['failure_reason'] and not forbidden),
                         'results':str(output/'results.json')}
                    report['rollouts'].append(row);print(json.dumps(row),flush=True)
                    (run/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
                if time.time()>=deadline:break
            report['completed']=len(report['rollouts'])==case_count
        finally:
            if server.poll() is None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
            report['groups']={'development_unseen_layouts':{'successes':sum(x['qualified_grasp_success'] for x in report['rollouts']),
                                'episodes':len(report['rollouts'])},'layouts':args.layouts}
            (run/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(report['groups']),flush=True)

if __name__=='__main__':main()
