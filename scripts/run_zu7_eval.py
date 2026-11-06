"""Bounded two-environment evaluation; only manages the server process it creates."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/jaka-vla')


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed-start',type=int,default=0);p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--k',type=int,default=8);p.add_argument('--max-steps',type=int,default=650)
    p.add_argument('--noise-mode',choices=['fixed','random','zero'],default='fixed');p.add_argument('--noise-seed',type=int)
    p.add_argument('--review-video',action='store_true')
    p.add_argument('--instruction', default='Use the left arm to grab the red block, hand it to the right arm, and place it on the blue pad.')
    p.add_argument('--task-scope', choices=['full_handover','left_grasp_lift_hold'], default='full_handover')
    p.add_argument('--initial-offset',type=float,default=0.)
    p.add_argument('--initial-condition-file',type=Path)
    args=p.parse_args()
    if not args.name.startswith('zu7_') or not all(c.isalnum() or c in '_-' for c in args.name):raise ValueError('Invalid experiment name')
    output=ROOT/'results'/args.name;socket=ROOT/'run'/f'{args.name}.sock'
    if output.exists() or socket.exists():raise FileExistsError('Existing experiment/socket; inspect it before rerunning')
    log=ROOT/'logs'/f'{args.name}-server.log'
    with log.open('x') as service_log,(ROOT/'logs'/f'{args.name}-eval.log').open('x') as evaluation_log:
        server_args=[str(ROOT/'envs/vla/bin/python'),'-u',str(ROOT/'scripts/policy_server_v2.py'),
            '--checkpoint',str(args.checkpoint),'--robot-id','dual_jaka_zu7','--socket',str(socket),'--noise-mode',args.noise_mode]
        if args.noise_seed is not None:server_args+=['--noise-seed',str(args.noise_seed)]
        server=subprocess.Popen(server_args,stdout=service_log,stderr=subprocess.STDOUT,cwd=ROOT)
        try:
            deadline=time.monotonic()+120
            while 'READY' not in log.read_text():
                if server.poll() is not None:raise RuntimeError(f'Policy server exited; inspect {log}')
                if time.monotonic()>deadline:raise TimeoutError('Policy server startup exceeded 120 s')
                time.sleep(.25)
            eval_command=[str(ROOT/'envs/robotwin/bin/python'),'-u',str(ROOT/'scripts/eval_v2.py'),
                '--socket',str(socket),'--output',str(output),'--seed-start',str(args.seed_start),'--episodes',str(args.episodes),
                '--k',str(args.k),'--max-steps',str(args.max_steps),'--instruction',args.instruction,'--task-scope',args.task_scope,'--initial-offset',str(args.initial_offset)]
            if args.review_video:eval_command+=['--review-video']
            if args.initial_condition_file:eval_command+=['--initial-condition-file',str(args.initial_condition_file)]
            subprocess.run(eval_command,stdout=evaluation_log,stderr=subprocess.STDOUT,cwd=ROOT,check=True,timeout=900)
            (output/'run_config.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
            server.wait(timeout=30)
            if server.returncode:raise RuntimeError(f'Policy server exited with {server.returncode}')
        finally:
            if server.poll() is None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
    print(output/'results.json',flush=True)


if __name__=='__main__':main()
