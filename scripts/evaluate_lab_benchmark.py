"""Frozen-checkpoint, predeclared multi-layout evaluation with bounded workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

ROOT = Path('/data/jaka-vla')
sys.path.insert(0, str(ROOT))


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--no-new-after', required=True)
    p.add_argument('--hard-stop', required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--execute-prepared', action='store_true')
    args = p.parse_args()
    if not args.name.startswith('lab_pi05_benchmark_') or not all(c.isalnum() or c == '_' for c in args.name):
        raise ValueError('Invalid run name')
    no_new = datetime.fromisoformat(args.no_new_after).timestamp()
    hard_stop = datetime.fromisoformat(args.hard_stop).timestamp()
    if hard_stop <= no_new or time.time() >= no_new:
        raise ValueError('Invalid deadline')
    from jaka_vla.lab_scene import scene_spec
    from jaka_vla.contact_acceptance import lab_forbidden_contact
    run = ROOT / 'results' / args.name
    if args.execute_prepared:
        plan = json.loads((run/'protocol.json').read_text())
        if plan['arguments']['checkpoint'] != str(args.checkpoint) or plan['arguments']['no_new_after'] != args.no_new_after or plan['arguments']['hard_stop'] != args.hard_stop:
            raise ValueError('Prepared protocol differs')
        if (run/'execution_started.json').exists():
            raise FileExistsError('Run was already started')
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run/'logs').mkdir(); (run/'scenes').mkdir()
        layouts = [('reproduction', 0)] + [('development', x) for x in range(1000, 1010)] + [('heldout', x) for x in range(2000, 2020)]
        cases = []
        for group_index, (split, layout) in enumerate(layouts):
            for target in range(3):
                spec = scene_spec(layout, target, 'train' if split == 'reproduction' else split)
                key = f'{split}_layout_{layout}_target_{target}'
                scene = run/'scenes'/f'{key}.json'
                save(scene, spec)
                cases.append({'key':key, 'split':split, 'layout':layout, 'target':target, 'color':spec['objects'][target]['color'],
                    'seed':60000+layout*3+target, 'worker':group_index%2, 'scene':str(scene), 'scene_sha256':digest(scene),
                    'instruction':spec['instruction'], 'review_video':layout%5 == 0})
        if len(cases) != 93 or len({c['key'] for c in cases}) != 93:
            raise ValueError('Unexpected case population')
        model_files = [args.checkpoint/'project_identity.json'] + sorted(x for x in (args.checkpoint/'params').rglob('*') if x.is_file()) + sorted(x for x in (args.checkpoint/'assets').rglob('*') if x.is_file())
        code_files = ['scripts/eval_v2.py','scripts/policy_server_pi05.py','scripts/evaluate_lab_benchmark.py','jaka_vla/control.py','jaka_vla/grasp_monitor.py','jaka_vla/lab_scene.py','jaka_vla/contact_acceptance.py','jaka_vla/openpi_zu7.py','configs/robots/dual_zu7_lab.json']
        plan = {'created_at':datetime.now().astimezone().isoformat(), 'arguments':vars(args),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'conditions':{'K':25,'H':50,'noise_seed':42,'noise_mode':'fixed','max_steps':650,'physics_hz':250,'control_hz':25,'workers':2},
            'scope':'Frozen current checkpoint, left grasp lift hold only; synchronous inference pauses simulation; no training or parameter selection during this evaluation.',
            'primary_population':'60 heldout episodes in layouts 2000-2019, three targets each. All attempts retained; layout is the independent sampling unit. Development and reproduction reported separately.',
            'success_definition':'Lift >0.05m, measured simulated aperture 0.08-0.90, at least 2 of 3 fingers contact target, no right target contact, continuous >=0.6s; no failure and no forbidden contact.',
            'deadline_rule':'Fixed shard order independent of outcomes. No new episode after no_new_after; active episode capped at hard_stop. Partial groups and operational errors reported explicitly.',
            'video_rule':'Every episode has policy head-camera video; HD multiview for layout 0 and layout IDs divisible by 5, chosen before evaluation.',
            'model_sha256':{str(x.relative_to(args.checkpoint)):digest(x) for x in model_files},
            'code_sha256':{str(x):digest(ROOT/x) for x in code_files},'cases':cases}
        save(run/'protocol.json',plan)
    print(json.dumps({'prepared':str(run), 'cases':len(plan['cases'])}),flush=True)
    if args.prepare_only:
        return
    save(run/'execution_started.json',{'at':datetime.now().astimezone().isoformat(), 'protocol_sha256':digest(run/'protocol.json')})
    rig = json.loads((ROOT/'configs/robots/dual_zu7_lab.json').read_text())
    env = os.environ.copy()
    env.update(HF_LEROBOT_HOME=str(ROOT/'data/openpi'),OPENPI_DATA_HOME='/data/pi05-cache/models',XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=':'.join(map(str,[ROOT,ROOT/'repos/openpi/src',ROOT/'repos/openpi/packages/openpi-client/src'])))

    def worker(index):
        cases = [c for c in plan['cases'] if c['worker']==index]
        status = {'worker':index,'planned':len(cases),'rollouts':[],'completed':False}
        summary = run/f'worker_{index}.json'
        socket = ROOT/'run'/f'benchmark_20261116_{index}.sock'
        if socket.exists():
            raise FileExistsError(socket)
        server = None
        try:
            with (run/'logs'/f'server_{index}.log').open('w') as slog:
                server = subprocess.Popen(['/data/pi05-env/bin/python','-u','scripts/policy_server_pi05.py','--checkpoint',str(args.checkpoint),'--robot-id',rig['robot_id'],'--socket',str(socket),'--noise-mode','fixed','--noise-seed','42','--connections',str(len(cases)),'--reload-output-dir',str(run/f'reload_{index}'),'--observation-dir',str(run/f'policy_observations_{index}')],cwd=ROOT,env=env,stdout=slog,stderr=subprocess.STDOUT,start_new_session=True)
                status.update(server_pid=server.pid,stage='starting_policy');save(summary,status)
                start=time.monotonic()
                while 'READY' not in (run/'logs'/f'server_{index}.log').read_text():
                    if server.poll() is not None:raise RuntimeError('Policy service exited')
                    if time.monotonic()-start > 420 or time.time()>=no_new:raise TimeoutError('Policy startup deadline')
                    time.sleep(.5)
                for case in cases:
                    if time.time()>=no_new or hard_stop-time.time()<60:
                        status['stop_reason']='deadline';break
                    if server.poll() is not None:
                        raise RuntimeError('Policy service no longer running')
                    status.update(stage='evaluating',active_case=case['key']);save(summary,status)
                    output=run/case['key']
                    command=[str(ROOT/'envs/robotwin/bin/python'),'-u','scripts/eval_v2.py','--robot-id',rig['robot_id'],'--rig-config','configs/robots/dual_zu7_lab.json','--scene-spec',case['scene'],'--seed-start',str(case['seed']),'--episodes','1','--socket',str(socket),'--output',str(output),'--k','25','--max-steps','650','--instruction',case['instruction'],'--task-scope','left_grasp_lift_hold','--policy-label','pi0.5',*(['--review-video'] if case['review_video'] else [])]
                    row=dict(case);row['started_at']=datetime.now().astimezone().isoformat()
                    with (run/'logs'/f"{case['key']}.log").open('w') as log:
                        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        try:
                            code=process.wait(timeout=min(700,max(1,hard_stop-time.time())))
                            if code!=0:raise RuntimeError(f'Evaluator exit {code}')
                            result=json.loads((output/'results.json').read_text())[0]
                            contacts=json.loads((output/f"seed_{case['seed']}_contacts.json").read_text())
                            forbidden=[c['pair'] for c in contacts if lab_forbidden_contact(c['pair'])]
                            row.update(status='completed',grasp_success=result['grasp_success'],failure_reason=result['failure_reason'],unintended_contacts=forbidden,qualified_grasp_success=bool(result['grasp_success'] and not result['failure_reason'] and not forbidden),results=str(output/'results.json'),wall_seconds=result['wall_seconds'])
                        except Exception as error:
                            stop(process)
                            row.update(status='operational_error',error=str(error),qualified_grasp_success=None)
                    row['finished_at']=datetime.now().astimezone().isoformat()
                    status['rollouts'].append(row);save(summary,status)
                    print(json.dumps(row,ensure_ascii=False),flush=True)
                status['completed']=len(status['rollouts'])==len(cases) and all(x['status']=='completed' for x in status['rollouts'])
        except Exception as error:
            status['fatal_error']=str(error)
        finally:
            if server is not None:stop(server)
            status.update(stage='finished',finished_at=datetime.now().astimezone().isoformat());status.pop('active_case',None);save(summary,status)

    threads=[threading.Thread(target=worker,args=(i,)) for i in range(2)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    save(run/'execution_finished.json',{'at':datetime.now().astimezone().isoformat()})

if __name__=='__main__':
    main()
