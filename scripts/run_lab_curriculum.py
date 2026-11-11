"""Bounded multi-target expert collection with mandatory same-scene servo replay."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data/jaka-vla')
sys.path.insert(0,str(ROOT))
from jaka_vla.lab_scene import scene_spec
from jaka_vla.contact_acceptance import lab_forbidden_contact

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--layout-start',type=int,default=0)
    p.add_argument('--target-start',type=int,default=0,choices=[0,1,2],help='First target in the first layout when continuing a stopped batch')
    p.add_argument('--layouts',type=int,default=10)
    p.add_argument('--max-attempts',type=int,default=30)
    p.add_argument('--reuse-planner',action='store_true',help='Run expert probes sequentially in this process and reuse only planner allocations; replay remains an independent process')
    p.add_argument('--full-review-layout-every',type=int,default=1,help='Keep extra HD observer videos for every Nth layout; all policy RGB and numeric acceptance remain recorded')
    p.add_argument('--deadline',default='2026-11-11T18:45:00+08:00')
    args=p.parse_args()
    if args.full_review_layout_every<1:raise ValueError('Review interval must be positive')
    if not 0<=args.layout_start<100 or not 1<=args.layouts<=100-args.layout_start:
        raise ValueError('Only training layouts 0..99 may be collected here; dev=1000..1009, final=2000..2019')
    if not args.name.startswith('lab_') or not all(c.isalnum() or c=='_' for c in args.name):raise ValueError('Invalid run name')
    run=ROOT/'results'/args.name;run.mkdir(parents=True,exist_ok=False)
    (run/'scenes').mkdir();(run/'logs').mkdir()
    report={'scope':'scripted expert; not learned policy','arguments':vars(args),'attempts':[],
            'splits':{'train_layouts':[0,99],'development_layouts':[1000,1009],'final_holdout_layouts':[2000,2019]},
            'planned_full_training_episodes':300,'accepted':0,'completed':False}
    deadline=datetime.fromisoformat(args.deadline).timestamp()
    accepted_probes=[];accepted_replays=[];consecutive_setup_failures=0
    reuse_failed=False
    def command(argv,log):
        nonlocal reuse_failed
        with log.open('w') as f:
            if args.reuse_planner and str(argv[0])=='scripts/probe_zu7_short_grasp.py':
                import contextlib,gc,signal,traceback
                from jaka_vla.planner_cache import release_scene_entities
                before_argv=sys.argv;before_dir=Path.cwd();before_env=os.environ.get('JAKA_REUSE_PLANNER')
                def timeout_handler(signum,frame):raise TimeoutError('Inline probe exceeded 600 s')
                previous_handler=signal.signal(signal.SIGALRM,timeout_handler)
                code=0
                try:
                    os.environ['JAKA_REUSE_PLANNER']='1';sys.argv=list(map(str,argv))
                    sys.path.insert(0,str(ROOT/'scripts'))
                    with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                        signal.alarm(600)
                        import probe_zu7_short_grasp
                        os.chdir(ROOT/'repos/RoboTwin')
                        probe_zu7_short_grasp.main()
                except Exception:
                    traceback.print_exc(file=f);reuse_failed=True;code=1
                finally:
                    signal.alarm(0);signal.signal(signal.SIGALRM,previous_handler)
                    release_scene_entities();gc.collect();sys.argv=before_argv;os.chdir(before_dir)
                    if before_env is None:os.environ.pop('JAKA_REUSE_PLANNER',None)
                    else:os.environ['JAKA_REUSE_PLANNER']=before_env
                return code
            try:return subprocess.run([sys.executable,'-u',*map(str,argv)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode
            except subprocess.TimeoutExpired:return 124
    try:
        for layout in range(args.layout_start,args.layout_start+args.layouts):
            for target in range(args.target_start if layout==args.layout_start else 0,3):
                if len(report['attempts'])>=args.max_attempts or time.time()>=deadline:break
                spec=scene_spec(layout,target,'train');key=f'layout_{layout:04d}_target_{target}'
                scene=run/'scenes'/f'{key}.json';scene.write_text(json.dumps(spec,indent=2)+'\n')
                seed=60000+layout*3+target;probe=run/f'{key}_probe';replay=run/f'{key}_replay'
                row={'layout':layout,'target':target,'color':spec['objects'][target]['color'],'seed':seed,
                     'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),'probe':str(probe),'replay':str(replay),'accepted':False}
                report['attempts'].append(row)
                full_review=layout%args.full_review_layout_every==0
                probe_video_args=[] if full_review else ['--no-review-video']
                row['full_review_video']=full_review
                row['probe_returncode']=command(['scripts/probe_zu7_short_grasp.py','--rig-config','configs/robots/dual_zu7_lab.json',
                    '--scene-spec',scene,'--seed',seed,'--output',probe,'--collect',*probe_video_args],run/'logs'/f'{key}_probe.log')
                result=json.loads((probe/'result.json').read_text()) if (probe/'result.json').exists() else {}
                row['probe_failure']=result.get('failure')
                if row['probe_returncode']==0 and result.get('grasp_success') and result.get('failure') is None:
                    consecutive_setup_failures=0
                    row['replay_returncode']=command(['scripts/eval_v2.py','--robot-id','dual_jaka_zu7_lab_archive',
                        '--rig-config','configs/robots/dual_zu7_lab.json','--scene-spec',scene,'--seed-start',seed,'--episodes',1,
                        '--expert-hdf5',probe/'raw/data','--output',replay,*(['--review-video'] if full_review else []),
                        '--instruction',spec['instruction'],'--task-scope','left_grasp_lift_hold'],run/'logs'/f'{key}_replay.log')
                    if row['replay_returncode']==0:
                        check=json.loads((replay/'results.json').read_text())[0]
                        contacts=json.loads((replay/f'seed_{seed}_contacts.json').read_text())
                        cross=any(any(s.startswith('left:') for s in c['pair']) and any(s.startswith('right:') for s in c['pair']) for c in contacts)
                        row.update(replay_grasp_success=check['grasp_success'],replay_failure=check['failure_reason'],interarm_contact=cross)
                        probe_contacts=json.loads((probe/'contacts.json').read_text())
                        forbidden=[c['pair'] for c in [*contacts,*probe_contacts] if lab_forbidden_contact(c['pair'])]
                        row['forbidden_contacts']=sorted(set(tuple(pair) for pair in forbidden))
                        row['accepted']=bool(check['grasp_success'] and not check['failure_reason'] and not cross and not forbidden)
                elif not result or result.get('sim_seconds',0)==0:
                    consecutive_setup_failures+=1
                else:consecutive_setup_failures=0
                if row['accepted']:
                    accepted_probes.append(probe);accepted_replays.append(replay);report['accepted']+=1
                (run/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(row),flush=True)
                if reuse_failed:
                    report['stop_reason']='persistent_worker_exception';break
                if consecutive_setup_failures>=3:
                    report['stop_reason']='three_consecutive_setup_or_zero_motion_failures';break
            if len(report['attempts'])>=args.max_attempts or time.time()>=deadline or report.get('stop_reason'):break
        if accepted_probes:
            dataset=ROOT/'data/raw'/args.name
            code=command(['scripts/assemble_grasp_curriculum.py','--probes',*accepted_probes,
                          '--replays',*accepted_replays,'--output',dataset],run/'logs/assemble.log')
            if code:raise RuntimeError('Acceptance assembly failed; see assemble.log')
            report['raw_dataset']=str(dataset)
        report['completed']=True
        report.setdefault('stop_reason','deadline' if time.time()>=deadline else 'attempt_budget')
    finally:
        (run/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'accepted':report['accepted'],'attempts':len(report['attempts']),'stop_reason':report.get('stop_reason')}),flush=True)

if __name__=='__main__':main()
