"""Finite continuation of the explicitly authorized 22:00 lab experiment.

Attach only to the existing c/d batches, start e after a slot is free, then
prepare, fine-tune, evaluate, and verify a lossless checkpoint backup.
No stage is retried and existing outputs are never overwritten.
"""
import json,os,shutil,signal,subprocess,time
from datetime import datetime
from pathlib import Path
ROOT=Path('/data/jaka-vla')
RUN=ROOT/'results/lab_evening_20261111'
BATCH_PREFIX='lab_multitarget_20261111_'


def main():
    RUN.mkdir(exist_ok=False)
    report={'completed':False,'stage':'waiting_for_collection','events':[]}
    def save(event=None):
        if event:
            report['events'].append({'time':datetime.now().astimezone().isoformat(timespec='seconds'),'event':event});print(event,flush=True)
        temporary=RUN/'progress.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(RUN/'progress.json')
    def read_batch(suffix):
        p=ROOT/'results'/f'{BATCH_PREFIX}{suffix}'/'progress.json'
        try:return json.loads(p.read_text()) if p.exists() else {}
        except json.JSONDecodeError:return {}
    def run(stage,argv,env=None,timeout=3600):
        report['stage']=stage;save('Starting '+stage)
        with (RUN/f'{stage}.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(list(map(str,argv)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                raise
            if code:raise subprocess.CalledProcessError(code,argv)
        save('Completed '+stage)
    def live_collectors(names):
        found=[]
        for proc in Path('/proc').glob('[0-9]*'):
            try:cmd=(proc/'cmdline').read_bytes().decode().split('\0')
            except (OSError,UnicodeDecodeError):continue
            if cmd and 'python' in cmd[0] and 'scripts/run_lab_curriculum.py' in cmd and any(n in cmd for n in names):found.append(int(proc.name))
        return found
    e_process=None;e_log=None
    collection_deadline=datetime.fromisoformat('2026-11-11T20:40:00+08:00').timestamp()
    collection_finish_limit=datetime.fromisoformat('2026-11-11T20:52:00+08:00').timestamp()
    try:
        if not all((ROOT/'results'/f'{BATCH_PREFIX}{s}').exists() for s in ['c','d']):raise ValueError('Expected existing c/d batches')
        save('Attached to existing c/d batches; at most one additional e batch')
        while True:
            batches={s:read_batch(s) for s in ['c','d']}
            if e_process is None and any(x.get('completed') for x in batches.values()) and time.time()<collection_deadline:
                e_log=(ROOT/'logs'/f'{BATCH_PREFIX}e.log').open('x',encoding='utf-8')
                e_process=subprocess.Popen([str(ROOT/'envs/robotwin/bin/python'),'-u','scripts/run_lab_curriculum.py',
                    '--name',BATCH_PREFIX+'e','--layout-start','25','--layouts','10','--max-attempts','30',
                    '--deadline','2026-11-11T20:40:00+08:00','--reuse-planner','--full-review-layout-every','5'],
                    cwd=ROOT,stdout=e_log,stderr=subprocess.STDOUT,start_new_session=True)
                report['e_pid']=e_process.pid;save('Started e on layouts 25–34 in the free collection slot')
            if e_process is not None:
                batches['e']=read_batch('e')
                if e_process.poll() is not None and not batches['e'].get('completed'):raise RuntimeError('e exited without a completed collection report')
            report['collection']={s:{'accepted':x.get('accepted',0),'attempts':len(x.get('attempts',[])),
                'completed':x.get('completed',False),'stop_reason':x.get('stop_reason')} for s,x in batches.items()}
            save()
            all_done=all(x.get('completed') for x in batches.values())
            names=[BATCH_PREFIX+s for s in batches]
            if all_done and not live_collectors(names):break
            if time.time()>collection_finish_limit:raise TimeoutError('Collection did not finish by 20:52; training not started')
            time.sleep(10)
        if e_process is not None:
            code=e_process.wait(timeout=10)
            if code:raise RuntimeError(f'e exited with code {code}')
        suffixes=['a','b','b2',*batches.keys()]
        report['source_batches']=[str(ROOT/'results'/f'{BATCH_PREFIX}{s}') for s in suffixes];save('Collection finished; no GPU collection workers remain')
        dataset='lab_multitarget_20261111_merged';experiment='lab_pi05_finetune_20261111';evaluation='lab_pi05_eval_20261111'
        robot=ROOT/'envs/robotwin/bin/python'
        run('prepare',[robot,'-u','scripts/prepare_lab_training.py','--batches',*report['source_batches'],'--name',dataset],timeout=1800)
        pi_env={**os.environ,'HF_LEROBOT_HOME':str(ROOT/'data/openpi'),'OPENPI_DATA_HOME':'/data/pi05-cache/models',
            'XLA_PYTHON_CLIENT_PREALLOCATE':'false','WANDB_MODE':'disabled',
            'PYTHONPATH':':'.join(map(str,[ROOT,ROOT/'repos/openpi/src',ROOT/'repos/openpi/packages/openpi-client/src']))}
        pi_env.pop('JAX_PLATFORMS',None)
        run('train',['/data/pi05-env/bin/python','-u','scripts/train_pi05_zu7.py','--dataset',dataset,'--name',experiment,
            '--steps','6000','--deadline','2026-11-11T21:15:00+08:00'],pi_env,timeout=3600)
        training=json.loads((ROOT/'results'/experiment/'audit.json').read_text())
        if not training['completed'] or training['completed_steps']<1:raise ValueError('Training did not produce an updated checkpoint')
        checkpoint=Path(training['checkpoint']);report['checkpoint']=str(checkpoint);report['completed_steps']=training['completed_steps'];save()
        run('evaluate',[robot,'-u','scripts/evaluate_lab_pi05.py','--checkpoint',checkpoint,'--name',evaluation],timeout=1800)
        run('analyze',[robot,'scripts/analyze_lab_pi05.py','--run',ROOT/'results'/evaluation],timeout=120)
        cpu_env={**pi_env,'JAX_PLATFORMS':'cpu'}
        reference=ROOT/'checkpoints/openpi/pi05_zu7_lora_h50/pi05_qualification_20261111/3'
        bundle=ROOT/'backups/lab_pi05_finetune_20261111_delta';restored=ROOT/'run/lab_pi05_delta_roundtrip_20261111'
        run('export_delta',['/data/pi05-env/bin/python','scripts/export_pi05_delta.py','--checkpoint',checkpoint,
            '--reference',reference,'--output',bundle],cpu_env,timeout=600)
        run('verify_delta',['/data/pi05-env/bin/python','scripts/restore_pi05_delta.py','--bundle',bundle,
            '--reference',reference,'--output',restored],cpu_env,timeout=600)
        verification=json.loads((restored/'delta_restore_verification.json').read_text())
        if not verification['passed']:raise ValueError('Delta restoration did not pass')
        shutil.copy2(restored/'delta_restore_verification.json',RUN/'delta_restore_verification.json')
        shutil.copy2(restored/'delta_restore_verification.json',bundle/'delta_restore_verification.json')
        shutil.rmtree(restored)
        report.update(completed=True,stage='local_backup_pending',delta_bundle=str(bundle),evaluation=str(ROOT/'results'/evaluation))
        save('Training, evaluation and exact delta reconstruction finished; local downloads still required')
    except Exception as exc:
        report['error']=repr(exc);save('Pipeline stopped: '+repr(exc));raise
    finally:
        if e_process is not None and e_process.poll() is None:
            os.killpg(e_process.pid,signal.SIGTERM)
            try:e_process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(e_process.pid,signal.SIGKILL);e_process.wait(timeout=10)
        if e_log is not None:e_log.close()
        save()

if __name__=='__main__':main()
