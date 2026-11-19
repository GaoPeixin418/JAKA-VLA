"""Bounded real pi0.5 LoRA training with per-module gradient/freeze evidence."""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path('/data/jaka-vla')
os.environ.setdefault('HF_LEROBOT_HOME',str(ROOT/'data/openpi'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('WANDB_MODE','disabled')
sys.path.insert(0,str(ROOT))
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.training import data_loader, checkpoints, sharding, weight_loaders
from jaka_vla.pi05_config import build_config
from jaka_vla.identity import rig_urdf_hash


def flat(state):
    return {'/'.join(map(str,k)):v for k,v in state.flat_state().items()}


def digest(state):
    result={}
    for name,v in flat(state).items():
        a=np.asarray(v.value)
        result[name]=hashlib.sha256(a.tobytes()).hexdigest()
    return result


def group(name):
    if 'PaliGemma/llm' in name:
        return 'action_expert_lora' if '_1' in name else 'vlm_lora'
    return name.split('/')[0]


def step(config,rng,state,batch):
    model=nnx.merge(state.model_def,state.params);model.train()
    def loss_fn(m):
        return jnp.mean(m.compute_loss(jax.random.fold_in(rng,state.step),batch[0],batch[1],train=True))
    loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,config.trainable_filter))(model)
    params=state.params.filter(config.trainable_filter)
    updates,new_opt=state.tx.update(grads,state.opt_state,params)
    nnx.update(model,optax.apply_updates(params,updates))
    info={'loss':loss,'grad_norm':optax.global_norm(grads),'update_norm':optax.global_norm(updates)}
    groups={}
    for name,v in flat(grads).items(): groups.setdefault(group(name),[]).append(v.value)
    for name,values in groups.items():
        info['grad_'+name]=optax.global_norm(values)
        info['finite_'+name]=jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in values]))
    return dataclasses.replace(state,step=state.step+1,params=nnx.state(model),opt_state=new_opt),info


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--name',required=True)
    p.add_argument('--steps',type=int,default=3);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--horizon',type=int,default=50);p.add_argument('--deadline',default='2026-11-11T19:05:00+08:00')
    p.add_argument('--episodes',type=int,nargs='+',help='Explicit episode subset for a bounded fit diagnostic')
    p.add_argument('--initial-checkpoint',type=Path,help='Initialize weights from this checkpoint; optimizer and schedule start fresh')
    p.add_argument('--allow-dataset-transfer',action='store_true',help='Allow a new dataset ID only after strict π0.5/horizon/14D/robot/URDF/rig compatibility checks')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.steps<1 or not all(c.isalnum() or c in '_-' for c in args.name):raise ValueError('Invalid run')
    cfg=dataclasses.replace(build_config(args.dataset,args.name,steps=args.steps,batch_size=args.batch_size,horizon=args.horizon),seed=args.seed)
    source_identity=None;checkpoint_transfer=None
    if args.initial_checkpoint is not None:
        args.initial_checkpoint=args.initial_checkpoint.resolve()
        source_identity=json.loads((args.initial_checkpoint/'project_identity.json').read_text())
        if source_identity['horizon']!=args.horizon or source_identity['model']!='pi05':
            raise ValueError('Initialization checkpoint model/horizon mismatch')
        if source_identity['dataset']!=args.dataset:
            if not args.allow_dataset_transfer:
                raise ValueError('Checkpoint dataset differs; use explicit audited dataset transfer')
            target_root=ROOT/'data/openpi'/args.dataset
            target_identity=json.loads((target_root/'meta/project_identity.json').read_text())
            target_info=json.loads((target_root/'meta/info.json').read_text())
            features=target_info['features']
            state_shapes=[features[key].get('shape') for key in ('observation.state','state') if key in features]
            action_shapes=[features[key].get('shape') for key in ('action','actions') if key in features]
            if (source_identity.get('robot_id')!=target_identity.get('robot_id') or
                source_identity.get('urdf_sha256')!=target_identity.get('urdf_sha256') or
                source_identity.get('state_dim')!=14 or source_identity.get('action_dim')!=14 or
                source_identity.get('rig')!=target_identity.get('rig') or
                not state_shapes or not action_shapes or
                any(shape!=[14] for shape in state_shapes+action_shapes)):
                raise ValueError('Unsafe dataset transfer: robot, URDF, rig, or 14D feature schema differs')
            stats=cfg.assets_dirs/args.dataset/'norm_stats.json'
            checkpoint_transfer={'from_dataset':source_identity['dataset'],'to_dataset':args.dataset,
                'model':'pi05','horizon':args.horizon,'robot_id':target_identity['robot_id'],
                'urdf_sha256':target_identity['urdf_sha256'],'rig_match':True,
                'state_action_dim':14,'target_normalization':str(stats),
                'target_normalization_sha256':hashlib.sha256(stats.read_bytes()).hexdigest() if stats.exists() else None,
                'optimizer_and_schedule':'reinitialized','reason':'Explicit transfer after RGB/state/action/timestamp and training-set audit'}
        cfg=dataclasses.replace(cfg,weight_loader=weight_loaders.CheckpointWeightLoader(str(args.initial_checkpoint/'params')))
    output=ROOT/'results'/args.name;output.mkdir(parents=True,exist_ok=False)
    report={'arguments':vars(args) | {'initial_checkpoint':str(args.initial_checkpoint) if args.initial_checkpoint else None},'checkpoint_transfer':checkpoint_transfer,'completed':False,'stage':'initializing','metrics':[],'started_at':time.time(),
            'scope':'engineering qualification; training loss is not task success' if args.steps<=3 else 'bounded pi0.5 fine-tuning; training loss is not task success'}
    provenance_paths=[ROOT/'data/openpi'/args.dataset/'meta'/name for name in
        ('info.json','episodes.jsonl','tasks.jsonl','project_identity.json')]
    provenance_paths += [cfg.assets_dirs/args.dataset/name for name in ('norm_stats.json','audit.json')]
    provenance_paths += [Path(__file__),ROOT/'jaka_vla/pi05_config.py',ROOT/'jaka_vla/openpi_zu7.py']
    report['provenance']={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'file_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in provenance_paths if path.exists()}}
    stop=threading.Event();gpu=[]
    def monitor():
        while not stop.is_set():
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if r.returncode==0:gpu.append({'time':time.time(),'used_mib':int(r.stdout.splitlines()[0])})
            stop.wait(1.)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    manager=None
    try:
        jax.config.update('jax_compilation_cache_dir',str(ROOT/'cache/jax-pi05'))
        mesh=sharding.make_mesh(1)
        ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        sample_order=None
        if args.episodes is None:
            loader=data_loader.create_data_loader(cfg,sharding=ds,shuffle=True)
        else:
            episodes=[json.loads(line) for line in (ROOT/'data/openpi'/args.dataset/'meta/episodes.jsonl').read_text().splitlines()]
            requested=set(args.episodes);selected=[];offset=0;chosen=[]
            for episode in episodes:
                if episode['episode_index'] in requested:
                    selected.extend(range(offset,offset+episode['length']));chosen.append(episode)
                offset+=episode['length']
            if len(chosen)!=len(requested) or not selected:
                raise ValueError('Requested episode subset is missing or empty')
            dc=cfg.data.create(cfg.assets_dirs,cfg.model)
            raw=data_loader.create_torch_dataset(dc,args.horizon,cfg.model)
            if len(raw)!=offset:raise ValueError('Episode metadata and dataset length disagree')
            transformed=data_loader.transform_dataset(raw,dc)
            generator=np.random.default_rng(args.seed);order=[]
            # Record exact anchors, with complete shuffled passes before any repeat.
            while len(order)<(args.steps+1)*args.batch_size:
                order.extend(generator.permutation(selected).tolist())
            sample_order=np.asarray(order[:(args.steps+1)*args.batch_size],dtype=np.int64)
            np.save(output/'planned_sample_indices.npy',sample_order)
            loader=data_loader.DataLoaderImpl(dc,data_loader.TorchDataLoader(transformed,
                local_batch_size=args.batch_size,sharding=ds,shuffle=False,sampler=sample_order.tolist(),
                num_workers=0,seed=args.seed))
            report['training_selection']={'episodes':chosen,'eligible_frames':len(selected),'seed':args.seed,
                'scope':'same-scene fit diagnostic; no generalization claim',
                'sample_order_sha256':hashlib.sha256(sample_order.tobytes()).hexdigest()}
        report['optimizer_initialization']='fresh optimizer and LR schedule'
        iterator=iter(loader);batch=next(iterator)
        spec=importlib.util.spec_from_file_location('pinned_openpi_train',ROOT/'repos/openpi/scripts/train.py')
        upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
        rng,init_rng=jax.random.split(jax.random.key(args.seed))
        print('INITIALIZING WEIGHTS',str(args.initial_checkpoint) if args.initial_checkpoint else 'public pi05 base',flush=True)
        t=time.monotonic();state,state_sharding=upstream.init_train_state(cfg,init_rng,mesh,resume=False)
        jax.block_until_ready(state);report['init_seconds']=time.monotonic()-t
        trainable=flat(state.params.filter(cfg.trainable_filter));all_params=flat(state.params)
        report['parameters']=[{'path':k,'shape':list(v.value.shape),'dtype':str(v.value.dtype),
                               'count':int(v.value.size),'trainable':k in trainable} for k,v in all_params.items()]
        report['trainable_parameters']=sum(int(v.value.size) for v in trainable.values())
        report['total_parameters']=sum(int(v.value.size) for v in all_params.values())
        if any('PaliGemma/img' in k for k in trainable): raise AssertionError('Vision backbone unexpectedly trainable')
        if not any('lora' in k and '_1' in k for k in trainable): raise AssertionError('No action-expert LoRA')
        if not any('lora' in k and '_1' not in k for k in trainable): raise AssertionError('No VLM LoRA')
        before=digest(state.params);report['stage']='training'
        (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print('TRAINABLE',report['trainable_parameters'],'TOTAL',report['total_parameters'],flush=True)
        compiled=jax.jit(functools.partial(step,cfg),in_shardings=(rep,state_sharding,ds),
                         out_shardings=(state_sharding,rep),donate_argnums=(1,))
        from datetime import datetime
        deadline=datetime.fromisoformat(args.deadline).timestamp()
        for i in range(args.steps):
            if time.time()>=deadline:break
            t=time.monotonic()
            with sharding.set_mesh(mesh):state,info=compiled(rng,state,batch)
            jax.block_until_ready(info)
            metric={k:float(v) for k,v in jax.device_get(info).items()}
            metric.update(step=i+1,seconds=time.monotonic()-t)
            if not all(np.isfinite(v) for v in metric.values()):raise FloatingPointError(metric)
            if any(v!=1 for k,v in metric.items() if k.startswith('finite_')):raise FloatingPointError(metric)
            report['metrics'].append(metric)
            if i<3 or (i+1)%10==0:
                print(json.dumps(metric),flush=True)
                (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
            batch=next(iterator)
        after=digest(state.params)
        report['changed_parameters']=[k for k in before if before[k]!=after[k]]
        report['changed_frozen_parameters']=[k for k in report['changed_parameters'] if k not in trainable]
        if report['changed_frozen_parameters']:raise AssertionError('Frozen parameters changed')
        if not report['changed_parameters']:raise AssertionError('No optimizer update occurred')
        report['frozen_hash_before']= {k:v for k,v in before.items() if k not in trainable}
        report['frozen_hash_after']= {k:v for k,v in after.items() if k not in trainable}
        report['stage']='saving'
        manager,_=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,keep_period=None,overwrite=False,resume=False)
        completed_steps=int(state.step)
        if sample_order is not None:
            consumed=sample_order[:completed_steps*args.batch_size]
            np.save(output/'consumed_sample_indices.npy',consumed)
            report['training_selection'].update(consumed_frames=len(consumed),unique_anchors=int(len(np.unique(consumed))))
        checkpoints.save_state(manager,state,loader,completed_steps);manager.wait_until_finished()
        report['checkpoint']=str(cfg.checkpoint_dir/str(completed_steps));report['completed_steps']=completed_steps
        dataset_identity=ROOT/'data/openpi'/args.dataset/'meta/project_identity.json'
        if dataset_identity.exists():
            src_identity=json.loads(dataset_identity.read_text());rig=src_identity['rig'];urdf_sha=src_identity['urdf_sha256']
        elif args.dataset=='zu7_grasp_coverage_v21_full':
            rig=json.loads((ROOT/'configs/robots/dual_zu7.json').read_text());urdf_sha=rig_urdf_hash(rig)
        else:raise ValueError('Dataset hardware identity missing')
        identity={'robot_id':rig['robot_id'],'state_dim':14,'action_dim':14,'dataset':args.dataset,
                  'rig':rig,'urdf_sha256':urdf_sha,
                  'model':'pi05','horizon':args.horizon,'discrete_state_input':True,
                  'config_name':cfg.name,'experiment':args.name,'completed_steps':completed_steps}
        identity['provenance']=report['provenance']
        if source_identity is not None:identity['initial_checkpoint']=str(args.initial_checkpoint)
        if 'training_selection' in report:identity['training_selection']=report['training_selection']
        (Path(report['checkpoint'])/'project_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        report.update(completed=True,stage='saved',deadline_shortened=completed_steps!=args.steps)
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:
        stop.set();worker.join(3)
        report['wall_seconds']=time.time()-report['started_at']
        report['gpu_observed_peak_mib']=max((x['used_mib'] for x in gpu),default=None)
        if jax.devices():report['jax_memory_stats']=jax.devices()[0].memory_stats()
        (output/'audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        (output/'gpu_memory.json').write_text(json.dumps(gpu)+'\n')
        if manager is not None:manager.close()
        print(json.dumps({k:v for k,v in report.items() if k not in ('parameters','metrics','frozen_hash_before','frozen_hash_after')}),flush=True)

if __name__=='__main__':main()
