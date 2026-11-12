"""Continue pi0.5 fine-tuning from a full OpenPI training checkpoint.

The previous experiment stopped after a bounded run. This utility restores the
optimizer and model state, writes to a new experiment directory, and records a
separate audit so the original checkpoint remains untouched.
"""
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

ROOT = Path('/data/jaka-vla')
os.environ.setdefault('HF_LEROBOT_HOME', str(ROOT / 'data/openpi'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('WANDB_MODE', 'disabled')
sys.path.insert(0, str(ROOT))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.training import checkpoints, data_loader, sharding
from jaka_vla.identity import rig_urdf_hash
from jaka_vla.pi05_config import build_config


def flat(state):
    return {'/'.join(map(str, key)): value for key, value in state.flat_state().items()}


def digest(state):
    return {name: hashlib.sha256(np.asarray(value.value).tobytes()).hexdigest()
            for name, value in flat(state).items()}


def group(name):
    if 'PaliGemma/llm' in name:
        return 'action_expert_lora' if '_1' in name else 'vlm_lora'
    return name.split('/')[0]


def step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(m):
        return jnp.mean(m.compute_loss(jax.random.fold_in(rng, state.step), batch[0], batch[1], train=True))

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(model)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    info = {'loss': loss, 'grad_norm': optax.global_norm(grads), 'update_norm': optax.global_norm(updates)}
    groups = {}
    for name, value in flat(grads).items():
        groups.setdefault(group(name), []).append(value.value)
    for name, values in groups.items():
        info['grad_' + name] = optax.global_norm(values)
        info['finite_' + name] = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in values]))
    return dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000, help='Additional updates, not total updates')
    parser.add_argument('--schedule-steps', type=int, default=6000,
                        help='Must match the source run so restored optimizer moments keep the same LR schedule')
    parser.add_argument('--seed', type=int, default=43,
                        help='Independent data-order seed; do not reuse the source run seed')
    parser.add_argument('--deadline', default=None, help='Optional ISO-8601 deadline with timezone')
    args = parser.parse_args()
    if args.steps < 1 or args.schedule_steps < 1 or not all(c.isalnum() or c in '_-' for c in args.name):
        raise ValueError('Invalid run arguments')

    source = args.source_checkpoint.resolve()
    source_identity = json.loads((source / 'project_identity.json').read_text())
    if source_identity['dataset'] != args.dataset or source_identity['model'] != 'pi05':
        raise ValueError('Source checkpoint dataset/model identity mismatch')
    source_audit_path = ROOT / 'results' / source_identity['experiment'] / 'audit.json'
    source_audit = json.loads(source_audit_path.read_text()) if source_audit_path.exists() else {}
    source_data_seed = source_audit.get('data_order_seed', source_audit.get('arguments', {}).get('data_seed', 42))
    if args.seed == source_data_seed:
        raise ValueError(f'Data-order seed {args.seed} repeats the source run seed')
    config = dataclasses.replace(build_config(args.dataset, args.name, steps=args.schedule_steps,
                                               batch_size=1, horizon=source_identity['horizon']), seed=args.seed)
    output = ROOT / 'results' / args.name
    output.mkdir(parents=True, exist_ok=False)
    report = {'arguments': vars(args) | {'source_checkpoint': str(source)}, 'completed': False,
              'stage': 'initializing', 'metrics': [], 'source_checkpoint': str(source),
              'scope': 'continued pi0.5 fine-tuning from full model+optimizer state; loss is not task success',
              'data_order_seed': args.seed, 'source_data_order_seed': source_data_seed,
              'started_at': time.time()}
    provenance_paths = [ROOT / 'scripts/continue_pi05_zu7.py', ROOT / 'scripts/train_pi05_zu7.py',
                        ROOT / 'jaka_vla/pi05_config.py', ROOT / 'jaka_vla/openpi_zu7.py']
    report['provenance'] = {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'file_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in provenance_paths if path.exists()},
    }

    stop = threading.Event()
    gpu = []

    def monitor():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                gpu.append({'time': time.time(), 'used_mib': int(result.stdout.splitlines()[0])})
            stop.wait(1.)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    source_manager = target_manager = None
    try:
        jax.config.update('jax_compilation_cache_dir', str(ROOT / 'cache/jax-pi05'))
        mesh = sharding.make_mesh(1)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        iterator = iter(loader)
        batch = next(iterator)

        upstream_path = ROOT / 'repos/openpi/scripts/train.py'
        spec = importlib.util.spec_from_file_location('pinned_openpi_train_continue', upstream_path)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        rng, init_rng = jax.random.split(jax.random.key(42))
        state, state_sharding = upstream.init_train_state(config, init_rng, mesh, resume=True)

        source_manager, source_resuming = checkpoints.initialize_checkpoint_dir(
            source.parent, keep_period=None, overwrite=False, resume=True)
        if not source_resuming or int(source.name) not in source_manager.all_steps():
            raise ValueError('Source checkpoint is not present in its training-state manager')
        state = checkpoints.restore_state(source_manager, state, loader, step=int(source.name))
        jax.block_until_ready(state)
        start_step = int(state.step)
        if start_step != int(source.name):
            raise ValueError(f'Restored step {start_step} does not match checkpoint directory {source.name}')
        before = digest(state.params)
        trainable = flat(state.params.filter(config.trainable_filter))
        report.update(source_step=start_step, trainable_parameters=sum(int(value.value.size) for value in trainable.values()),
                      total_parameters=sum(int(value.value.size) for value in flat(state.params).values()),
                      stage='training')
        (output / 'audit.json').write_text(json.dumps(report, indent=2, default=str) + '\n')

        compiled = jax.jit(functools.partial(step, config), in_shardings=(replicated, state_sharding, data_sharding),
                           out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        deadline = None if args.deadline is None else __import__('datetime').datetime.fromisoformat(args.deadline).timestamp()
        for local_step in range(args.steps):
            if deadline is not None and time.time() >= deadline:
                break
            started = time.monotonic()
            with sharding.set_mesh(mesh):
                state, info = compiled(rng, state, batch)
            jax.block_until_ready(info)
            metric = {key: float(value) for key, value in jax.device_get(info).items()}
            metric.update(step=int(state.step), additional_step=local_step + 1,
                          seconds=time.monotonic() - started)
            if not all(np.isfinite(value) for value in metric.values()):
                raise FloatingPointError(metric)
            if any(value != 1 for key, value in metric.items() if key.startswith('finite_')):
                raise FloatingPointError(metric)
            report['metrics'].append(metric)
            if local_step < 3 or (local_step + 1) % 10 == 0:
                print(json.dumps(metric), flush=True)
                (output / 'audit.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
            batch = next(iterator)

        after = digest(state.params)
        changed = [key for key in before if before[key] != after[key]]
        changed_frozen = [key for key in changed if key not in trainable]
        if changed_frozen:
            raise AssertionError('Frozen parameters changed')
        if not changed:
            raise AssertionError('No optimizer update occurred')
        report['changed_parameters'] = changed
        report['changed_frozen_parameters'] = changed_frozen
        report['frozen_hash_before'] = {key: value for key, value in before.items() if key not in trainable}
        report['frozen_hash_after'] = {key: value for key, value in after.items() if key not in trainable}

        target_manager, _ = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir, keep_period=None, overwrite=False, resume=False)
        report['stage'] = 'saving'
        completed_steps = int(state.step)
        checkpoints.save_state(target_manager, state, loader, completed_steps)
        target_manager.wait_until_finished()
        checkpoint = config.checkpoint_dir / str(completed_steps)
        rig, urdf_sha = source_identity['rig'], source_identity['urdf_sha256']
        identity = dict(source_identity)
        identity.update(config_name=config.name, experiment=args.name, completed_steps=completed_steps,
                        parent_checkpoint=str(source), continuation_updates=completed_steps - start_step)
        identity['provenance'] = report['provenance']
        (checkpoint / 'project_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        report.update(checkpoint=str(checkpoint), completed_steps=completed_steps,
                      additional_updates=completed_steps - start_step, deadline_shortened=(completed_steps - start_step < args.steps),
                      stage='saved', completed=True)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        stop.set()
        worker.join(3)
        report['wall_seconds'] = time.time() - report['started_at']
        report['gpu_observed_peak_mib'] = max((item['used_mib'] for item in gpu), default=None)
        if jax.devices():
            report['jax_memory_stats'] = jax.devices()[0].memory_stats()
        (output / 'audit.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        (output / 'gpu_memory.json').write_text(json.dumps(gpu) + '\n')
        if target_manager is not None:
            target_manager.close()
        if source_manager is not None:
            source_manager.close()
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ('metrics', 'frozen_hash_before', 'frozen_hash_after')}, default=str), flush=True)


if __name__ == '__main__':
    main()
