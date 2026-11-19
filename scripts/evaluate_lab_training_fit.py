"""Small in-sample diagnostic; never report as held-out generalization."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/data/jaka-vla')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--episodes', type=int, nargs='+', required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--max-steps', type=int, default=650)
    p.add_argument('--k', type=int, default=8)
    a = p.parse_args()
    if not a.name.startswith('lab_pi05_fit_') or not all(c.isalnum() or c == '_' for c in a.name):
        raise ValueError('Fit diagnostic name must start with lab_pi05_fit_')
    if len(set(a.episodes)) != len(a.episodes):
        raise ValueError('Duplicate episode index')
    deadline = datetime.fromisoformat(a.deadline).timestamp()
    run = ROOT / 'results' / a.name
    run.mkdir(parents=True, exist_ok=False)
    (run / 'scenes').mkdir()
    checkpoint = (ROOT / a.checkpoint).resolve() if not a.checkpoint.is_absolute() else a.checkpoint.resolve()
    identity_path = ROOT / 'data/openpi' / a.dataset / 'meta/project_identity.json'
    identity = json.loads(identity_path.read_text())
    info = json.loads((ROOT / 'data/openpi' / a.dataset / 'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (ROOT / 'data/openpi' / a.dataset / 'meta/episodes.jsonl').read_text().splitlines()]
    ident_episodes = identity['scenes_by_episode']
    conditions = identity['initial_conditions_by_episode']
    selected = []
    for index in a.episodes:
        if not 0 <= index < len(episodes):
            raise ValueError(f'Episode index out of range: {index}')
        scene = ident_episodes[str(index)]
        if scene.get('split') != 'train':
            raise ValueError(f'Fit diagnostic requires a training scene: episode {index}')
        selected.append((index, scene, episodes[index], conditions[str(index)]))
    report = {'checkpoint': str(checkpoint), 'dataset': a.dataset, 'scope': 'in-sample training-layout fit diagnostic; no generalization claim',
              'episodes': [], 'completed': False, 'conditions': {'K': a.k, 'H': 50, 'noise_seed': 42, 'max_steps': a.max_steps}}
    socket = ROOT / 'run' / f'{a.name}.sock'
    if socket.exists():
        raise FileExistsError(socket)
    rig = json.loads((ROOT / 'configs/robots/dual_zu7_lab.json').read_text())
    service_env = os.environ.copy()
    service_env.update(HF_LEROBOT_HOME=str(ROOT / 'data/openpi'), OPENPI_DATA_HOME='/data/pi05-cache/models',
                       XLA_PYTHON_CLIENT_PREALLOCATE='false', WANDB_MODE='disabled',
                       PYTHONPATH=':'.join(map(str, [ROOT, ROOT/'repos/openpi/src', ROOT/'repos/openpi/packages/openpi-client/src'])))
    with (run / 'server.log').open('w') as server_log:
        server = subprocess.Popen(['/data/pi05-env/bin/python', '-u', 'scripts/policy_server_pi05.py',
                                   '--checkpoint', str(checkpoint), '--robot-id', rig['robot_id'], '--socket', str(socket),
                                   '--noise-mode', 'fixed', '--noise-seed', '42', '--connections', str(len(selected))],
                                  cwd=ROOT, env=service_env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic() + 360
            while 'READY' not in (run/'server.log').read_text():
                if server.poll() is not None:
                    raise RuntimeError('Policy service exited before READY')
                if time.monotonic() > start:
                    raise TimeoutError('Policy service startup exceeded 360 s')
                if time.time() >= deadline:
                    raise TimeoutError('Deadline reached during service startup')
                time.sleep(.25)
            for ordinal, (index, scene, ep, condition) in enumerate(selected):
                if time.time() >= deadline:
                    break
                spec = dict(scene)
                spec['split'] = 'train_fit_diagnostic'
                spec['instruction'] = ep['tasks'][0]
                spec['scope'] = 'left_grasp_lift_hold'
                spec_path = run/'scenes'/f'episode_{index:04d}.json'
                spec_path.write_text(json.dumps(spec, indent=2) + '\n')
                condition_path = run/'scenes'/f'episode_{index:04d}_initial.json'
                condition_path.write_text(json.dumps(condition, indent=2) + '\n')
                seed = 81000 + index
                output = run/f'episode_{index:04d}'
                cmd = [str(ROOT/'envs/robotwin/bin/python'), '-u', 'scripts/eval_v2.py', '--robot-id', rig['robot_id'],
                       '--rig-config', 'configs/robots/dual_zu7_lab.json', '--scene-spec', str(spec_path),
                       '--initial-condition-file', str(condition_path), '--seed-start', str(seed), '--episodes', '1',
                       '--socket', str(socket), '--output', str(output), '--k', str(a.k), '--max-steps', str(a.max_steps),
                       '--review-video', '--instruction', spec['instruction'], '--task-scope', 'left_grasp_lift_hold',
                       '--policy-label', 'pi0.5-fit-diagnostic']
                with (run/f'episode_{index:04d}.log').open('w') as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=min(700, max(1, deadline-time.time())), check=True)
                result = json.loads((output/'results.json').read_text())[0]
                row = {'episode_index': index, 'layout': scene['layout_id'], 'target': scene['target_index'],
                       'split': 'train_fit_diagnostic', 'grasp_success': result['grasp_success'],
                       'failure_reason': result['failure_reason'], 'steps': result['steps'],
                       'oracle_max_object_lift_m': result.get('oracle_max_object_lift_m'),
                       'result_path': str(output/'results.json'), 'video': str(output/f'seed_{seed}.mp4')}
                report['episodes'].append(row)
                (run/'summary.json').write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(row), flush=True)
            report['completed'] = len(report['episodes']) == len(selected)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
            report['qualified_successes'] = sum(bool(x['grasp_success'] and not x['failure_reason']) for x in report['episodes'])
            report['summary'] = f"{report['qualified_successes']}/{len(report['episodes'])} in-sample diagnostic episodes passed"
            (run/'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
