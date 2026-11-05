"""Inspect the bounded pilot; optionally save a documentation snapshot.

Uses only the standard library. Never launches or interrupts experiments.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def tail(path, limit=20000):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return ANSI.sub('', stream.read().decode(errors='replace')).replace('\r', '\n')


def collect_status():
    stage_path = ROOT / 'results/pilot_status.txt'
    pid_path = ROOT / 'results/pilot.pid'
    report = {
        'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'stage': stage_path.read_text().strip() if stage_path.exists() else 'not_started',
        'pid': None, 'process_exists': False, 'process_running': False,
        'process_state': None, 'process_matches_pipeline': False,
    }
    if pid_path.exists():
        try:
            report['pid'] = int(pid_path.read_text().strip())
            proc = Path('/proc') / str(report['pid'])
            report['process_state'] = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
            report['process_exists'] = True
            command = (proc / 'cmdline').read_bytes().split(b'\0')
            report['process_matches_pipeline'] = os.fsencode(ROOT / 'scripts/pilot_pipeline.sh') in command
            report['process_running'] = report['process_matches_pipeline'] and report['process_state'] not in ('Z', 'X')
        except (OSError, ValueError, IndexError) as exc:
            report['process_check_note'] = type(exc).__name__
    raw = ROOT / 'data/raw/jaka_pilot/handover_block/aloha_agilex'
    seeds = raw / 'seed.txt'
    report.update(
        planned_episodes=len(seeds.read_text().split()) if seeds.exists() else 0,
        saved_episodes=len(list((raw / 'data').glob('*.hdf5'))),
        disk_free_gib=shutil.disk_usage(ROOT).free / 2**30,
        checkpoint_present=(ROOT / 'checkpoints/smolvla_pilot_seed42/checkpoints/last/pretrained_model/model.safetensors').exists(),
    )
    for name in ('collect-pilot', 'convert-pilot', 'train-pilot', 'eval-baselines'):
        content = tail(ROOT / f'logs/{name}.log')
        if content:
            report[f'{name}_last_lines'] = [line for line in content.splitlines() if line][-3:]
        if name == 'train-pilot':
            progress = re.findall(r'Training:\s*\d+%[^\n]*?\s(\d+)/2000\b', content)
            report['training_steps_observed'] = int(progress[-1]) if progress else None
            metrics = [line for line in content.splitlines() if 'step:' in line and 'loss:' in line]
            report['training_last_metric'] = metrics[-1] if metrics else None
    artifacts = {
        'dataset_validation': 'results/handover_pilot_validation.json',
        'training_resources': 'checkpoints/smolvla_pilot_seed42/resource_profile.json',
        'fixed_k_results': 'results/pilot_baselines.json',
    }
    for key, relative in artifacts.items():
        path = ROOT / relative
        if path.exists():
            try:
                value = json.loads(path.read_text())
                if key == 'dataset_validation':
                    value = {'episodes': len(value), 'frames': sum(ep['frames'] for ep in value),
                             'fps': sorted({ep['fps'] for ep in value}),
                             'state_semantics': sorted({ep['state_semantics'] for ep in value})}
                report[key] = value
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report.setdefault('artifact_read_notes', {})[key] = type(exc).__name__
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        report['gpu_snapshot'] = gpu.stdout.strip() if gpu.returncode == 0 else 'unavailable'
    except (OSError, subprocess.TimeoutExpired):
        report['gpu_snapshot'] = 'unavailable'
    zu7={'raw_datasets':{},'experiments':{},'training':{},'training_logs':{},'active_scripts':[]}
    for path in sorted((ROOT/'data/raw').glob('zu7*/handover_block/*/robot_identity.json')):
        zu7['raw_datasets'][str(path.parent.relative_to(ROOT))]={'episodes':len(list((path.parent/'data').glob('*.hdf5'))),
                'identity':json.loads(path.read_text())}
    for path in sorted((ROOT/'results').glob('zu7*/result*.json')):
        zu7['experiments'][str(path.relative_to(ROOT))]=json.loads(path.read_text())
    for path in sorted((ROOT/'checkpoints').glob('smolvla_zu7*/resource_profile.json')):
        zu7['training'][str(path.relative_to(ROOT))]=json.loads(path.read_text())
    for path in sorted((ROOT/'logs').glob('train-zu7*.log')):
        content=tail(path)
        progress=re.findall(r'Training:\s*\d+%[^\n]*?\s(\d+)/(\d+)\b',content)
        if progress:zu7['training_logs'][path.name]={'step':int(progress[-1][0]),'total_steps':int(progress[-1][1])}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command=(proc/'cmdline').read_bytes().split(b'\0')
            # Record only project script names, never shell commands/connection arguments.
            names=[os.fsdecode(x) for x in command if x.startswith(b'scripts/') and any(t in x for t in
                    [b'zu7',b'training_v2',b'eval_v2',b'policy_server_v2',b'dataset_v2'])]
            if names and b'python' in command[0]:zu7['active_scripts'].append({'pid':int(proc.name),'scripts':names})
        except OSError:pass
    report['zu7']=zu7
    return report


def render_markdown(report):
    stages = {'collecting': '采集示教', 'converting': '转换与校验数据', 'training': '训练基础策略',
              'evaluating_fixed_k': '固定 K 闭环评测',
              'pilot_complete_review_results_before_scaling': '当前批次结束，待复核结果'}
    stage = stages.get(report['stage'], report['stage'])
    progress = report['training_steps_observed']
    lines = ['# 当前运行状态（自动生成快照）', '',
             f"采样时间：{report['observed_at']}（北京时间）。", '',
             '此文件不是实时监控；刷新命令 `python3 /data/jaka-vla/scripts/status.py --write-doc`。',
             '阶段标记可能滞后，须结合进程和已落盘结果判断。', '',
             '| 项目 | 已观察到的状态 |', '| --- | --- |',
             f'| 阶段标记 | {stage} |',
             f"| 流水线 PID / 进程状态 | {report['pid']} / {report['process_state']} |",
             f"| 匹配流水线且非僵尸进程 | {report['process_running']} |",
             f"| 原始示教已保存 / 规划种子 | {report['saved_episodes']} / {report['planned_episodes']} |",
             f"| 训练进度（日志最近值） | {str(progress) + ' / 2000' if progress is not None else '尚未观察到训练步数'} |",
             f"| 最终检查点权重文件存在 | {report['checkpoint_present']} |",
             f"| 数据盘可用 | {report['disk_free_gib']:.2f} GiB |",
             f"| GPU 快照 | {report['gpu_snapshot'].replace(chr(10), '; ')} |", '']
    zu7=report.get('zu7',{})
    current=['# 当前 Zu7 修正批次','',f"采样时间：{report['observed_at']}。已确认两台 Zu7；夹爪/安装为暂定仿真配置，未接真机。",'',
             f"活跃项目脚本：{json.dumps(zu7.get('active_scripts',[]),ensure_ascii=False)}",'',
             '| Zu7 实验产物 | 验收结果 |','| --- | --- |']
    for path,value in zu7.get('experiments',{}).items():
        rows=value if isinstance(value,list) else [value]
        if rows and all('success' in x for x in rows):current.append(f"| {path} | {sum(bool(x['success']) for x in rows)}/{len(rows)} 成功 |")
    for path,value in zu7.get('raw_datasets',{}).items():current.append(f"| {path} | {value['episodes']} 条原始示教 |")
    for path,value in zu7.get('training',{}).items():current.append(f"| {path} | completed={value['completed']} |")
    for path,value in zu7.get('training_logs',{}).items():current.append(f"| {path} | 已观察到 {value['step']}/{value['total_steps']} 步 |")
    current+=['','下面为已结束的 ALOHA 历史 pilot 快照。不能当作当前 Zu7 实验状态。','']
    lines=current+lines
    validation = report.get('dataset_validation')
    if validation:
        lines.append(f"转换验证报告：{validation['episodes']} 条、{validation['frames']} 帧，频率 {validation['fps']} Hz，状态语义 {validation['state_semantics']}。")
    else:
        lines.append('尚未读到完整数据转换验证报告；已保存 HDF5 数量不等于转换完成数量。')
    resources = report.get('training_resources')
    if resources:
        lines.extend(['', f"训练资源报告：completed={resources.get('completed')}；峰值分配 {resources.get('peak_allocated_gib')} GiB；包含加载/保存的用时 {resources.get('wall_seconds_including_load_and_save')} 秒。"])
    results = report.get('fixed_k_results', [])
    lines.extend(['', '## 已写入的固定 K 汇总', ''])
    if results:
        lines.extend(['| K | 成功/回合数 | 成功率 |', '| --- | --- | --- |'])
        for item in results:
            lines.append(f"| {item['k']} | {item['successes']}/{item['episodes']} | {item['success_rate']:.1%} |")
        lines.append('\n仅展示汇总文件中已完成的项目；逐回合证据在 `results/pilot_k*/results.json`，不得将部分结果当作完整批次。')
    else:
        lines.append('尚未读到固定 K 完整组的成功率结果。训练完成、检查点存在均不等于任务成功。')
    lines.extend(['', '当前是同步、零模拟推理延迟评测；不是事件监测器、真实延迟或节卡真机实验。', '',
                  '## 证据入口', '',
                  '- `results/project_status.json`：本次机器可读快照。',
                  '- `results/pilot_status.txt`、`results/pilot.pid`：流水线标记；必须交叉核实。',
                  '- `logs/collect-pilot.log`、`logs/convert-pilot.log`、`logs/train-pilot.log`、`logs/eval-baselines.log`：各阶段日志。',
                  '- `results/handover_pilot_validation.json`、`checkpoints/smolvla_pilot_seed42/resource_profile.json`、`results/pilot_baselines.json`：阶段结束后生成。',
                  '- `docs/EXPERIMENT_LOG.md`：历史验证、实验参数和结论边界。', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-doc', action='store_true', help='Save docs/STATUS.md and results/project_status.json')
    args = parser.parse_args()
    report = collect_status()
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.write_doc:
        for path, content in ((ROOT / 'docs/STATUS.md', render_markdown(report)),
                              (ROOT / 'results/project_status.json', serialized + '\n')):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(content, encoding='utf-8')
            temporary.replace(path)
    print(serialized)


if __name__ == '__main__':
    main()
