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
    paths=set((ROOT/'results').glob('zu7*/result*.json'))
    paths.update((ROOT/'results').glob('zu7*/seed_*/results.json'))
    for path in sorted(paths):
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
                    [b'zu7',b'training_v2',b'eval_v2',b'policy_server_v2',b'dataset_v2',b'lab_',b'pi05'])]
            if names and b'python' in command[0]:zu7['active_scripts'].append({'pid':int(proc.name),'scripts':names})
        except OSError:pass
    report['zu7']=zu7
    current={'curricula':{},'pi05':{},'hardware_profile':'configs/robots/lab_archive_20261111.json'}
    for path in sorted((ROOT/'results').glob('lab_*/progress.json')):
        data=json.loads(path.read_text())
        if 'attempts' not in data:continue
        current['curricula'][str(path.relative_to(ROOT))]={k:data.get(k) for k in ('accepted','completed','stop_reason','raw_dataset')}
        current['curricula'][str(path.relative_to(ROOT))]['attempts']=len(data.get('attempts',[]))
    for path in sorted((ROOT/'results').glob('*pi05*/audit.json')):
        data=json.loads(path.read_text())
        current['pi05'][str(path.relative_to(ROOT))]={k:data.get(k) for k in ('completed','stage','completed_steps','checkpoint','error','gpu_observed_peak_mib')}
    current['pi05_evaluations']={}
    for path in sorted((ROOT/'results').glob('lab_pi05_eval_*/summary.json')):
        data=json.loads(path.read_text())
        current['pi05_evaluations'][str(path.relative_to(ROOT))]={k:data.get(k) for k in ('completed','groups','checkpoint','scope')}
    coverage=ROOT/'data/raw/lab_multitarget_20261111_merged/coverage.json'
    if coverage.exists():
        data=json.loads(coverage.read_text())
        current['accepted_dataset']={k:data[k] for k in ('episodes','frames','distinct_layouts','complete_three_target_layouts','color_counts')}
    report['archive_rig_and_pi05']=current
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
    current=['# 历史 Zu7 临时夹爪实验','',f"采样时间：{report['observed_at']}。已确认两台 Zu7；夹爪/安装为暂定仿真配置，未接真机。",'',
             f"活跃项目脚本：{json.dumps(zu7.get('active_scripts',[]),ensure_ascii=False)}",'',
             '| Zu7 实验产物 | 验收结果 |','| --- | --- |']
    for path,value in zu7.get('experiments',{}).items():
        rows=value if isinstance(value,list) else [value]
        if rows and all(x.get('complete_handover_tested') is False and 'grasp_success' in x for x in rows):
            verdict=f"抓取子任务 {sum(bool(x['grasp_success']) for x in rows)}/{len(rows)}；未测试完整交接"
            current.append(f"| {path} | {verdict} |")
        elif rows and all('success' in x for x in rows):
            verdict=f"完整任务 {sum(bool(x['success']) for x in rows)}/{len(rows)} 成功"
            if all('grasp_success' in x for x in rows):
                verdict+=f"；左抓取 {sum(bool(x['grasp_success']) for x in rows)}/{len(rows)}"
            current.append(f"| {path} | {verdict} |")
        elif rows and all('grasp_success' in x for x in rows):
            verdict=f"专家抓取子任务 {sum(bool(x['grasp_success']) for x in rows)}/{len(rows)}；未测试完整交接"
            current.append(f"| {path} | {verdict} |")
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
    active=report.get('archive_rig_and_pi05',{})
    recent=['# 实验室归档硬件与 π0.5 当前状态','',f"采样时间：{report['observed_at']}（北京时间）。",'',
            '按归档建模，尚未实机核对；夹爪行程、腕相机和光学坐标轴仍有假设。',
            '硬件与规划证据见 `docs/LAB_ARCHIVE_20261111.md`；π0.5工程验收见 `docs/PI05_QUALIFICATION_20261111.md`。','',
            f"活跃脚本：{json.dumps(zu7.get('active_scripts',[]),ensure_ascii=False)}",'',
            '| 新批次 | 当前结果 |','| --- | --- |']
    for name,data in active.get('curricula',{}).items():
        recent.append(f"| {name} | {data['accepted']} 条验收通过 / {data['attempts']} 次尝试；completed={data['completed']}；{data['stop_reason']} |")
    if not active.get('curricula'):recent.append('| 新硬件专家数据 | 尚无批量验收记录；诊断失败不能计为示教 |')
    for name,data in active.get('pi05',{}).items():
        recent.append(f"| {name} | 更新 {data['completed_steps']} 步；completed={data['completed']}；仅说明训练工程状态 |")
    if active.get('accepted_dataset'):
        data=active['accepted_dataset']
        recent.append(f"| 新硬件合并训练集 | {data['episodes']} 条 / {data['frames']} 帧 / {data['distinct_layouts']} 布局；数据充分性未确立 |")
    for name,data in active.get('pi05_evaluations',{}).items():
        groups=data.get('groups') or {}
        counted=[]
        if isinstance(groups, dict):
            for key, value in groups.items():
                if isinstance(value, dict) and "successes" in value and "episodes" in value:
                    counted.append("{} 合格抓取 {}/{}".format(key, value["successes"], value["episodes"]))
        text="；".join(counted) or "未提供可汇总分组计数"
        recent.append("| {} | {}；completed={}；小样本开发诊断 |".format(name, text, data.get("completed")))
    return '\n'.join(recent+['','以下为旧硬件及 ALOHA 历史记录。','']+lines)


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
