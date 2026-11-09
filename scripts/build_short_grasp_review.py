"""Portable, labelled short-path diagnostic / learned-grasp video review."""
import argparse
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments', nargs='+', type=Path, required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--evidence', nargs='*', type=Path, default=[])
    args = p.parse_args()
    if args.output.exists() or args.archive.exists():
        raise FileExistsError('Review output already exists')
    args.output.mkdir(parents=True)
    cards = []
    summary_rows = []
    for source in args.experiments:
        target = args.output/source.name
        target.mkdir()
        report_path = source/'results.json'
        if not report_path.exists():
            report_path = source/'result.json'
        data = json.loads(report_path.read_text())
        rows = data if isinstance(data, list) else [data]
        by_seed = {int(row['seed']): row for row in rows}
        summary_rows.append((source.name, by_seed))
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.mp4', '.npz', '.json', '.py', '.png'):
                if path.suffix != '.mp4' or path.name.endswith('_multiview.mp4'):
                    shutil.copy2(path, target/path.name)
        for row in rows:
            seed = row['seed']
            video = target/f'seed_{seed}_multiview.mp4'
            if not video.is_file():
                raise FileNotFoundError(video)
            expert = 'mode' in row and ('expert' in row['mode'])
            label = '专家参考' if expert else '学习策略 SmolVLA'
            if row.get('initial_condition',{}).get('scope')=='intervened_recovery_start_not_executed_prefix':
                label+=' · 重设中途初态后的恢复子任务'
            elif '_matched_' in source.name:
                label+=' · 未见种子 + ±0.04rad 初态扰动'
            elif '_new_seed' in source.name:
                label+=' · 新场景及初态扰动'
            elif '_seen_seed' in source.name:
                label+=' · 已见场景名义初态'
            grasp = '抓取判据通过' if row.get('grasp_success') else '未通过抓取判据'
            failure = row.get('failure') or row.get('failure_reason')
            path_length = row.get('preclose_tcp_path_m')
            extra = f'；抓取前TCP行程 {path_length:.3f}m' if path_length is not None else ''
            full = '；未测试完整交接' if row.get('complete_handover_tested') is False else f'；完整交接成功：{row.get("success", False)}'
            motion = ''
            if (target/'control_metrics.json').is_file():
                motion = f'<p><a href="{source.name}/control_metrics.json">查看指令与实测运动指标</a></p>'
            plot = target/f'seed_{seed}_control.png'
            if plot.is_file():
                motion += f'<details><summary>速度、跟踪与夹爪曲线</summary><img style="width:100%" src="{source.name}/{plot.name}"></details>'
            cards.append(f'''<section><h2>{html.escape(label)} · seed {seed}</h2>
<p>{html.escape(source.name)}</p><p>{grasp}{extra}{full}</p>
<video controls preload="metadata" src="{source.name}/{video.name}"></video>
<div><button onclick="step(this,-1)">上一帧</button><button onclick="step(this,1)">下一帧</button>
<button onclick="speed(this,.5)">0.5倍速</button><button onclick="speed(this,1)">正常速度</button></div>
<p>最长连续有效夹持：{row.get('longest_verified_hold_s', row.get('longest_verified_left_grasp_s', 0)):.3f} 秒；
仿真时长：{row.get('sim_seconds', 0):.3f} 秒。</p><p>终止/失败：{html.escape(str(failure))}</p>
<a href="{source.name}/{report_path.name}">查看原始指标</a>{motion}</section>''')
    evidence_dir = args.output/'evidence'
    evidence_dir.mkdir()
    evidence_links = []
    for source in args.evidence:
        target = evidence_dir/source.name
        shutil.copy2(source, target)
        evidence_links.append(f'<li><a href="evidence/{html.escape(source.name)}">{html.escape(source.name)}</a></li>')
    summary_table = '<table><thead><tr><th>检查点</th><th>抓取成功</th>' + ''.join(
        f'<th>seed {seed}</th>' for seed in sorted({seed for _, rows in summary_rows for seed in rows})
    ) + '</tr></thead><tbody>'
    for name, by_seed in summary_rows:
        successes = sum(bool(row.get('grasp_success')) for row in by_seed.values())
        cells = ''.join(
            f'<td>{"成功" if by_seed[seed].get("grasp_success") else html.escape(str((by_seed[seed].get("failure_reason") or {}).get("type", "失败")))}</td>'
            if seed in by_seed else '<td>—</td>'
            for seed in sorted({s for _, rows in summary_rows for s in rows})
        )
        summary_table += f'<tr><td>{html.escape(name)}</td><td>{successes}/{len(by_seed)}</td>{cells}</tr>'
    summary_table += '</tbody></table>'
    links_html = '<ul>' + ''.join(evidence_links) + '</ul>' if evidence_links else ''
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Zu7 配对检查点抓取对比 · 2026-11-09</title>
<style>body{font:17px/1.7 sans-serif;background:#edf2f7;color:#172638;max-width:1280px;margin:auto;padding:28px}section{background:white;border-radius:12px;padding:24px;margin:24px 0}video{width:100%;background:#172638}button{padding:8px 16px;margin:8px}h1{margin-bottom:8px}p{overflow-wrap:anywhere}</style>
<h1>Zu7 配对检查点抓取对比 · 2026-11-09</h1>
<p>同一任务指令、相同5个新种子（203–207）、相同左臂初态±0.04rad偏移、K=8、固定噪声seed42、每回合最多250步，比较stable1500、coverage3000与recovery4000三个检查点。属于小样本开发诊断，不是盲测；不能据此声称统计显著或泛化提升。</p>
<p>任务：左臂抓红色方块、抬升并保持。仅测抓取子任务，未测试双臂交接、其他物体、语言选择或真机。评测为同步仿真，推理期间暂停仿真。</p>
<h2>逐种子结果</h2>''' + summary_table + '<h2>可下载的审计证据</h2>' + links_html + '''
<p>录像包含全景、俯视、头部与左腕；默认半速。片头2秒与片尾3秒标注为冻结画面，不计入夹持时长。保留失败尝试。</p>
<p>抓取判据：物体抬升超过5厘米，实测夹爪开度15%–85%，两指均有有效接触且无右臂接触，连续0.6秒。达到判据即结束本抓取子任务；不能由片尾静帧推断更长的持物能力。</p>
<p>若标注“重设中途初态”，该回合在采样前将机器人置于已有轨迹的中途姿态；它只验证此初态的恢复子任务，未执行原策略前缀。</p>''' + '\n'.join(cards) + '''<script>document.querySelectorAll('video').forEach(v=>v.playbackRate=.5);function step(b,d){let v=b.closest('section').querySelector('video');v.pause();v.currentTime=Math.max(0,v.currentTime+d/25)}function speed(b,s){b.closest('section').querySelector('video').playbackRate=s}</script></html>'''
    (args.output/'index.html').write_text(page)
    sums = {str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(args.output.rglob('*')) if f.is_file()}
    (args.output/'SHA256.json').write_text(json.dumps(sums, indent=2)+'\n')
    with zipfile.ZipFile(args.archive, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                archive.write(path, Path(args.output.name)/path.relative_to(args.output))
    print(json.dumps({'archive': str(args.archive), 'bytes': args.archive.stat().st_size,
                      'sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
