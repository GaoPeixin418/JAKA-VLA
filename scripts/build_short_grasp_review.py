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
    args = p.parse_args()
    if args.output.exists() or args.archive.exists():
        raise FileExistsError('Review output already exists')
    args.output.mkdir(parents=True)
    cards = []
    for source in args.experiments:
        target = args.output/source.name
        target.mkdir()
        report_path = source/'results.json'
        if not report_path.exists():
            report_path = source/'result.json'
        data = json.loads(report_path.read_text())
        rows = data if isinstance(data, list) else [data]
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.mp4', '.npz', '.json', '.py'):
                if path.suffix != '.mp4' or path.name.endswith('_multiview.mp4'):
                    shutil.copy2(path, target/path.name)
        for row in rows:
            seed = row['seed']
            video = target/f'seed_{seed}_multiview.mp4'
            if not video.is_file():
                raise FileNotFoundError(video)
            expert = 'mode' in row and ('expert' in row['mode'])
            label = '专家参考' if expert else '学习策略 SmolVLA'
            grasp = '抓取判据通过' if row.get('grasp_success') else '未通过抓取判据'
            failure = row.get('failure') or row.get('failure_reason')
            path_length = row.get('preclose_tcp_path_m')
            extra = f'；抓取前TCP行程 {path_length:.3f}m' if path_length is not None else ''
            full = '；未测试完整交接' if row.get('complete_handover_tested') is False else f'；完整交接成功：{row.get("success", False)}'
            cards.append(f'''<section><h2>{html.escape(label)} · seed {seed}</h2>
<p>{html.escape(source.name)}</p><p>{grasp}{extra}{full}</p>
<video controls preload="metadata" src="{source.name}/{video.name}"></video>
<div><button onclick="step(this,-1)">上一帧</button><button onclick="step(this,1)">下一帧</button>
<button onclick="speed(this,.5)">0.5倍速</button><button onclick="speed(this,1)">正常速度</button></div>
<p>最长连续有效夹持：{row.get('longest_verified_hold_s', row.get('longest_verified_left_grasp_s', 0)):.3f} 秒；
仿真时长：{row.get('sim_seconds', 0):.3f} 秒。</p><p>终止/失败：{html.escape(str(failure))}</p>
<a href="{source.name}/{report_path.name}">查看原始指标</a></section>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Zu7 短路径抓取实验</title>
<style>body{font:17px/1.7 sans-serif;background:#edf2f7;color:#172638;max-width:1280px;margin:auto;padding:28px}section{background:white;border-radius:12px;padding:24px;margin:24px 0}video{width:100%;background:#172638}button{padding:8px 16px;margin:8px}h1{margin-bottom:8px}p{overflow-wrap:anywhere}</style>
<h1>Zu7 短路径抓取实验 · 2026-11-06</h1>
<p>任务：左臂抓红色长方块、抬升、保持。蓝色块是接收台。专家、专家控制器回放、学习策略分别标注；子任务成功不等于完整交接或泛化。</p>
<p>录像包含全景、俯视、头部与左腕；默认半速。片头2秒与片尾3秒标注为冻结画面，不计入夹持时长。保留失败尝试。</p>''' + '\n'.join(cards) + '''<script>document.querySelectorAll('video').forEach(v=>v.playbackRate=.5);function step(b,d){let v=b.closest('section').querySelector('video');v.pause();v.currentTime=Math.max(0,v.currentTime+d/25)}function speed(b,s){b.closest('section').querySelector('video').playbackRate=s}</script></html>'''
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
