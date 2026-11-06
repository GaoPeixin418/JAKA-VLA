"""Build an offline player with full recordings, phase markers and frame stepping."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path('/data/jaka-vla')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.archive.exists():
        raise FileExistsError(args.archive)
    cards = []; all_rows = []; checks = []
    for name in args.experiments:
        if Path(name).name != name:
            raise ValueError('Expected experiment folder names')
        source = ROOT / 'results' / name
        rows = json.loads((source / 'results.json').read_text())
        dest = args.output / name; dest.mkdir()
        for pattern in ('results.json', 'run_config.json', 'trace_schema.json', '*_trace.npz', '*_contacts.json'):
            for path in source.glob(pattern):
                shutil.copy2(path, dest / path.name)
        for row in rows:
            seed = row['seed']; meta = row['review_video']
            video = source / meta['file']; shutil.copy2(video, dest / video.name)
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration', '-of', 'json', str(video)], text=True))['streams'][0]
            assert [info['width'], info['height']] == meta['resolution']
            assert int(info['nb_frames']) == meta['frames']
            assert info['r_frame_rate'] == '25/1'
            assert abs(float(info['duration']) - meta['duration_s']) < .05
            checks.append({'file': str(video.relative_to(ROOT)), 'sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'probe': info})
            learned = row['mode'] != 'expert_replay'
            kind = 'SmolVLA 学习策略' if learned else '专家动作回放（仅参考）'
            grasp = '抓取通过' if row.get('grasp_success') else '抓取未通过'
            full = '完整任务通过' if row['success'] else '完整任务未通过'
            failure = row.get('failure_reason') or {}
            reason = {'empty_grasp': '空抓保护', 'persistent_tracking_error': '持续跟踪异常保护',
                      'step_budget_exhausted': '达到回合时长上限'}.get(failure.get('type'), '任务完成')
            prefix = meta['intro_freeze_s']
            markers = [(0., '初始画面'), (prefix, '动作开始')]
            if row.get('grasp_success_time_s') is not None:
                markers += [(prefix + max(0., row['grasp_success_time_s'] - 1.), '抓取过程'),
                            (prefix + row['grasp_success_time_s'], '抓取确认')]
            markers += [(prefix + max(0., row['sim_seconds'] - 1.), '终止前一秒')]
            buttons = ''.join(f'<button data-seek="{t:.3f}">{text}</button>' for t, text in markers)
            checkpoint = ''
            if (source / 'run_config.json').exists():
                cfg = json.loads((source / 'run_config.json').read_text())
                checkpoint = Path(cfg['checkpoint']).parents[2].name
            card = f'''<article><div class="heading"><span class="badge">{kind}</span><h2>Seed {seed} · {grasp}</h2>
<p>{full} · {reason}</p></div>
<video controls playsinline preload="metadata" src="{name}/{video.name}"></video>
<div class="controls"><button data-frame="-1">上一帧</button><button data-frame="1">下一帧</button>
<label>播放速度 <select><option value="0.25">0.25 倍</option><option value="0.5" selected>0.5 倍</option><option value="1">1 倍</option></select></label>
<button data-fullscreen>全屏</button><span class="clock"></span></div>
<div class="markers">{buttons}</div>
<p>有效仿真动作 {row['sim_seconds']:.2f} 秒；片头静帧 2 秒、片尾静帧 3 秒，均有标注。半速播放约 {meta['duration_s']*2:.1f} 秒。</p>
<p>最大抬升 {row['oracle_max_object_lift_m']*100:.2f} cm · 经双指接触验证的连续左持物 {row['longest_verified_left_grasp_s']:.2f} 秒。</p>
<details><summary>模型与判据</summary><p><code>{html.escape(checkpoint)}</code></p>
<p>抓取：物体抬升超过 5 cm，左夹爪开度 15%–85%，双指同时有正接触冲量，右臂不接触物体，持续 0.6 秒。
该判据只用于评价，不参与策略动作或控制。完整任务仍要求交接后稳定放置。</p>
<p>该场景属于开发/训练场景，不是独立盲测；单个展示案例不能代表泛化成功率。</p>
<a href="{name}/results.json">结果 JSON</a> · <a href="{name}/seed_{seed}_trace.npz">完整数值轨迹</a></details></article>'''
            cards.append(card)
            all_rows.append({'experiment': name, **row})
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JAKA Zu7 · 全景抓取验证</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#172638;font:16px/1.7 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:24px}
article{background:#fff;border:1px solid #d2dae5;border-radius:14px;margin:28px 0;padding:20px}h1,h2{line-height:1.3}h2{margin:10px 0}
video{display:block;width:100%;background:#111;border-radius:8px}p{color:#475569}button,select{font:inherit;padding:7px 12px;border:1px solid #bac7d8;border-radius:7px;background:#f7faff;cursor:pointer}
button:hover{background:#dceaff}.controls,.markers{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}.badge{background:#e5efff;color:#204b86;padding:5px 10px;border-radius:20px;font-size:14px}a{color:#1859a7}code{overflow-wrap:anywhere}details{margin-top:12px}
</style><main><h1>两台 JAKA Zu7 · 全景抓取验证</h1>
<p>大画面：全局斜视；右侧：俯视、头部近景、左腕视角。默认半速播放，可全屏和逐帧查看。</p>
<p><strong>抓取成功与完整任务成功分别报告。</strong>本页保留从初始姿态到终止的完整回合，包含后续失败。
保护触发后终止仿真；片尾是标注过的冻结画面，不是新的动作或自动恢复。</p>
<p>录像使用仿真时间；推理等待不计入画面时长。全局/俯视相机仅用于观看，策略仍使用原来的三路 RGB。</p>
'''+''.join(cards)+'''</main><script>
document.querySelectorAll('article').forEach(card=>{const v=card.querySelector('video'),s=card.querySelector('select');
v.defaultPlaybackRate=0.5;v.playbackRate=0.5;v.addEventListener('loadedmetadata',()=>v.playbackRate=Number(s.value));
s.onchange=()=>v.playbackRate=Number(s.value);
card.querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>{v.currentTime=Number(b.dataset.seek);v.pause()});
card.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>{v.pause();v.currentTime=Math.max(0,Math.min(v.duration,v.currentTime+Number(b.dataset.frame)/25))});
card.querySelector('[data-fullscreen]').onclick=()=>v.requestFullscreen();
v.addEventListener('timeupdate',()=>{card.querySelector('.clock').textContent=`视频 ${v.currentTime.toFixed(2)} / ${Number(v.duration||0).toFixed(2)} 秒`});
});</script></html>'''
    (args.output / 'index.html').write_text(page)
    (args.output / 'summary.json').write_text(json.dumps(all_rows, indent=2)+'\n')
    (args.output / 'video_validation.json').write_text(json.dumps(checks, indent=2)+'\n')
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob('*')):
            if path.is_file(): archive.write(path, Path('jaka-zu7-multiview') / path.relative_to(args.output))
    with zipfile.ZipFile(args.archive) as archive: assert archive.testzip() is None
    print(json.dumps({'page': str(args.output/'index.html'), 'archive': str(args.archive),
                      'size': args.archive.stat().st_size, 'sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
