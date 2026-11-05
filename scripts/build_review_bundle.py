"""Build an offline, dependency-free viewer for existing pilot results."""
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/review_bundle'


def load(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    (DEST / 'videos').mkdir(parents=True, exist_ok=True)
    (DEST / 'evidence').mkdir(exist_ok=True)
    generated = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    status = load('results/project_status.json')
    resources = load('checkpoints/smolvla_pilot_seed42/resource_profile.json')
    validation = load('results/handover_pilot_validation.json')
    summaries = load('results/pilot_baselines.json')
    rows = []
    sections = []
    files = []

    def copy(relative, target):
        source = ROOT / relative
        destination = DEST / target
        shutil.copy2(source, destination)
        files.append(destination)
        return target

    def video_card(folder, ep, label, prefix):
        name = f"{prefix}_seed_{ep['seed']}"
        video = copy(f"results/{folder}/seed_{ep['seed']}.mp4", f'videos/{name}.mp4')
        success = '成功' if ep['success'] else '未成功'
        css = 'success' if ep['success'] else 'failure'
        return f'''<article class="card"><h3>{html.escape(label)} · seed {ep['seed']} <span class="{css}">{success}</span></h3>
<video controls playsinline preload="metadata" src="{video}">浏览器无法播放时，请下载MP4用本地播放器打开。</video>
<p>{ep['steps']} 控制步 · {ep['sim_seconds']:.2f} 秒仿真时间 · {ep['wall_seconds']:.2f} 秒墙钟时间</p>
<a href="{video}" download>下载 MP4</a></article>'''

    expert = load('results/expert_replay/results.json')
    copy('results/expert_replay/results.json', 'evidence/expert_replay.json')
    expert_cards = ''.join(video_card('expert_replay', ep, '专家回放', 'expert') for ep in expert)
    sections.append(f'<section><h2>专家参考动作</h2><p>已知示教动作通过相同控制器回放，用于观察完整交接流程；不计入模型成功率。</p><div class="grid expert">{expert_cards}</div></section>')

    for summary in summaries:
        k = summary['k']
        episodes = load(f'results/pilot_k{k}/results.json')
        assert len(episodes) == summary['episodes']
        copy(f'results/pilot_k{k}/results.json', f'evidence/pilot_k{k}.json')
        rows.append(f"<tr><td>K={k}</td><td>{k/25:.2f} 秒</td><td>{summary['successes']} / {summary['episodes']}</td><td><a href=\"evidence/pilot_k{k}.json\">逐回合 JSON</a></td></tr>")
        cards = ''.join(video_card(f'pilot_k{k}', ep, f'模型 K={k}', f'k{k}') for ep in episodes)
        sections.append(f'<section><h2>K={k} · 每执行 {k} 步重新观察</h2><div class="grid">{cards}</div></section>')
    for source, target in [
        ('results/pilot_baselines.json', 'evidence/pilot_baselines.json'),
        ('results/project_status.json', 'evidence/project_status.json'),
        ('results/handover_pilot_validation.json', 'evidence/dataset_validation.json'),
        ('checkpoints/smolvla_pilot_seed42/resource_profile.json', 'evidence/training_resources.json'),
    ]:
        copy(source, target)
    total = sum(s['episodes'] for s in summaries)
    successes = sum(s['successes'] for s in summaries)
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双臂 VLA · 首轮实验结果</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#172033;font:16px/1.65 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}main{max-width:1280px;margin:auto;padding:40px 24px}h1{font-size:32px;margin:0 0 8px}h2{font-size:22px;margin:0 0 8px}h3{font-size:16px;margin:0 0 12px}p{margin:8px 0;color:#4a576e}.meta{font-size:13px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}.stat,.card,section{background:white;border:1px solid #e0e5ee;border-radius:12px}.stat{padding:18px}.stat strong{display:block;font-size:26px}.stat span{font-size:14px;color:#526079}section{padding:24px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.expert{max-width:780px;grid-template-columns:1fr}.card{padding:16px;min-width:0}video{width:100%;aspect-ratio:4/3;background:#111827;border-radius:6px}.card p{font-size:12px}.success,.failure{font-size:12px;padding:3px 8px;border-radius:12px;white-space:nowrap}.success{background:#e2f5e9;color:#17623b}.failure{background:#fff0e6;color:#93420d}table{border-collapse:collapse;width:100%;text-align:left}th,td{padding:12px;border-bottom:1px solid #e6eaf0}a{color:#235ac1;text-decoration:none}a:hover{text-decoration:underline}.note{border-left:4px solid #849ac2;padding:10px 16px;background:#edf2fa}@media(max-width:850px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}main{padding:24px 14px}section{padding:18px}}
</style></head><body><main>'''
    page += f'''<h1>双臂 VLA · 首轮实验结果</h1><p>方块交接 / RoboTwin · ALOHA-AgileX / SmolVLA</p>
<p class="meta">生成时间：{generated} · 进度采样：{html.escape(status['observed_at'])}</p>
<div class="stats"><div class="stat"><strong>{successes}/{total}</strong><span>模型测试成功回合</span></div><div class="stat"><strong>{len(validation)} 条</strong><span>专家训练示教 · {sum(e['frames'] for e in validation):,} 帧</span></div>
<div class="stat"><strong>{status.get('training_steps_observed', '未知')} 步</strong><span>训练已完成 · {resources['wall_seconds_including_load_and_save']/60:.2f} 分钟</span></div>
<div class="stat"><strong>{resources['peak_allocated_gib']:.2f} GiB</strong><span>训练峰值分配显存</span></div></div>
<p class="note">这是基础策略的首轮小规模实验。每个 K 仅 3 个测试种子；当前模型尚未成功完成任务。仿真在模型推理期间暂停，录像按 25 FPS 的仿真时间播放，不包含推理等待。事件监测器和节卡真机迁移尚未实现。</p>
<section><h2>固定执行长度对照</h2><p>K 表示每次预测后执行多少个控制步；各组使用同一检查点、相同测试种子和控制器。</p><table><thead><tr><th>执行长度</th><th>对应仿真时长</th><th>成功 / 测试</th><th>原始记录</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(sections)}
<section><h2>查看方式与证据</h2><p>点击各视频播放；可先看专家参考动作，再按相同 seed 比较不同 K。页面可离线打开，所有录像与数据均包含在当前文件夹中。</p>
<p><a href="evidence/pilot_baselines.json">评测汇总</a> · <a href="evidence/training_resources.json">训练资源</a> · <a href="evidence/dataset_validation.json">数据校验</a> · <a href="evidence/project_status.json">进度快照</a></p></section></main></body></html>'''
    index = DEST / 'index.html'
    index.write_text(page, encoding='utf-8')
    files.append(index)
    output = ROOT / 'results/jaka-vla-review.zip'
    with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(Path('jaka-vla-review') / path.relative_to(DEST)))
    with ZipFile(output) as archive:
        assert archive.testzip() is None
    print(json.dumps({'viewer': str(index), 'zip': str(output), 'bytes': output.stat().st_size,
                      'videos': len(list((DEST/'videos').glob('*.mp4'))),
                      'successes': successes, 'episodes': total}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
