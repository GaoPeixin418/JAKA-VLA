"""Build a separate viewer for Zu7 evidence; never overwrite the ALOHA pilot."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,html,json,shutil,zipfile
ROOT=Path('/data/jaka-vla');DEST=ROOT/'results/zu7_review'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=DEST)
    p.add_argument('--archive',type=Path,default=ROOT/'results/jaka-zu7-review.zip')
    args=p.parse_args();dest=args.output
    dest.mkdir(parents=True,exist_ok=True);cards=[];summary=[]
    experiments=[('zu7_expert_replay_v2','过低限速的失败回放','专家回放，1.5 rad/s；保留用于解释时序错误。'),
                 ('zu7_expert_replay_v3','中间版本：尚有双臂碰撞','虽然完成放置，接近交接时仍有碰撞，保留为诊断证据。'),
                 ('zu7_expert_collision_replay','加入双臂避碰后的专家回放','新控制器回放；任务成功与接触情况分别报告，不计入学习策略成功率。'),
                 ('zu7_learned_seen3000','学习策略：训练场景','只检验小样本拟合，不代表泛化。'),
                 ('zu7_learned_seen6000','继续微调后：训练场景','累计6000步，第二阶段更换示教、重新初始化优化器。'),
                 ('zu7_learned_seen6000_zero','零噪声采样诊断：训练场景','推理采样方式消融；不与默认随机采样混合统计。'),
                 ('zu7_learned_unseen6000','继续微调后：未见场景','小样本排错，不能据此声称统计显著性。'),
                 ('zu7_learned_unseen3000','学习策略：未见场景','小样本排错，不能据此声称统计显著性。'),
                 ('zu7_learned_unseen6000_seed3_5','6000 步策略：K8，种子 3–5','本轮开发对照；同步推理时仿真暂停。'),
                 ('zu7_learned_unseen6000_seed3_5_k4','6000 步策略：K4，种子 3–5','与 K8 使用同一权重、场景和噪声。'),
                 ('zu7_learned_unseen6000_seed3_5_k1','6000 步策略：K1，种子 3–5','每步重新观察；不是实时延迟实验。'),
                 ('zu7_independent_expert_replay','新增六条示教的专家回放','按 seed.txt 匹配场景；专家成功不计入学习策略成功率。'),
                 ('zu7_learned_independent9000_seed3_5','扩充示教后：K8，种子 3–5','新增六条专家示教、额外 3000 步微调；这些种子已用于开发诊断，不是最终盲测集。')]
    for folder,title,note in experiments:
        source=ROOT/'results'/folder;report=source/'results.json'
        reports=[report] if report.exists() else sorted(source.glob('seed_*/results.json'))
        if not reports:continue
        entries=[(row,path.parent) for path in reports for row in json.loads(path.read_text())]
        rows=[row for row,_ in entries];target=dest/folder;target.mkdir(exist_ok=True)
        (target/'results.json').write_text(json.dumps(rows,indent=2)+'\n');items=[]
        for row,media_source in entries:
            name=f"seed_{row['seed']}.mp4"
            if (media_source/name).exists():shutil.copy2(media_source/name,target/name)
            verdict='通过' if row['success'] else '未通过'
            reason=row.get('failure_reason')
            failure=(' · '+html.escape(reason['type'])) if reason else ''
            contact_path=media_source/f"seed_{row['seed']}_contacts.json";contact_note=''
            if contact_path.exists():
                contact_rows=json.loads(contact_path.read_text())
                interarm=[v for v in contact_rows if any(s.startswith('left:') for s in v['pair']) and any(s.startswith('right:') for s in v['pair'])]
                contact_note=f" · 双臂接触对 {len(interarm)}"
                shutil.copy2(contact_path,target/contact_path.name)
            items.append(f'<article><h3>Seed {row["seed"]} · {verdict}{failure}</h3><video controls preload="metadata" src="{folder}/{name}"></video><p>{row["steps"]} 步 · 仿真 {row["sim_seconds"]:.2f} 秒{contact_note}</p></article>')
        metric=source/'control_metrics.json'
        if metric.exists():shutil.copy2(metric,target/metric.name)
        image=source/'seed_0_control.png'
        plot=''
        if image.exists():
            shutil.copy2(image,target/image.name);plot=f'<details><summary>展开动作与实际运动曲线</summary><img src="{folder}/{image.name}"></details>'
        n=sum(bool(r['success']) for r in rows);summary.append({'experiment':folder,'successes':n,'episodes':len(rows)})
        cards.append(f'<section><h2>{title} · {n}/{len(rows)}</h2><p>{note}</p><div class="grid">{"".join(items)}</div>{plot}<p><a href="{folder}/results.json">完整结果 JSON</a></p></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zu7 双臂修正验收</title><style>
body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f3f5f8;color:#17202e}main{max-width:1200px;margin:auto;padding:28px}section{background:white;border-radius:12px;margin:24px 0;padding:20px;border:1px solid #dce2ea}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}video,img{width:100%;border-radius:8px}h3{font-size:16px}p{color:#4b586e}a{color:#1958b2}details{margin-top:18px}@media(max-width:750px){.grid{grid-template-columns:1fr}}
</style><main><h1>两台 JAKA Zu7 · 修正验收</h1><p>官方机器人模型 + 暂定平行夹爪；左抓取 → 双臂交接 → 右放置到 8 cm 接收台。</p><p>专家与学习策略分别统计。录像按仿真时间播放；模型推理等待不包含在录像中。尚未连接真机。</p>'''
    page+=f'<p>生成时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}</p>'+''.join(cards)+'</main></html>'
    (dest/'index.html').write_text(page);(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    archive=args.archive
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in dest.rglob('*'):
            if file.is_file():z.write(file,Path('jaka-zu7-review')/file.relative_to(dest))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(dest/'index.html',archive)


if __name__=='__main__':main()
