"""Write final project handoff from audited results already staged for publishing."""
import argparse,json
from pathlib import Path
from datetime import datetime
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/data/jaka-vla'));a=p.parse_args();r=a.root
pub=r/'reviews/2026-11-16';s=json.loads((pub/'summary.json').read_text());delivery=json.loads((pub/'DELIVERY_PACKAGE.json').read_text());g=s['groups'];held=g['heldout_all']
assert s['finished'],'Final docs require ended batches'
rate=lambda v:f"{v['successes']}/{v['episodes']}"+(f"（{v['successes']/v['episodes']:.1%}）" if v['episodes'] else '')
ci=held['layout_cluster_bootstrap_95'];interval=f'{ci[0]:.1%} - {ci[1]:.1%}' if ci else '未估计'
blurb=f"本轮固定π0.5 three_targets/5744、K25/H50/noise42，共完成{s['completed']}/{s['planned']}回合；训练布局复现{rate(g['reproduction'])}，开发集{rate(g['development'])}，原定留出{rate(g['heldout'])}，追加留出{rate(g['heldout_extended'])}，全部独立留出{rate(held)}。完整独立留出布局{held['complete_layouts']}个，布局聚类bootstrap 95%区间{interval}。运行异常{len(s['operational_errors'])}，未完成{len(s['uncompleted'])}。"
scope='仅验证归档硬件仿真中的左臂抓取、抬升和0.6秒持物；同步推理暂停仿真，不代表长期持物、双臂交接、语言作用消融、抗推理延迟或真机结果。此次未训练、未调整控制器或成功判据。'
nextsteps='下一步保留当前模型与评测作为对照，针对空抓和低成功位置补充经专家及独立回放验收的接近/恢复示教，再做固定开发协议的匹配比较。此次留出2000-2099已经用于评估；后续若用这些结果改进方法，应另外预留最终独立布局。不要自动续训或重新运行本次已完成批次。用户已停止小时自动反馈，不恢复automation。'
links=f"[汇报版PDF](JAKA_pi05_性能评测报告.pdf) · [详细报告](REPORT.md) · [成功与失败Demo](demo_success_and_failure.mp4) · [逐回合CSV](results.csv) · [全部录像查看包]({delivery['archive']})"
(pub/'README.md').write_text('# 2026-11-16 当前 π0.5 模型性能评测\n\n'+blurb+'\n\n'+links+'\n\n下载查看包，完整解压后打开review/index.html，可按分组、颜色和成败筛选所有录像；无需服务器在线。Demo是明确标注的说明案例，不用于估计成功率。\n\n'+scope+'\n\n模型已经具备部分抓取行为，但当前成功率不足以视为可靠操作策略。首批和追加批次均在执行前固定场景清单，失败未丢弃，全部已完成回合独立复核物理证据。原始轨迹及模型输入输出已服务器/本地保存并校验，查看包不包含训练集和模型权重。\n\n包字节数：'+str(delivery['bytes'])+'；SHA256：`'+delivery['sha256']+'`。文件校验与播放检查见交付清单。\n\n'+nextsteps+'\n')
report=(pub/'REPORT.md').read_text()
report=report.replace('打开 [index.html](index.html) 可按分组、颜色、成功/失败筛选全部录像；',f'下载并解压[全部录像查看包]({delivery["archive"]})，打开review/index.html，可按分组、颜色、成功/失败筛选全部录像；')
(pub/'REPORT.md').write_text(report)
# Keep relative report links usable from docs.
for name in ['index.html','performance.png','JAKA_pi05_性能评测报告.pdf','demo_success_and_failure.mp4','results.csv','summary.json','protocol.json',delivery['archive']]:
 report=report.replace(']('+name+')','](../reviews/2026-11-16/'+name+')')
report=report.replace('[index.html](../reviews/2026-11-16/index.html)',f'[完整查看包](../reviews/2026-11-16/{delivery["archive"]})')
(r/'docs/BENCHMARK_20261116_RESULTS.md').write_text(report)
entry='## 2026-11-16 固定模型批量评测已结束\n\n'+blurb+'\n\n'+scope+'\n\n报告与录像：docs/BENCHMARK_20261116_RESULTS.md，reviews/2026-11-16/README.md。模型参数/统计哈希和全部物理证据已复核，完整视频解码与本地逐文件备份清单见该目录。旧开发1000的蓝块本次未复现；首次输入逐值相同、预测动作存在最大0.001895差异，原因尚未隔离。\n\n'+nextsteps+'\n\n'
for name in ['AGENT.md','MEMORY.md','skill.md']:
 path=r/name;text=path.read_text();pos=text.find('\n')+1;path.write_text(text[:pos]+'\n'+entry+text[pos:])
with (r/'docs/EXPERIMENT_LOG.md').open('a') as f:f.write('\n\n'+entry+'全部独立留出失败类型：'+json.dumps(held['failures'],ensure_ascii=False)+'。非预定接触回合'+str(held['episodes_with_forbidden_contact'])+'，非目标接触回合'+str(held['episodes_with_non_target_contact'])+'（与失败类型可能重叠）。此次结束时间：'+datetime.now().astimezone().isoformat()+'。\n')
path=r/'docs/STATUS.md';path.write_text('# 2026-11-16 固定模型评测最终状态\n\n'+blurb+'\n\n'+scope+'\n\n本轮评测已结束，后续运行状态应另查实际进程；下方为旧状态工具的历史阶段快照。报告见BENCHMARK_20261116_RESULTS.md。\n\n'+path.read_text())
readme=r/'README.md';text=readme.read_text();heading=text.split('\n',1)[0]
rest=text.split('## 项目记录',1)[1] if '## 项目记录' in text else text
current=heading+'\n\n## 最新结果：2026-11-16 固定模型性能评测\n\n'+blurb+'\n\n[评测资料与全部录像](reviews/2026-11-16/README.md) · [汇报版PDF](reviews/2026-11-16/JAKA_pi05_性能评测报告.pdf) · [详细报告](docs/BENCHMARK_20261116_RESULTS.md)\n\n'+scope+'\n\n## 当前技术路线\n\n- 场景：RoboTwin / SAPIEN，按实验室归档建模的两台JAKA Zu7；左右模型、基座安装、DO夹爪映射有归档依据。三指机构行程、连续开度及三RGB/腕相机仍为待实机核验的仿真假设。\n- 模型：OpenPI π0.5，LoRA加投影微调；97条验收示教/41206帧/33训练布局已保存。当前检查点在已有权重上做三目标课程训练，不等于充分利用全数据后的最优模型。\n- 输入：三路RGB、14维实测关节/夹爪状态和语言；输出H50×14维目标动作，K25执行，25Hz控制、250Hz物理仿真。\n- 当前阶段：固定模型能力评估已扩大到多个布局；先提升基础抓取稳定性，再推进完整双臂交接、事件重规划和真机迁移。\n- 专家成功、学习策略成功、研究方法效果和真机结果分别报告。历史SmolVLA试验保留在对应日期文档。\n\n## 项目记录'+rest
current=current.replace('服务器RTX4090 24GB，约60GB系统内存，150GB数据盘。运行前`source env.sh`。','本轮服务器识别为RTX 4080 SUPER，报告显存32760MiB，约58GiB内存、148GiB数据盘。运行前`source env.sh`。')
current=current.replace('仿真使用`envs/robotwin/bin/python`；转换、训练和模型服务使用`envs/vla/bin/python`。','仿真使用`envs/robotwin/bin/python`；π0.5使用独立`/data/pi05-env/bin/python`及`scripts/pi05_env.sh`。历史SmolVLA使用`envs/vla/bin/python`。')
current=current.replace('机器人主配置`configs/robots/dual_zu7.json`；生成/注册入口`scripts/setup_zu7.py`。','当前归档硬件配置`configs/robots/dual_zu7_lab.json`，归档证据`configs/robots/lab_archive_20261111.json`；生成/注册入口`scripts/setup_lab_zu7.py`。旧配置保留用于历史实验。')
readme.write_text(current)
for name in ['docs/HOME_ACCESS.md','docs/VIEW_RESULTS.md']:
 path=r/name;text=path.read_text();pos=text.find('\n')+1
 intro='\n\n## 最新：2026-11-16 π0.5 性能评测\n\n'+blurb+'\n\n打开[本轮资料目录](../reviews/2026-11-16/README.md)，可直接看PDF与Demo，或下载全部录像查看包。GitHub文件页面点Download raw file；完整解压后打开review/index.html。服务器关机后仍可查看。以下旧日期资料均为历史结果。\n\n'
 path.write_text(text[:pos]+intro+text[pos:])
print(blurb)
