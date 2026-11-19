"""Record actual training/data states without counting plans as completed work."""
from pathlib import Path
from datetime import datetime
import collections,json
R=Path('/data/jaka-vla');B=R/'results/expert_multilayout_20261118';A=B/'acceptance_1913';review=R/'reviews/2026-11-18';review.mkdir(exist_ok=True)
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
train=read(R/'results/lab_pi05_multilayout_20261118/audit.json')
collected=[read(p) for p in (B/'episodes').glob('*/result.json')]
checks=[read(p) for p in A.glob('layout*.json')];accepted=[r for r in checks if r['accepted']]
recovery=read(R/'results/lab_recovery_20261118/progress.json',{'completed':False,'attempts':[]})
conversion=read(R/'results/lab_expert_expansion_20261118_conversion/progress.json',{})
evalroot=R/'results/lab_pi05_benchmark_base_20261118';evaluated=[r for p in evalroot.glob('worker_*.json') for r in read(p)['rollouts']]
interrupted=[read(p).get('active_case') for p in evalroot.glob('worker_*.json') if read(p).get('active_case')]
interrupt={'reason':'User reprioritized expert data collection on 2026-11-18; stopped mid-batch, not a completed 90-case benchmark','completed_cases':len(evaluated),'qualified_successes':sum(r.get('qualified_grasp_success') is True for r in evaluated),'interrupted_cases':interrupted,'future_model_evaluation_started':False}
(evalroot/'interruption.json').write_text(json.dumps(interrupt,indent=2)+'\n')
summary={'at':datetime.now().astimezone().isoformat(),'training':{k:train.get(k) for k in ['completed','completed_steps','checkpoint','changed_frozen_parameters']},'collection':{'planned':30,'saved':len(collected),'layouts':len({d['scene_spec']['layout_id'] for d in collected}),'missing':['layout_4003_target_2'],'same_layout_trajectory_diversity':'one trajectory per target; repeated varied paths still missing'},'acceptance':{'checked':len(checks),'accepted':len(accepted),'frames':sum(r.get('frames',0) for r in accepted),'pending':len(collected)-len(checks),'rejected':[r for r in checks if not r['accepted']]},'recovery':recovery,'conversion':conversion,'partial_old_model_evaluation':interrupt,'new_data_used_in_training':False}
(review/'status.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
for name,src in [('training_audit.json',R/'results/lab_pi05_multilayout_20261118/audit.json'),('expert_acceptance.json',A/'progress.json'),('recovery_candidates.json',R/'results/lab_recovery_20261118/candidates.json')]:
 if src.exists():(review/name).write_bytes(src.read_bytes())
raccepted=sum(r.get('accepted',False) for r in recovery['attempts'])
text=f'''# 2026-11-18 多布局训练与专家数据扩充

更新时间：{summary['at']}。用户本日预计19:30关机，优先数据扩充；每30分钟状态反馈已重新获授权。

- 旧97条、33布局、41206帧数据上，从three_targets/5744继续微调3500步；保存multilayout_20261118/3500，冻结参数未变化。没有使用今天新增轨迹；新模型尚未评测，效果提升未知。
- 新布局4000–4009计划30条，实际保存{len(collected)}条，10个布局；4003目标2缺失。采集成功与入库验收分别计数。
- 三RGB逐帧解码、14维有限状态动作、25Hz观测及下一动作时间戳检查、相同场景/初态的有界控制器独立回放：已检查{len(checks)}条，通过{len(accepted)}条、{summary['acceptance']['frames']}帧；未检查{summary['acceptance']['pending']}条。原始轨迹和回放均禁止非预定接触。
- 恢复示教尝试{len(recovery['attempts'])}条，通过{raccepted}条；以progress.json逐条错误为准。源为11月16日策略失败的接近中途状态；仅重置左臂配置后专家重规划，没有执行策略前缀。布局2020以后作为训练/开发，不能再作为未来独立盲测。
- 格式转换已完成阶段：{conversion.get('stages',[])}，整体完成={conversion.get('completed',False)}。本日扩充数据尚未进入π0.5训练。
- 旧模型新布局基线按用户要求中止，已完成{len(evaluated)}回合、成功{interrupt['qualified_successes']}回合；原计划90回合，不能报告完成90次。未启动新模型90次评测。

## 更正与边界

新数据4000段是用户要求的专家训练扩充，生成时split误写heldout；源证据保留原样，在验收后的派生训练身份中明确改为train并记录出处。不能把这些场景继续称为盲测。

本批仅增加原场景生成器内的位置、尺寸、朝向及颜色排列组合；没有增加新物体类别、动态障碍或双臂协作任务。每布局每目标仅一条轨迹，尚未解决同布局路径多样性不足。历史旧恢复轨迹是否缺图此前只做了浅目录检查，不能据此作确定结论；今天恢复采集以实际HDF5、图像解码与回放为准。

最初多布局训练因只source env.sh而缺openpi_client导入路径，改为scripts/pi05_env.sh后完成。采集首次180秒超时与重启的历史失败保留可用残余产物；耗时根因没有完成隔离，不能把“图像写盘慢”当成已证实结论。

29条采集结束后曾处于空闲等待，并没有自动接上回放；本次已补齐验收与限时转换串联。任何尚未验收或转换完成的数据不计入可训练数量。

## 下一步

1. 以验收清单完成转换和版本化合并；保留原97条与归一化统计，检查跨数据集续训时的身份、归一化及动作变换兼容性。
2. 先补4003目标2，并在多个布局中增加多条路径、接近中途偏差和重对齐；失败案例与首次规划失败原因保留。已完成数据不重复采集。
3. 新增数据入训后先做少量固定开发案例验收，再对新布局做配对评测；暂停旧模型大规模评测。可靠抓取后再推进交接、放置、复位。

## 证据

- results/expert_multilayout_20261118/episodes：采集证据；acceptance_1913：数据检查和逐条回放。
- results/lab_recovery_20261118：候选来源、采集/回放成功与失败。
- results/lab_expert_expansion_20261118_conversion/progress.json：转换的实际完成阶段。
- results/lab_pi05_multilayout_20261118/audit.json：3500步真实训练审计。
- reviews/2026-11-18/status.json：机器可读摘要。备份完成情况另见本地校验清单，Git代码同步不等于权重/数据备份。
'''
(R/'docs/ITERATION_20261118.md').write_text(text)
(review/'README.md').write_text('# 2026-11-18 数据扩充\n\n详见 [本轮技术记录](../../docs/ITERATION_20261118.md) 和 [实际状态](status.json)。专家成功不能代替学习策略成功；新增数据尚未入训。\n')
for name in ['MEMORY.md','AGENT.md','docs/STATUS.md']:
 p=R/name;old=p.read_text();heading,rest=old.split('\n',1)
 marker='## 2026-11-18 当前状态（覆盖旧日期授权与时限）'
 if marker in rest:
  start=rest.index(marker);end=rest.find('\n## ',start+len(marker));rest=rest[:start]+(rest[end:] if end>=0 else '')
 entry=f'\n{marker}\n\n以docs/ITERATION_20261118.md和reviews/2026-11-18/status.json为最新证据。3500步训练已保存；新模型未评测。专家新增{len(collected)}条，完整验收{len(accepted)}条，恢复验收{raccepted}条；新增数据未入训。用户要求优先扩充数据，旧模型评测已中止，不恢复；每30分钟汇报已授权，本日19:30预计关机。先查实际进程和转换状态，不重复采集已完成数据。\n'
 p.write_text(heading+'\n'+entry+rest)
with (R/'docs/EXPERIMENT_LOG.md').open('a') as f:f.write('\n\n'+text)
with (R/'docs/TECHNICAL_DESIGN.md').open('a') as f:f.write('\n\n## 2026-11-18 数据验收与划分\n\n新增专家扩充使用4000–4009训练布局；原采集split误标heldout只在派生训练身份中显式纠正，保留源证据。恢复候选取历史2020布局的学习失败中途状态，2020不再供未来盲测。入训须逐帧RGB/时间戳检查、同场景同初态有界控制器回放、无非预定接触。当前仅扩大原生成器内布局，并未新增动态干扰/协作任务。\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['recovery']},ensure_ascii=False))
