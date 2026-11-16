from pathlib import Path
import json
import numpy as np
import argparse
p=argparse.ArgumentParser();p.add_argument('--review',type=Path,default=Path('/home/lightwheel/Documents/Codex/2026-11-11/du-q/outputs/20261116/review'));root=p.parse_args().review
s=json.loads((root/'summary.json').read_text());rows=[r for r in s['rows'] if r['split'].startswith('heldout')]
metrics={}
for label,subset in [('qualified_success',[r for r in rows if r['qualified_success']]),('failure',[r for r in rows if not r['qualified_success']])]:
 fields={}
 for key in ['first_close_horizontal_error_cm','target_movement_before_first_close_cm','max_target_horizontal_movement_cm','max_lift_cm','hold_seconds','max_measured_joint_speed_rad_s']:
  x=[r[key] for r in subset if r[key] is not None]
  fields[key]={'n':len(x),'median':float(np.median(x)) if x else None,'p25':float(np.quantile(x,.25)) if x else None,'p75':float(np.quantile(x,.75)) if x else None,'max':float(max(x)) if x else None}
 metrics[label]={'episodes':len(subset),'metrics':fields}
report={'scope':'Descriptive correlations only; successful/failed groups differ in scene and trajectory. No causal root-cause claim.', 'groups':metrics}
(root/'failure_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
good=metrics['qualified_success']['metrics']['first_close_horizontal_error_cm']
bad=metrics['failure']['metrics']['first_close_horizontal_error_cm']
if good['n'] and bad['n']:
 text=(root/'REPORT.md').read_text()
 section=('## 闭爪阶段的描述性证据\n\n'
          f"在发出过闭爪指令的独立留出回合中，首次闭爪指令时工具中心到目标中心的水平距离中位数：成功组 {good['median']:.2f} cm（{good['n']}回合），失败组 {bad['median']:.2f} cm（{bad['n']}回合）。未发出闭爪指令的回合不计入该距离统计。\n\n"
          '这提示应优先检查末段接近、物体位移后的恢复以及闭爪时机，但属于不同场景与轨迹之间的相关性；目标可能已被推动，工具中心距离也不等于完整夹持几何误差，不能据此确定唯一根因。分位数与目标移动证据见failure_diagnostics.json。\n\n')
 text=text.replace('## 历史结果复现与解释边界',section+'## 历史结果复现与解释边界')
 (root/'REPORT.md').write_text(text)
print(json.dumps(report))
