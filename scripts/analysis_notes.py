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
print(json.dumps(report))
