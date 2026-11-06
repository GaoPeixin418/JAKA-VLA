"""Read-only path-length diagnostics from existing Zu7 control traces."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/data/jaka-vla')


def main():
    output = ROOT/'results/zu7_path_quality_audit_20261106_v2.json'
    if output.exists(): raise FileExistsError(output)
    rows = []
    for folder, seed in [('zu7_multiview_expert_seed6',6),('zu7_grasp_review6000_seed0_1',0),
                         ('zu7_grasp_review6000_seed0_1',1),('zu7_grasp_review6000_seed2',2),
                         ('zu7_grasp_review9000_seed6',6)]:
        source = ROOT/'results'/folder/f'seed_{seed}_trace.npz'
        data = np.load(source); control = data['control_trace']
        # [step, measured14, velocity14, raw14, clipped14, object3, leftTCP7, rightTCP7]
        assert control.shape[1] == 74
        closed = np.flatnonzero(control[:,35] < .95)
        end = int(closed[0]) if len(closed) else len(control)-1
        tcp = control[:end+1,60:63]
        path = float(np.linalg.norm(np.diff(tcp,axis=0),axis=1).sum())
        direct = float(np.linalg.norm(tcp[-1]-tcp[0]))
        rows.append({'folder':folder,'seed':seed,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'approach_end_step':end,'approach_s':end/25,'tcp_length_m':path,
                     'direct_distance_m':direct,'detour_ratio':path/direct if direct>1e-8 else None,
                     'start_left_joint_deg':np.rad2deg(control[0,1:7]).tolist(),
                     'end_left_joint_deg':np.rad2deg(control[end,1:7]).tolist()})
    report = {'scope':'Existing traces only; approach ends at first requested left aperture below .95.',
              'limitations':['Straight-line distance is not a feasible collision-free path.',
                             'Ratios are descriptive; do not compare failed endpoints to successful grasps as path-quality scores.',
                             'No alternative IK branch or counterfactual collision-free path has been tested.'],
              'rows':rows}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output)


if __name__ == '__main__': main()
