"""Offline diagnostics for all saved lab-policy rollouts; no extra inference."""
import argparse,json
from itertools import combinations
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    output=a.run/'offline_diagnostics.json'
    if output.exists():raise FileExistsError(output)
    summary=json.loads((a.run/'summary.json').read_text());rows=[];initial_chunks={}
    for entry in summary['rollouts']:
        result_path=Path(entry['results']);result=json.loads(result_path.read_text())[0];folder=result_path.parent;seed=entry['seed']
        with np.load(folder/f'seed_{seed}_trace.npz') as data:
            control=data['control_trace'];physics=data['physics_trace'];evidence=data['grasp_evidence']
            chunks=data['raw_chunks']
            initial_chunks[(entry['split'],entry['target'])]=(control[0,1:15],chunks[0])
        closed=np.flatnonzero(control[:,35]<.5)
        first_close=None
        if len(closed):
            i=int(closed[0]);first_close={'control_step':int(control[i,0]),'target_xy_distance_m':float(np.linalg.norm(control[i,60:62]-control[i,57:59])),
                'target_horizontal_displacement_m':float(np.linalg.norm(control[i,57:59]-control[0,57:59])),
                'measured_aperture':float(control[i,7]),'command_aperture':float(control[i,35])}
        consecutive=longest=0
        for valid in evidence[:,5]:
            consecutive=consecutive+1 if valid else 0;longest=max(longest,consecutive)
        verified=longest*.004
        if not np.isclose(verified,result['longest_verified_left_grasp_s'],atol=1e-8):raise ValueError('Saved grasp duration disagrees with physics evidence')
        if bool(verified>=.6)!=result['grasp_success']:raise ValueError('Saved grasp verdict disagrees with physics evidence')
        contacts=json.loads((folder/f'seed_{seed}_contacts.json').read_text())
        target_contacts=[x for x in contacts if 'scene:box' in x['pair']]
        distractor_contacts=[x for x in contacts if any(s.startswith('scene:distractor_') for s in x['pair'])]
        joints=[*range(6),*range(7,13)]
        rows.append({**entry,'steps':result['steps'],'sim_seconds':result['sim_seconds'],
            'recomputed_verified_hold_s':verified,'first_command_below_half_aperture':first_close,
            'maximum_target_horizontal_displacement_m':float(np.linalg.norm(control[:,57:59]-control[0,57:59],axis=1).max()),
            'target_contacts':target_contacts,'distractor_contacts':distractor_contacts,
            'max_right_joint_displacement_rad':float(np.abs(control[:,8:14]-control[0,8:14]).max()),
            'measured_joint_speed_peak_rad_s':float(np.abs(physics[:,43:57][:,joints]).max()),
            'inference_median_seconds':float(np.median(result['inference_seconds'])),
            'normalized_state_abs_max':max((x['normalized_state_abs_max'] for x in result['normalization_checks']),default=None)})
    pairs=[]
    for split in ['seen','development']:
        for left,right in combinations([key for key in initial_chunks if key[0]==split],2):
            s1,c1=initial_chunks[left];s2,c2=initial_chunks[right]
            pairs.append({'split':split,'targets':[left[1],right[1]],'same_initial_state':bool(np.array_equal(s1,s2)),
                'first_k8_left_joint_prediction_rms_difference_rad':float(np.sqrt(np.mean((c1[:8,:6]-c2[:8,:6])**2))),
                'first_k8_left_aperture_prediction_rms_difference':float(np.sqrt(np.mean((c1[:8,6]-c2[:8,6])**2)))})
    report={'run':str(a.run),'rollouts':rows,'first_prediction_instruction_pairs':pairs,
        'scope':'Offline descriptive diagnostics, all attempted rollouts retained. Prompt-dependent predictions alone do not establish correct instruction following. Object truth is evaluation only.'}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(output),'rollouts':len(rows),'grasp_evidence_recomputed':True}))

if __name__=='__main__':main()
