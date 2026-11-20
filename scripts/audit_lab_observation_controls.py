"""Compare task instructions only across exactly matched initial RGB/state observations."""
import argparse,hashlib,json,itertools
from collections import defaultdict
from pathlib import Path
import cv2
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    accepted=json.loads((a.raw/'acceptance.json').read_text());groups=defaultdict(list)
    names=['cam_head','cam_left_wrist','cam_right_wrist']
    for entry in accepted:
        scene=entry['scene_spec'];path=a.raw/f'data/episode_{entry["episode"]:07d}.hdf5'
        with h5py.File(path) as f:
            state=np.concatenate([f[f'state/{arm}_{part}'][0] for arm in ['left','right'] for part in ['arm_joint_states','ee_joint_states']]).astype(np.float32)
            frames=[cv2.imdecode(np.frombuffer(f[f'vision/{name}/colors'][0],np.uint8),cv2.IMREAD_COLOR) for name in names]
            if any(x is None for x in frames):raise ValueError('Initial image failed to decode')
        digest=hashlib.sha256(state.tobytes()+b''.join(x.tobytes() for x in frames)).hexdigest()
        groups[(scene['layout_id'],digest)].append({'episode':entry['episode'],'target':scene['target_index'],
            'instruction':scene['instruction'],'state':state,'frames':frames})
    comparisons=[]
    for (layout,_),rows in sorted(groups.items()):
        for left,right in itertools.combinations(rows,2):
            if left['target']==right['target']: continue
            comparisons.append({'layout':layout,'episodes':[left['episode'],right['episode']],
                'targets':[left['target'],right['target']],
                'different_instruction':left['instruction']!=right['instruction'],
                'state_max_absolute_difference':float(np.max(np.abs(left['state']-right['state']))),
                'camera_comparisons':{name:{'equal_pixels':bool(np.array_equal(x,y)),
                    'mean_absolute_pixel_difference':float(np.abs(x.astype(float)-y.astype(float)).mean()),
                    'reference_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                    'other_sha256':hashlib.sha256(y.tobytes()).hexdigest()} for name,x,y in zip(names,left['frames'],right['frames'])}})
    controlled=bool(comparisons) and all(x['different_instruction'] and x['state_max_absolute_difference']==0 and all(c['equal_pixels'] for c in x['camera_comparisons'].values()) for x in comparisons)
    report={'raw_dataset':str(a.raw),'episodes':len(accepted),'layouts':len({x['scene_spec']['layout_id'] for x in accepted}),
        'matched_initial_observation_groups':sum(len(rows)>1 for rows in groups.values()),
        'matched_cross_target_comparisons':len(comparisons),
        'initial_observations_identical_across_matched_target_choices':controlled,'comparisons':comparisons,
        'scope':'Exact initial observations are compared only when state and all RGB images match; this allows intentional per-episode robot-start perturbations. Does not establish learned instruction following.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'}))
    if not controlled:raise ValueError('No exact matched cross-target observation pair or mismatched instruction controls; inspect audit')

if __name__=='__main__':main()
