"""Assemble completed accepted batches and report target/layout coverage."""
import argparse,json,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
import h5py
import numpy as np
ROOT=Path('/data/jaka-vla')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[];keys=set();objects={};layout_targets=defaultdict(set);split_overrides={}
    for batch in a.batches:
        progress=json.loads((batch/'progress.json').read_text())
        if not progress['completed']:raise ValueError(f'Batch not finished: {batch}')
        declared=set(progress.get('split_assignment',{}).get('train',[]))
        for row in progress['attempts']:
            if not row['accepted']:continue
            spec=json.loads((Path(row['probe'])/'result.json').read_text())['scene_spec']
            if spec['split']!='train':
                if row['layout'] not in declared:raise ValueError('Nontraining layout cannot enter the training dataset')
                split_overrides[row['layout']]={'from':spec['split'],'to':'train','batch':str(batch),'reason':progress.get('split_assignment',{}).get('note','declared training layout')}
            key=(spec['version'],row['layout'],row['target'])
            if key in keys:raise ValueError(f'Duplicate episode condition: {key}')
            keys.add(key);layout_targets[row['layout']].add(row['target'])
            geometry=json.dumps(spec['objects'],sort_keys=True)
            if row['layout'] in objects and objects[row['layout']]!=geometry:raise ValueError('Target choice changed scene geometry')
            objects[row['layout']]=geometry;rows.append(row)
    if not rows:raise ValueError('No accepted episodes')
    subprocess.run([sys.executable,str(ROOT/'scripts/assemble_grasp_curriculum.py'),'--probes',*[r['probe'] for r in rows],
                    '--replays',*[r['replay'] for r in rows],'--output',str(a.output)],check=True,stdout=subprocess.DEVNULL)
    if split_overrides:
        identity_path=a.output/'robot_identity.json';identity=json.loads(identity_path.read_text())
        for spec in identity.get('scenes_by_episode',{}).values():
            if spec.get('layout_id') in split_overrides:spec['split']='train'
        identity['split_assignment_provenance']={'corrections':split_overrides,'source_evidence_unchanged':True}
        identity_path.write_text(json.dumps(identity,indent=2)+'\n')
        (a.output/'split_assignment.json').write_text(json.dumps(identity['split_assignment_provenance'],indent=2)+'\n')
    frame_counts=[];holds=[];positions=[]
    for row in rows:
        probe=Path(row['probe']);report=json.loads((probe/'result.json').read_text())
        with h5py.File(probe/'raw/data/episode_0000000.hdf5') as f:frame_counts.append(len(f['action/left_arm_joint_states']))
        holds.append(report['longest_verified_hold_s']);positions.append(report['scene_spec']['objects'][row['target']]['xyz'])
    report={'episodes':len(rows),'frames':sum(frame_counts),'fps':25,'duration_seconds':sum(frame_counts)/25,
        'source_batches':[str(x) for x in a.batches],'color_counts':dict(Counter(r['color'] for r in rows)),
        'color_position_counts':{color:{str(target):sum(r['color']==color and r['target']==target for r in rows)
            for target in range(3)} for color in ('red','blue','yellow')},
        'target_position_counts':dict(Counter(r['target'] for r in rows)),
        'layout_targets':{str(k):sorted(v) for k,v in layout_targets.items()},
        'distinct_layouts':len(layout_targets),'complete_three_target_layouts':sum(len(x)==3 for x in layout_targets.values()),
        'frame_count_range':[min(frame_counts),max(frame_counts)],'minimum_verified_hold_s':min(holds),
        'target_xyz_min':np.min(positions,axis=0).tolist(),'target_xyz_max':np.max(positions,axis=0).tolist(),
        'same_objects_across_target_choices':True,'task_scope':'left_grasp_lift_hold',
        'sufficiency_established':False,'split_assignment_corrections':split_overrides,
        'note':'Expert and replay acceptance only; adequacy requires separate learned-policy development evaluation.'}
    (a.output/'coverage.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
