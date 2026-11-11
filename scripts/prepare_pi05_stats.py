"""Audit episode boundaries and compute Zu7 statistics after joint delta conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
ROOT=Path('/data/jaka-vla')
sys.path.insert(0,str(ROOT))
from openpi.shared import normalize
from jaka_vla.pi05_config import build_config
from jaka_vla.openpi_zu7 import JOINT_DELTA_MASK


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--horizon',type=int,default=50)
    args=p.parse_args(); source=ROOT/'data/openpi'/args.dataset
    config=build_config(args.dataset,'stats',horizon=args.horizon)
    dest=config.assets_dirs/args.dataset
    if dest.exists(): raise FileExistsError(dest)
    info=json.loads((source/'meta/info.json').read_text())
    if info['fps']!=25: raise ValueError('25Hz required')
    episodes=[json.loads(x) for x in (source/'meta/episodes.jsonl').read_text().splitlines()]
    states=[]; chunks=[]; records=[]
    mask=np.array(JOINT_DELTA_MASK)
    for ep in episodes:
        i=ep['episode_index'];path=source/info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i)
        rows=pq.read_table(path).to_pylist()
        if len(rows)!=ep['length'] or any(r['episode_index']!=i or r['frame_index']!=j for j,r in enumerate(rows)):
            raise ValueError('Episode/frame mismatch')
        t=np.array([r['timestamp'] for r in rows])
        if len(rows)>1 and not np.allclose(np.diff(t),.04,atol=1e-5): raise ValueError('Nonuniform time')
        s=np.array([r['state'] for r in rows],dtype=np.float64)
        a=np.array([r['actions'] for r in rows],dtype=np.float64)
        if s.shape!=(len(rows),14) or a.shape!=s.shape or not np.isfinite(s).all() or not np.isfinite(a).all():
            raise ValueError('Invalid state/action')
        # Match LeRobot: every chunk is clipped to its own episode; final action repeats.
        idx=np.minimum(np.arange(len(rows))[:,None]+np.arange(args.horizon)[None,:],len(rows)-1)
        c=a[idx].copy();c[:,:,mask]-=s[:,None,mask]
        states.append(s);chunks.append(c.reshape(-1,14))
        records.append({'episode':i,'frames':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    stats={};audit={}
    for key,parts in [('state',states),('actions',chunks)]:
        x=np.concatenate(parts);q01,q99=np.quantile(x,[.01,.99],axis=0)
        floor=np.full(14,.05 if key=='state' else .01);floor[[6,13]]=.05
        mid=(q01+q99)/2; half=np.maximum((q99-q01)/2,floor)
        stats[key]=normalize.NormStats(mean=x.mean(0),std=np.maximum(x.std(0),floor),q01=mid-half,q99=mid+half)
        audit[key]={'samples':len(x),'raw_q01':q01.tolist(),'raw_q99':q99.tolist(),
                    'half_range_floor':floor.tolist(),'widened_dims':np.flatnonzero((q99-q01)/2<floor).tolist()}
    normalize.save(dest,stats)
    report={'dataset':args.dataset,'horizon':args.horizon,'episodes':records,'frames':sum(x['frames'] for x in records),
            'joint_delta_mask':JOINT_DELTA_MASK,'padding':'repeat last action within episode',
            'quantile_normalization':True,'stats_audit':audit,'scope':'training-source statistics only'}
    (dest/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(dest),'episodes':len(episodes),'frames':report['frames'],'stats_audit':audit}),flush=True)

if __name__=='__main__': main()
