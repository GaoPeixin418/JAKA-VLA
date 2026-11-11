"""Behavior checks for real LeRobot chunks, ownership, finite outputs, and state tokens."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path('/data/jaka-vla');sys.path.insert(0,str(ROOT))
from openpi import transforms
from openpi.training.data_loader import create_torch_dataset, transform_dataset
from jaka_vla.pi05_config import build_config
from jaka_vla.openpi_zu7 import Zu7Inputs, Zu7Outputs, JOINT_DELTA_MASK


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg=build_config(a.dataset,'audit');dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    ds=create_torch_dataset(dc,50,cfg.model);transformed=transform_dataset(ds,dc)
    source=ROOT/'data/openpi'/a.dataset;eps=[json.loads(x) for x in (source/'meta/episodes.jsonl').read_text().splitlines()]
    checks=[];offset=0;mask=np.array(JOINT_DELTA_MASK)
    for ep in eps:
        start=ds[offset];end=ds[offset+ep['length']-1]
        for label,sample in [('first',start),('last',end)]:
            s=np.asarray(sample['state']).copy();acts=np.asarray(sample['actions']).copy()
            inp=Zu7Inputs()(sample);delta=transforms.DeltaActions(JOINT_DELTA_MASK)(inp)
            expected=acts.copy();expected[:,mask]-=s[mask]
            np.testing.assert_allclose(delta['actions'],expected,atol=1e-6)
            inverse=transforms.AbsoluteActions(JOINT_DELTA_MASK)(delta)
            np.testing.assert_allclose(inverse['actions'],acts,atol=1e-6)
            np.testing.assert_array_equal(np.asarray(sample['actions']),acts)
            np.testing.assert_array_equal(np.asarray(sample['state']),s)
            if label=='last':np.testing.assert_array_equal(acts,np.repeat(acts[:1],50,axis=0))
            checks.append({'episode':ep['episode_index'],'sample':label,'roundtrip':True,'source_unmodified':True})
        model_sample=transformed[offset]
        assert model_sample['state'].shape==(32,) and model_sample['actions'].shape==(50,32)
        assert all(x.shape==(224,224,3) for x in model_sample['image'].values())
        assert np.isfinite(model_sample['state']).all() and np.isfinite(model_sample['actions']).all()
        offset+=ep['length']
    for val in [np.nan,np.inf,-np.inf]:
        bad=np.zeros((50,32));bad[0,31]=val
        try:Zu7Outputs()({'actions':bad})
        except ValueError:pass
        else:raise AssertionError('Nonfinite prediction accepted')
    sample=ds[0];base=transforms.compose([*dc.data_transforms.inputs,transforms.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])
    original=base(sample);changed=dict(sample);changed['state']=np.asarray(sample['state']).copy();changed['state'][0]-=.2
    modified=base(changed)
    if np.array_equal(original['tokenized_prompt'],modified['tokenized_prompt']):raise AssertionError('State missing from pi0.5 tokens')
    report={'passed':True,'episodes':len(eps),'frames':offset,'horizon':50,'discrete_state_input':True,
            'boundary_checks':checks,'nonfinite_rejection':True,'state_changes_tokens':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
