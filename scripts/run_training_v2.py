"""Use an explicit dataset-derived config and validate it before loading weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys,time
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    raw=json.loads(args.config.read_text());identity=json.loads(args.config.with_suffix('.identity.json').read_text())
    # Remove our own CLI before upstream validation so policy.path cannot override our explicit features.
    sys.argv=[sys.argv[0]]
    import draccus
    with draccus.config_type('json'):
        cfg=draccus.parse(TrainPipelineConfig,config_path=str(args.config),args=[])
    size=identity['state_dim']
    assert cfg.policy.input_features['observation.state'].shape==(size,)
    assert cfg.policy.output_features['action'].shape==(size,)
    assert raw['dataset']['root']==identity['dataset']
    stats_path=Path(raw['dataset']['root'])/'meta/stats.json'
    stats=json.loads(stats_path.read_text())
    state_std=stats['observation.state']['std']
    if min(state_std)<.001:
        raise ValueError('Near-constant state dimensions amplify tracking noise; run stabilize_zu7_statistics.py on a new dataset copy before training')
    if 'normalization' in identity:
        if hashlib.sha256(stats_path.read_bytes()).hexdigest()!=identity['normalization']['effective_stats_sha256']:
            raise ValueError('Normalization statistics changed after training config creation')
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();ok=False
    try:
        train(cfg);torch.cuda.synchronize();ok=True
    finally:
        output=Path(raw['output_dir'])
        if output.exists():
            (output/'resource_profile.json').write_text(json.dumps({'completed':ok,'wall_seconds_including_load_and_save':time.perf_counter()-start,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'robot_id':identity['robot_id']},indent=2)+'\n')
            for checkpoint in (output/'checkpoints').glob('*/pretrained_model'):
                config=json.loads((checkpoint/'config.json').read_text())
                assert config['input_features']['observation.state']['shape']==[size]
                (checkpoint/'project_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    if not ok:raise RuntimeError('Training did not complete')


if __name__=='__main__':main()
