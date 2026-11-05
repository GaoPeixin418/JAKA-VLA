"""Reload trained weights and processors and infer on real demonstration data."""
import json
from pathlib import Path
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

root = Path('/data/jaka-vla')
checkpoint = root / 'checkpoints/smolvla_smoke/checkpoints/last/pretrained_model'
dataset = LeRobotDataset('jaka-local/handover-smoke', root=root / 'data/lerobot/handover_smoke', video_backend='pyav')
policy = SmolVLAPolicy.from_pretrained(checkpoint).to('cuda').eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
    preprocessor_overrides={'device_processor': {'device': 'cuda'}})
with torch.inference_mode():
    batch = pre(dataset[0])
    normalized_actions = policy.predict_action_chunk(batch)
    physical_actions = post(normalized_actions)
assert physical_actions.shape == (1, 50, 14), physical_actions.shape
assert torch.isfinite(physical_actions).all()
report = {'checkpoint': str(checkpoint.resolve()), 'reload_ok': True,
          'normalization_state_shape': list(batch['observation.state'].shape),
          'physical_action_shape': list(physical_actions.shape),
          'test': 'checkpoint_and_processors_roundtrip_only_not_task_success'}
(root / 'results/checkpoint_smoke.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
