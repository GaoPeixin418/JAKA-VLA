"""Synthetic tensor smoke test. This is not task evaluation or a success metric."""
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

root = Path('/data/jaka-vla')
weights = root / 'cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
cfg = PreTrainedConfig.from_pretrained(weights)
cfg.input_features = {'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(14,))}
for i in range(1, 4):
    cfg.input_features[f'observation.images.camera{i}'] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320))
cfg.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
cfg.device = 'cuda'
cfg.push_to_hub = False
policy = SmolVLAPolicy.from_pretrained(weights, config=cfg).to('cuda').eval()
tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
tokens = tokenizer(['Pass the red block from the left arm to the right arm and place it on the blue target.'],
                   padding='max_length', max_length=cfg.tokenizer_max_length, truncation=True, return_tensors='pt')
batch = {'observation.state': torch.zeros(1, 14, device='cuda'),
         'observation.language.tokens': tokens['input_ids'].to('cuda'),
         'observation.language.attention_mask': tokens['attention_mask'].to('cuda').bool()}
for i in range(1, 4):
    batch[f'observation.images.camera{i}'] = torch.zeros(1, 3, 240, 320, device='cuda')
durations = []
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    for i in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        if i >= 2:
            durations.append(time.perf_counter() - start)
assert actions.shape == (1, cfg.chunk_size, 14), actions.shape
assert torch.isfinite(actions).all()
report = {'test': 'synthetic_input_only_not_task_evaluation', 'action_shape': list(actions.shape),
          'parameters': sum(p.numel() for p in policy.parameters()),
          'trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
          'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
          'steady_inference_seconds': durations,
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
(root / 'results/policy_smoke.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
