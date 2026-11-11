"""Explicit, local-only pi0.5 LoRA configuration for Zu7 experiments."""
from pathlib import Path
import json

import flax.nnx as nnx
from openpi.models.pi0_config import Pi0Config
from openpi.shared.nnx_utils import PathRegex
from openpi.training import config, optimizer, weight_loaders

from jaka_vla.openpi_zu7 import LeRobotZu7DataConfig

ROOT = Path('/data/jaka-vla')
BASE_PARAMS = '/data/pi05-cache/models/openpi-assets/checkpoints/pi05_base/params'


def build_config(dataset, experiment, *, steps=3, batch_size=1, horizon=50):
    identity_path=ROOT/'data/openpi'/dataset/'meta/project_identity.json'
    robot_id=json.loads(identity_path.read_text())['robot_id'] if identity_path.exists() else 'dual_jaka_zu7'
    model = Pi0Config(pi05=True, action_horizon=horizon, discrete_state_input=True,
                      paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    # Freeze SigLIP as well as original language/action-expert weights.
    # The upstream default LoRA filter only freezes the LLM; that would train SigLIP too.
    trainable = nnx.Any(PathRegex('.*lora.*'),
                        PathRegex('(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*'))
    return config.TrainConfig(
        name='pi05_zu7_lora_h' + str(horizon), exp_name=experiment, model=model,
        freeze_filter=nnx.Not(trainable), ema_decay=None,
        data=LeRobotZu7DataConfig(repo_id=dataset,
            assets=config.AssetsConfig(asset_id=dataset),
            base_config=config.DataConfig(prompt_from_task=True)),
        assets_base_dir=str(ROOT/'data/openpi_assets'),
        checkpoint_base_dir=str(ROOT/'checkpoints/openpi'),
        weight_loader=weight_loaders.CheckpointWeightLoader(BASE_PARAMS),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10, peak_lr=2e-5,
                                                   decay_steps=max(steps, 1000), decay_lr=2e-6),
        batch_size=batch_size, num_workers=0, num_train_steps=steps,
        log_interval=10, save_interval=max(steps, 1), keep_period=None,
        wandb_enabled=False,
        policy_metadata={'robot_id':robot_id, 'action_semantics':'next_sample_absolute_drive_targets',
                         'state_semantics':'measured_qpos_and_gripper_aperture', 'horizon':horizon,
                         'discrete_state_input':True, 'vision_backbone_frozen':True})
