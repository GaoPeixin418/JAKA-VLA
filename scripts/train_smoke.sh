#!/usr/bin/env bash
set -euo pipefail
source /data/jaka-vla/env.sh
source /data/jaka-vla/envs/vla/bin/activate
cd /data/jaka-vla
exec python /data/jaka-vla/scripts/run_training.py \
  --policy.path=/data/jaka-vla/cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --policy.input_features='{}' \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.use_amp=true \
  --dataset.repo_id=jaka-local/handover-smoke \
  --dataset.root=/data/jaka-vla/data/lerobot/handover_smoke \
  --dataset.video_backend=pyav \
  --output_dir=/data/jaka-vla/checkpoints/smolvla_smoke \
  --job_name=smolvla_smoke \
  --batch_size=8 \
  --steps=20 \
  --save_freq=20 \
  --eval_freq=0 \
  --log_freq=1 \
  --num_workers=0 \
  --wandb.enable=false \
  --seed=42
