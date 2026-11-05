#!/usr/bin/env bash
# Bounded first pilot: 50 episodes, 2,000 train steps, 3 held-out seeds for each K.
set -euo pipefail
source /data/jaka-vla/env.sh
status_file=/data/jaka-vla/results/pilot_status.txt
on_error() {
  local code=$?
  printf 'failed at %s (exit %s)\n' "$(date -Iseconds)" "$code" > "$status_file"
  exit "$code"
}
trap on_error ERR
printf 'collecting\n' > "$status_file"
source /data/jaka-vla/envs/robotwin/bin/activate
cd /data/jaka-vla/repos/RoboTwin
python /data/jaka-vla/scripts/configure_smoke.py
python /data/jaka-vla/scripts/patch_sim_timing.py
timeout 5400 python -u scripts/collect_data.py handover_block jaka_pilot > /data/jaka-vla/logs/collect-pilot.log 2>&1
deactivate
source /data/jaka-vla/envs/vla/bin/activate
printf 'converting\n' > "$status_file"
python -u /data/jaka-vla/scripts/convert_dataset.py \
  --source /data/jaka-vla/data/raw/jaka_pilot/handover_block/aloha_agilex/data \
  --output /data/jaka-vla/data/lerobot/handover_pilot \
  --repo-id jaka-local/handover-pilot > /data/jaka-vla/logs/convert-pilot.log 2>&1
printf 'training\n' > "$status_file"
timeout 5400 python /data/jaka-vla/scripts/run_training.py \
  --policy.path=/data/jaka-vla/cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --policy.input_features='{}' --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=true \
  --dataset.repo_id=jaka-local/handover-pilot --dataset.root=/data/jaka-vla/data/lerobot/handover_pilot \
  --dataset.video_backend=pyav --output_dir=/data/jaka-vla/checkpoints/smolvla_pilot_seed42 \
  --job_name=smolvla_pilot_seed42 --batch_size=8 --steps=2000 --save_freq=2000 \
  --eval_freq=0 --log_freq=100 --num_workers=4 --wandb.enable=false --seed=42 \
  > /data/jaka-vla/logs/train-pilot.log 2>&1
printf 'evaluating_fixed_k\n' > "$status_file"
timeout 6000 python -u /data/jaka-vla/scripts/run_baselines.py > /data/jaka-vla/logs/eval-baselines.log 2>&1
printf 'pilot_complete_review_results_before_scaling\n' > "$status_file"
