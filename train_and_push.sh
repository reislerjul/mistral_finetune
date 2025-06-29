#!/bin/bash

# Exit immediately on error
set -e

# Optional: set log file
LOG_FILE="overnight_run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Starting training at $(date) ==="

# Step 1: Run training
python run_training.py --config_path configs/dpo_config_combined.yaml

echo "=== Training complete at $(date) ==="

# Step 2: Push to Hugging Face Hub
python push_checkpoints_to_hub.py \
  --base_path /home/ubuntu/mistral-finetune/mistral_finetune/models/mistral-combined/ \
  --hf_user jreisler

echo "=== Upload complete at $(date) ==="
