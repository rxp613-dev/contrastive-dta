#!/bin/bash
# ============================================================================
# Step 1.1: Re-run GAT Cold-Target Experiments with FULLY Authoritative Protocol
# Date: 2026-09-04
# Objective: Fix the beta=0.05, gamma=0.05 deviation in Batch 3
#
# Background:
#   Batch 3 (GAT_E{0,7}_cold_target_correct_seed*) used:
#     - beta: 0.05 ❌ (should be 0.0 for E7)
#     - gamma: 0.05 ❌ (should be 0.0 for E7)
#
#   According to the paper's E7 definition:
#     E7 = "noise + InfoNCE (α=0.05)" with NO Uniformity or CrossModal losses
#     Therefore: beta=0.0, gamma=0.0
#
# Authoritative Protocol (from Table 4 + E7 definition):
#   - learning_rate: 0.0005
#   - batch_size: 512
#   - max_epochs: 500
#   - early_stop_patience: 50
#   - weight_decay: 0.0001
#   - alpha: 0.05 (for E7)
#   - beta: 0.0 (for E7)
#   - gamma: 0.0 (for E7)
#   - augmentation: noise
#   - architecture: GAT
#   - split_type: cold_target
#   - dataset: davis
#
# Experiments:
#   - 5 seeds: [7, 42, 123, 2024, 3407]
#   - 2 conditions: E0 (no CL) and E7 (with CL)
#   - Total: 10 experiments
#   - Estimated time: 5-10 GPU hours
#
# Output:
#   Results saved to: /home/rxp/dta2/results/cold_start/GAT_E{0,7}_cold_target_authoritative_seed*
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Step 1.1: GAT Cold-Target Authoritative Protocol Re-run"
echo "Date: $(date)"
echo "=========================================="

# Seed 7
echo 'Starting: GAT_E0_cold_target_authoritative_seed7'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E0_cold_target_authoritative_seed7 \
  --architecture GAT \
  --split_type cold_target \
  --seed 7 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --augmentation noise
echo 'Completed: GAT_E0_cold_target_authoritative_seed7'

echo 'Starting: GAT_E7_cold_target_authoritative_seed7'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E7_cold_target_authoritative_seed7 \
  --architecture GAT \
  --split_type cold_target \
  --seed 7 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --use_contrastive \
  --alpha 0.05 \
  --beta 0.0 \
  --gamma 0.0 \
  --augmentation noise
echo 'Completed: GAT_E7_cold_target_authoritative_seed7'

# Seed 42
echo 'Starting: GAT_E0_cold_target_authoritative_seed42'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E0_cold_target_authoritative_seed42 \
  --architecture GAT \
  --split_type cold_target \
  --seed 42 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --augmentation noise
echo 'Completed: GAT_E0_cold_target_authoritative_seed42'

echo 'Starting: GAT_E7_cold_target_authoritative_seed42'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E7_cold_target_authoritative_seed42 \
  --architecture GAT \
  --split_type cold_target \
  --seed 42 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --use_contrastive \
  --alpha 0.05 \
  --beta 0.0 \
  --gamma 0.0 \
  --augmentation noise
echo 'Completed: GAT_E7_cold_target_authoritative_seed42'

# Seed 123
echo 'Starting: GAT_E0_cold_target_authoritative_seed123'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E0_cold_target_authoritative_seed123 \
  --architecture GAT \
  --split_type cold_target \
  --seed 123 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --augmentation noise
echo 'Completed: GAT_E0_cold_target_authoritative_seed123'

echo 'Starting: GAT_E7_cold_target_authoritative_seed123'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E7_cold_target_authoritative_seed123 \
  --architecture GAT \
  --split_type cold_target \
  --seed 123 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --use_contrastive \
  --alpha 0.05 \
  --beta 0.0 \
  --gamma 0.0 \
  --augmentation noise
echo 'Completed: GAT_E7_cold_target_authoritative_seed123'

# Seed 2024
echo 'Starting: GAT_E0_cold_target_authoritative_seed2024'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E0_cold_target_authoritative_seed2024 \
  --architecture GAT \
  --split_type cold_target \
  --seed 2024 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --augmentation noise
echo 'Completed: GAT_E0_cold_target_authoritative_seed2024'

echo 'Starting: GAT_E7_cold_target_authoritative_seed2024'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E7_cold_target_authoritative_seed2024 \
  --architecture GAT \
  --split_type cold_target \
  --seed 2024 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --use_contrastive \
  --alpha 0.05 \
  --beta 0.0 \
  --gamma 0.0 \
  --augmentation noise
echo 'Completed: GAT_E7_cold_target_authoritative_seed2024'

# Seed 3407
echo 'Starting: GAT_E0_cold_target_authoritative_seed3407'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E0_cold_target_authoritative_seed3407 \
  --architecture GAT \
  --split_type cold_target \
  --seed 3407 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --augmentation noise
echo 'Completed: GAT_E0_cold_target_authoritative_seed3407'

echo 'Starting: GAT_E7_cold_target_authoritative_seed3407'
python /home/rxp/dta2/scripts/train_controlled.py \
  --experiment_id GAT_E7_cold_target_authoritative_seed3407 \
  --architecture GAT \
  --split_type cold_target \
  --seed 3407 \
  --learning_rate 0.0005 \
  --batch_size 512 \
  --max_epochs 500 \
  --early_stop_patience 50 \
  --weight_decay 0.0001 \
  --use_contrastive \
  --alpha 0.05 \
  --beta 0.0 \
  --gamma 0.0 \
  --augmentation noise
echo 'Completed: GAT_E7_cold_target_authoritative_seed3407'

echo "=========================================="
echo "All Step 1.1 experiments completed!"
echo "Date: $(date)"
echo "=========================================="
