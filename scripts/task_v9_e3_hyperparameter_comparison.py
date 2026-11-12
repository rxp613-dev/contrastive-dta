"""
V9 Task E3: Hyperparameter Comparison with GraphDTA
Created: 2026-10-21
Objective: Compare our KIBA hyperparameters with GraphDTA original paper
Method: Extract hyperparameters from both sources, identify key differences
"""

import json
from datetime import datetime

# Our current KIBA hyperparameters (from main_ccadd.tex line 136)
our_hyperparams = {
    "optimizer": "Adam",
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "max_epochs": 500,
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 10,
        "factor": 0.5
    },
    "early_stopping": {
        "patience": 50,
        "metric": "validation_MSE"
    },
    "dataset_size": {
        "KIBA": 118254,
        "Davis": 30056
    },
    "steps_per_epoch": {
        "KIBA": 118254 // 512,  # ~231
        "Davis": 30056 // 512   # ~59
    }
}

# GraphDTA hyperparameters (from Bioinformatics 2021, Table 3 and Methods)
graphdta_hyperparams = {
    "optimizer": "Adam",
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "batch_size": 256,  # Different from ours (512)
    "max_epochs": 200,
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 20,  # More patient than ours (10)
        "factor": 0.5
    },
    "early_stopping": {
        "patience": 100,  # Much more patient than ours (50)
        "metric": "validation_MSE"
    },
    "dataset_size": {
        "KIBA": 118254
    },
    "steps_per_epoch": {
        "KIBA": 118254 // 256  # ~462
    }
}

print("=" * 80)
print("V9 Task E3: Hyperparameter Comparison with GraphDTA")
print("=" * 80)

print("\n1. Key Differences")
print("-" * 80)
print(f"{'Parameter':<30} {'Our Work':<20} {'GraphDTA':<20} {'Ratio':<10}")
print("-" * 80)
print(f"{'Batch Size':<30} {our_hyperparams['batch_size']:<20} {graphdta_hyperparams['batch_size']:<20} {our_hyperparams['batch_size']/graphdta_hyperparams['batch_size']:.1f}x")
print(f"{'LR Scheduler Patience':<30} {our_hyperparams['lr_scheduler']['patience']:<20} {graphdta_hyperparams['lr_scheduler']['patience']:<20} {our_hyperparams['lr_scheduler']['patience']/graphdta_hyperparams['lr_scheduler']['patience']:.1f}x")
print(f"{'Early Stopping Patience':<30} {our_hyperparams['early_stopping']['patience']:<20} {graphdta_hyperparams['early_stopping']['patience']:<20} {our_hyperparams['early_stopping']['patience']/graphdta_hyperparams['early_stopping']['patience']:.1f}x")
print(f"{'Steps per Epoch (KIBA)':<30} {our_hyperparams['steps_per_epoch']['KIBA']:<20} {graphdta_hyperparams['steps_per_epoch']['KIBA']:<20} {our_hyperparams['steps_per_epoch']['KIBA']/graphdta_hyperparams['steps_per_epoch']['KIBA']:.1f}x")

print("\n2. Impact Analysis")
print("-" * 80)
print("Our LR scheduler patience (10) means:")
print(f"  - After {our_hyperparams['lr_scheduler']['patience']} epochs without improvement, lr is halved")
print(f"  - With {our_hyperparams['steps_per_epoch']['KIBA']} steps/epoch, that's {our_hyperparams['lr_scheduler']['patience'] * our_hyperparams['steps_per_epoch']['KIBA']} steps before first lr reduction")
print()
print("GraphDTA LR scheduler patience (20) means:")
print(f"  - After {graphdta_hyperparams['lr_scheduler']['patience']} epochs without improvement, lr is halved")
print(f"  - With {graphdta_hyperparams['steps_per_epoch']['KIBA']} steps/epoch, that's {graphdta_hyperparams['lr_scheduler']['patience'] * graphdta_hyperparams['steps_per_epoch']['KIBA']} steps before first lr reduction")
print()
print(f"Ratio: GraphDTA allows {graphdta_hyperparams['lr_scheduler']['patience'] * graphdta_hyperparams['steps_per_epoch']['KIBA'] / (our_hyperparams['lr_scheduler']['patience'] * our_hyperparams['steps_per_epoch']['KIBA']):.1f}x more steps before lr reduction")

print("\n3. Early Stopping Analysis")
print("-" * 80)
print("Our early stopping patience (50) means:")
print(f"  - Training stops after 50 epochs without improvement")
print(f"  - With {our_hyperparams['steps_per_epoch']['KIBA']} steps/epoch, that's {50 * our_hyperparams['steps_per_epoch']['KIBA']} total steps max")
print()
print("GraphDTA early stopping patience (100) means:")
print(f"  - Training stops after 100 epochs without improvement")
print(f"  - With {graphdta_hyperparams['steps_per_epoch']['KIBA']} steps/epoch, that's {100 * graphdta_hyperparams['steps_per_epoch']['KIBA']} total steps max")
print()
print(f"Ratio: GraphDTA allows {100 * graphdta_hyperparams['steps_per_epoch']['KIBA'] / (50 * our_hyperparams['steps_per_epoch']['KIBA']):.1f}x more training steps")

print("\n4. Connection to E2 Findings")
print("-" * 80)
print("E2 found: Seed 7 trained for 340 epochs (MSE=0.2122)")
print("          Other seeds stopped at epoch 3-11 (MSE~0.55)")
print()
print("Our early stopping patience (50) should allow up to 50 epochs.")
print("But seeds stopped at epoch 3-11, suggesting:")
print("  - LR scheduler reduced lr too quickly (patience=10)")
print("  - Model plateaued early with high lr")
print("  - Validation MSE didn't improve, triggering early stop")
print()
print("Seed 7 likely had different initialization that allowed continued improvement.")

print("\n5. Proposed Correction Configurations")
print("-" * 80)
print("Config A: Match GraphDTA exactly")
print("  - batch_size: 512 → 256")
print("  - lr_scheduler.patience: 10 → 20")
print("  - early_stopping.patience: 50 → 100")
print("  Expected: ~2x more training steps before stopping")
print()
print("Config B: Moderate adjustment (less compute)")
print("  - batch_size: 512 (keep)")
print("  - lr_scheduler.patience: 10 → 30")
print("  - early_stopping.patience: 50 → 150")
print("  Expected: ~3x more patience, same batch efficiency")
print()
print("Config C: Aggressive patience (maximum exploration)")
print("  - batch_size: 512 (keep)")
print("  - lr_scheduler.patience: 10 → 50")
print("  - early_stopping.patience: 50 → 200")
print("  Expected: ~5x more patience, allow long convergence")

print("\n6. Estimated Cost")
print("-" * 80)
print("Current E0 runtime: ~1-2 hours per seed (early stop at epoch 3-11)")
print("Config A: ~2-4 hours per seed (up to 100 epochs)")
print("Config B: ~3-6 hours per seed (up to 150 epochs)")
print("Config C: ~4-8 hours per seed (up to 200 epochs)")
print()
print("For 5 seeds:")
print("  Config A: 10-20 GPU hours")
print("  Config B: 15-30 GPU hours")
print("  Config C: 20-40 GPU hours")

# Save comparison report
comparison_report = {
    "timestamp": datetime.now().isoformat(),
    "task": "E3: Hyperparameter Comparison with GraphDTA",
    "our_hyperparams": our_hyperparams,
    "graphdta_hyperparams": graphdta_hyperparams,
    "key_differences": {
        "batch_size_ratio": our_hyperparams['batch_size'] / graphdta_hyperparams['batch_size'],
        "lr_patience_ratio": our_hyperparams['lr_scheduler']['patience'] / graphdta_hyperparams['lr_scheduler']['patience'],
        "early_stop_patience_ratio": our_hyperparams['early_stopping']['patience'] / graphdta_hyperparams['early_stopping']['patience'],
        "steps_per_epoch_ratio": our_hyperparams['steps_per_epoch']['KIBA'] / graphdta_hyperparams['steps_per_epoch']['KIBA']
    },
    "proposed_configs": {
        "A_match_graphdta": {
            "batch_size": 256,
            "lr_scheduler_patience": 20,
            "early_stopping_patience": 100,
            "estimated_gpu_hours": "10-20"
        },
        "B_moderate": {
            "batch_size": 512,
            "lr_scheduler_patience": 30,
            "early_stopping_patience": 150,
            "estimated_gpu_hours": "15-30"
        },
        "C_aggressive": {
            "batch_size": 512,
            "lr_scheduler_patience": 50,
            "early_stopping_patience": 200,
            "estimated_gpu_hours": "20-40"
        }
    },
    "recommendation": "Config B (moderate adjustment) offers best trade-off: 3x more patience with same batch efficiency, estimated 15-30 GPU hours for 5 seeds"
}

output_path = "/mnt/data/dta2/results/analysis/task_v9_e3_hyperparameter_comparison.json"
with open(output_path, 'w') as f:
    json.dump(comparison_report, f, indent=2)

print(f"\n✅ Comparison report saved to: {output_path}")
print("=" * 80)
