"""
Experiment 1: Baseline Performance Comparison

Train all models (RealMLP, ComplexMLP, RealCNN, ComplexCNN) on MNIST and CIFAR-10
Repeat each training 5 times with different seeds for statistical validation
Save best models and generate performance tables
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import json

# Add project root to path
sys.path.append('.')

import config
from models import RealMLP, ComplexMLP, RealCNN, ComplexCNN
from utils.data import get_dataloaders
from utils.metrics import count_parameters, evaluate_model, train_one_epoch
from utils.visualization import plot_training_curves, create_results_table

# Set device
device = config.DEVICE
print(f"Using device: {device}")


def train_model(model, train_loader, val_loader, test_loader, model_name,
                dataset_name, trial, epochs=50):
    """
    Train a single model

    Args:
        model: PyTorch model
        train_loader, val_loader, test_loader: Data loaders
        model_name: Name of the model (e.g., 'RealMLP')
        dataset_name: 'MNIST' or 'CIFAR10'
        trial: Trial number (0-4)
        epochs: Number of epochs to train

    Returns:
        Dictionary with training history and best results
    """
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                          lr=config.LEARNING_RATE,
                          weight_decay=config.WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training history
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Training {model_name} on {dataset_name} (Trial {trial + 1}/{config.NUM_TRIALS})")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_history['loss'].append(train_metrics['loss'])
        train_history['accuracy'].append(train_metrics['accuracy'])

        # Validate
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        val_history['loss'].append(val_metrics['loss'])
        val_history['accuracy'].append(val_metrics['accuracy'])

        # Update learning rate
        scheduler.step(val_metrics['accuracy'])

        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0

            # Save model checkpoint
            save_dir = f"{config.MODEL_SAVE_DIR}/{dataset_name}/{model_name}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/trial_{trial}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
            }, save_path)

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model for final test evaluation
    best_model_path = f"{config.MODEL_SAVE_DIR}/{dataset_name}/{model_name}/trial_{trial}_best.pth"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    test_metrics = evaluate_model(model, test_loader, device, criterion)

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"{'='*70}\n")

    # Plot training curves
    plot_save_path = f"{config.RESULTS_DIR}/figures/training_{dataset_name}_{model_name}_trial{trial}.png"
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plot_training_curves(train_history, val_history, save_path=plot_save_path)

    return {
        'train_history': train_history,
        'val_history': val_history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_metrics['accuracy'],
        'test_loss': test_metrics['loss']
    }


def run_experiment(dataset_name, model_class, model_config, model_name):
    """
    Run all trials for a single model on a dataset

    Args:
        dataset_name: 'MNIST' or 'CIFAR10'
        model_class: Model class (RealMLP, ComplexMLP, etc.)
        model_config: Configuration dict from config.py
        model_name: Name for saving/logging

    Returns:
        Dictionary with aggregated results
    """
    results = {
        'test_accuracies': [],
        'val_accuracies': [],
        'test_losses': []
    }

    # Get data loaders (will be reused for all trials)
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR,
        val_split=0.1,
        seed=config.RANDOM_SEED
    )

    # Run multiple trials
    for trial in range(config.NUM_TRIALS):
        # Set seed for this trial
        trial_seed = config.RANDOM_SEED + trial
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(trial_seed)

        # Create model
        if 'MLP' in model_name:
            model = model_class(
                input_size=model_config['input_size'],
                hidden_sizes=model_config['hidden_sizes'],
                output_size=model_config['output_size']
            )
        else:  # CNN
            model = model_class(
                conv_channels=model_config['conv_channels'],
                fc_size=model_config['fc_size'],
                output_size=model_config['output_size']
            )

        # Train
        trial_results = train_model(
            model, train_loader, val_loader, test_loader,
            model_name, dataset_name, trial, epochs=config.EPOCHS
        )

        # Store results
        results['test_accuracies'].append(trial_results['test_acc'])
        results['val_accuracies'].append(trial_results['best_val_acc'])
        results['test_losses'].append(trial_results['test_loss'])

    # Compute statistics
    results['mean_test_acc'] = np.mean(results['test_accuracies'])
    results['std_test_acc'] = np.std(results['test_accuracies'])
    results['mean_val_acc'] = np.mean(results['val_accuracies'])
    results['std_val_acc'] = np.std(results['val_accuracies'])

    return results


def main():
    """Main experiment runner"""

    print("\n" + "="*70)
    print("EXPERIMENT 1: BASELINE PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\nSettings:")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Trials per model: {config.NUM_TRIALS}")
    print(f"  - Device: {device}")
    print("="*70 + "\n")

    # Create directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(f"{config.RESULTS_DIR}/figures", exist_ok=True)
    os.makedirs(f"{config.RESULTS_DIR}/tables", exist_ok=True)

    # Store all results
    all_results = {}

    # =========================================================================
    # MNIST Experiments
    # =========================================================================
    print("\n" + "="*70)
    print("MNIST EXPERIMENTS")
    print("="*70 + "\n")

    # RealMLP on MNIST
    print("\n### RealMLP on MNIST ###\n")
    all_results['RealMLP_MNIST'] = run_experiment(
        'MNIST', RealMLP, config.MLP_CONFIG['real'], 'RealMLP'
    )

    # ComplexMLP on MNIST
    print("\n### ComplexMLP on MNIST ###\n")
    all_results['ComplexMLP_MNIST'] = run_experiment(
        'MNIST', ComplexMLP, config.MLP_CONFIG['complex'], 'ComplexMLP'
    )

    # =========================================================================
    # CIFAR-10 Experiments
    # =========================================================================
    print("\n" + "="*70)
    print("CIFAR-10 EXPERIMENTS")
    print("="*70 + "\n")

    # RealCNN on CIFAR-10
    print("\n### RealCNN on CIFAR-10 ###\n")
    all_results['RealCNN_CIFAR10'] = run_experiment(
        'CIFAR10', RealCNN, config.CNN_CONFIG['real'], 'RealCNN'
    )

    # ComplexCNN on CIFAR-10
    print("\n### ComplexCNN on CIFAR-10 ###\n")
    all_results['ComplexCNN_CIFAR10'] = run_experiment(
        'CIFAR10', ComplexCNN, config.CNN_CONFIG['complex'], 'ComplexCNN'
    )

    # =========================================================================
    # Generate Results Table
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING RESULTS TABLE")
    print("="*70 + "\n")

    # Create parameter count dict
    real_mlp = RealMLP(**{k: v for k, v in config.MLP_CONFIG['real'].items() if k != 'activation'})
    complex_mlp = ComplexMLP(**{k: v for k, v in config.MLP_CONFIG['complex'].items() if k != 'activation'})
    real_cnn = RealCNN(**{k: v for k, v in config.CNN_CONFIG['real'].items() if k not in ['activation', 'kernel_size']})
    complex_cnn = ComplexCNN(**{k: v for k, v in config.CNN_CONFIG['complex'].items() if k not in ['activation', 'kernel_size']})

    results_dict = {
        'Model': ['RealMLP', 'ComplexMLP', 'RealCNN', 'ComplexCNN'],
        'Parameters': [
            count_parameters(real_mlp),
            count_parameters(complex_mlp),
            count_parameters(real_cnn),
            count_parameters(complex_cnn)
        ],
        'MNIST Acc (%)': [
            all_results['RealMLP_MNIST']['mean_test_acc'],
            all_results['ComplexMLP_MNIST']['mean_test_acc'],
            '-',
            '-'
        ],
        'MNIST Std': [
            all_results['RealMLP_MNIST']['std_test_acc'],
            all_results['ComplexMLP_MNIST']['std_test_acc'],
            '-',
            '-'
        ],
        'CIFAR-10 Acc (%)': [
            '-',
            '-',
            all_results['RealCNN_CIFAR10']['mean_test_acc'],
            all_results['ComplexCNN_CIFAR10']['mean_test_acc']
        ],
        'CIFAR-10 Std': [
            '-',
            '-',
            all_results['RealCNN_CIFAR10']['std_test_acc'],
            all_results['ComplexCNN_CIFAR10']['std_test_acc']
        ]
    }

    # Save table
    table_path = f"{config.RESULTS_DIR}/tables/baseline_results.md"
    create_results_table(results_dict, save_path=table_path)

    # Save raw results as JSON
    json_path = f"{config.RESULTS_DIR}/baseline_results.json"
    with open(json_path, 'w') as f:
        json.dump({k: {ki: float(vi) if isinstance(vi, (np.floating, np.integer)) else vi
                      for ki, vi in v.items()}
                  for k, v in all_results.items()}, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - Table: {table_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Figures: {config.RESULTS_DIR}/figures/")
    print(f"  - Models: {config.MODEL_SAVE_DIR}/")

    print("\n" + "="*70)
    print("EXPERIMENT 1 COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
