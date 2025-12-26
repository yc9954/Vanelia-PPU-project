"""
Experiment 2: Neuron Damage Robustness Test

Load trained models and test their robustness to neuron damage
- Test damage rates: 0%, 10%, 20%, ..., 90%
- Multiple trials per damage rate (different random masks)
- Generate robustness curves showing accuracy degradation
"""

import torch
import numpy as np
import os
import sys
import json
from tqdm import tqdm

sys.path.append('.')

import config
from models import RealMLP, ComplexMLP, RealCNN, ComplexCNN
from utils.data import get_dataloaders
from utils.metrics import count_parameters, evaluate_with_damage, evaluate_model
from utils.visualization import plot_robustness_curves

device = config.DEVICE
print(f"Using device: {device}")


def load_best_model(model, dataset_name, model_name, trial=0):
    """
    Load the best saved model from Experiment 1

    Args:
        model: Model instance
        dataset_name: 'MNIST' or 'CIFAR10'
        model_name: e.g., 'RealMLP'
        trial: Which trial to load (0-4)

    Returns:
        Loaded model
    """
    model_path = f"{config.MODEL_SAVE_DIR}/{dataset_name}/{model_name}/trial_{trial}_best.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nPlease run Experiment 1 first!")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded {model_name} from {model_path}")
    print(f"  Validation accuracy when saved: {checkpoint['val_accuracy']:.2f}%")

    return model


def test_damage_robustness(model, test_loader, model_name, damage_rates=None):
    """
    Test model robustness across different damage rates

    Args:
        model: Trained model
        test_loader: Test data loader
        model_name: Name for logging
        damage_rates: List of damage rates to test (0.0 to 1.0)

    Returns:
        Dictionary with results for each damage rate
    """
    if damage_rates is None:
        damage_rates = config.DAMAGE_RATES

    results = {
        'damage_rates': [],
        'mean_accuracies': [],
        'std_accuracies': [],
        'all_accuracies': []
    }

    print(f"\n{'='*70}")
    print(f"Testing {model_name} Robustness")
    print(f"{'='*70}\n")

    for damage_rate in tqdm(damage_rates, desc=f"{model_name} Damage Test"):
        # Test with multiple random masks
        damage_results = evaluate_with_damage(
            model=model,
            dataloader=test_loader,
            device=device,
            damage_rate=damage_rate,
            seed=config.RANDOM_SEED,
            layer_idx=None,  # Damage all layers
            num_trials=config.DAMAGE_TRIALS_PER_RATE
        )

        results['damage_rates'].append(damage_rate)
        results['mean_accuracies'].append(damage_results['mean'])
        results['std_accuracies'].append(damage_results['std'])
        results['all_accuracies'].append(damage_results['accuracies'])

        print(f"  Damage {damage_rate*100:3.0f}%: Accuracy = {damage_results['mean']:.2f}% ± {damage_results['std']:.2f}%")

    print(f"\n{'='*70}\n")

    return results


def run_experiment_for_dataset(dataset_name, model_configs):
    """
    Run damage robustness experiment for all models on a dataset

    Args:
        dataset_name: 'MNIST' or 'CIFAR10'
        model_configs: List of (model_class, config_dict, model_name) tuples

    Returns:
        Dictionary with all results
    """
    print(f"\n{'#'*70}")
    print(f"# DAMAGE ROBUSTNESS: {dataset_name}")
    print(f"{'#'*70}\n")

    # Load test data
    _, _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR,
        seed=config.RANDOM_SEED
    )

    all_results = {}

    for model_class, model_config, model_name in model_configs:
        # Create model instance
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

        # Load best trained model from Experiment 1 (trial 0)
        model = load_best_model(model, dataset_name, model_name, trial=0)

        # Test baseline (no damage) first
        print(f"\nBaseline (no damage) test:")
        baseline_result = evaluate_model(model, test_loader, device)
        print(f"  Accuracy: {baseline_result['accuracy']:.2f}%")

        # Test robustness to damage
        results = test_damage_robustness(model, test_loader, model_name)
        results['baseline_accuracy'] = baseline_result['accuracy']
        results['model_name'] = model_name

        all_results[model_name] = results

    return all_results


def main():
    """Main experiment runner"""

    print("\n" + "="*70)
    print("EXPERIMENT 2: NEURON DAMAGE ROBUSTNESS")
    print("="*70)
    print(f"\nSettings:")
    print(f"  - Damage rates: {[f'{int(d*100)}%' for d in config.DAMAGE_RATES]}")
    print(f"  - Trials per damage rate: {config.DAMAGE_TRIALS_PER_RATE}")
    print(f"  - Device: {device}")
    print("="*70 + "\n")

    # Check if Experiment 1 models exist
    model_dir = config.MODEL_SAVE_DIR
    if not os.path.exists(model_dir):
        print(f"❌ ERROR: No trained models found in {model_dir}")
        print(f"Please run Experiment 1 first to train models!")
        print(f"\nRun: python experiments/exp1_baseline.py")
        return

    # Create results directory
    os.makedirs(f"{config.RESULTS_DIR}/figures", exist_ok=True)
    os.makedirs(f"{config.RESULTS_DIR}/tables", exist_ok=True)

    all_results = {}

    # =========================================================================
    # MNIST Experiments
    # =========================================================================
    mnist_configs = [
        (RealMLP, config.MLP_CONFIG['real'], 'RealMLP'),
        (ComplexMLP, config.MLP_CONFIG['complex'], 'ComplexMLP')
    ]

    mnist_results = run_experiment_for_dataset('MNIST', mnist_configs)
    all_results.update(mnist_results)

    # Plot MNIST robustness curves
    mnist_plot_data = {
        name: {
            'damage_rates': results['damage_rates'],
            'mean_accuracies': results['mean_accuracies'],
            'std_accuracies': results['std_accuracies']
        }
        for name, results in mnist_results.items()
    }

    plot_save_path = f"{config.RESULTS_DIR}/figures/robustness_curves_MNIST.png"
    plot_robustness_curves(
        mnist_plot_data,
        save_path=plot_save_path,
        title="Neuron Damage Robustness - MNIST"
    )

    # =========================================================================
    # CIFAR-10 Experiments
    # =========================================================================
    cifar_configs = [
        (RealCNN, config.CNN_CONFIG['real'], 'RealCNN'),
        (ComplexCNN, config.CNN_CONFIG['complex'], 'ComplexCNN')
    ]

    cifar_results = run_experiment_for_dataset('CIFAR10', cifar_configs)
    all_results.update(cifar_results)

    # Plot CIFAR-10 robustness curves
    cifar_plot_data = {
        name: {
            'damage_rates': results['damage_rates'],
            'mean_accuracies': results['mean_accuracies'],
            'std_accuracies': results['std_accuracies']
        }
        for name, results in cifar_results.items()
    }

    plot_save_path = f"{config.RESULTS_DIR}/figures/robustness_curves_CIFAR10.png"
    plot_robustness_curves(
        cifar_plot_data,
        save_path=plot_save_path,
        title="Neuron Damage Robustness - CIFAR-10"
    )

    # =========================================================================
    # Generate Summary Table
    # =========================================================================
    print("\n" + "="*70)
    print("DAMAGE ROBUSTNESS SUMMARY")
    print("="*70 + "\n")

    print("Accuracy at Different Damage Rates:\n")
    print(f"{'Model':<15} {'0%':<10} {'30%':<10} {'50%':<10} {'70%':<10} {'90%':<10}")
    print("-" * 65)

    for model_name, results in all_results.items():
        damage_rates = results['damage_rates']
        mean_accs = results['mean_accuracies']

        # Find accuracies at specific damage rates
        acc_0 = mean_accs[damage_rates.index(0.0)] if 0.0 in damage_rates else '-'
        acc_30 = mean_accs[damage_rates.index(0.3)] if 0.3 in damage_rates else '-'
        acc_50 = mean_accs[damage_rates.index(0.5)] if 0.5 in damage_rates else '-'
        acc_70 = mean_accs[damage_rates.index(0.7)] if 0.7 in damage_rates else '-'
        acc_90 = mean_accs[damage_rates.index(0.9)] if 0.9 in damage_rates else '-'

        print(f"{model_name:<15} {acc_0:<10.2f} {acc_30:<10.2f} {acc_50:<10.2f} {acc_70:<10.2f} {acc_90:<10.2f}")

    print("\n" + "="*70 + "\n")

    # Save results as JSON
    json_path = f"{config.RESULTS_DIR}/damage_robustness_results.json"

    # Convert numpy types to Python types for JSON serialization
    json_results = {}
    for name, results in all_results.items():
        json_results[name] = {
            'damage_rates': [float(x) for x in results['damage_rates']],
            'mean_accuracies': [float(x) for x in results['mean_accuracies']],
            'std_accuracies': [float(x) for x in results['std_accuracies']],
            'baseline_accuracy': float(results['baseline_accuracy'])
        }

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - Figures: {config.RESULTS_DIR}/figures/robustness_curves_*.png")

    print("\n" + "="*70)
    print("EXPERIMENT 2 COMPLETE!")
    print("="*70 + "\n")

    # =========================================================================
    # Analysis: Compare Complex vs Real
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS: Complex vs Real Networks")
    print("="*70 + "\n")

    # MNIST comparison
    if 'RealMLP' in all_results and 'ComplexMLP' in all_results:
        print("MNIST (RealMLP vs ComplexMLP):")
        real_50 = all_results['RealMLP']['mean_accuracies'][all_results['RealMLP']['damage_rates'].index(0.5)]
        complex_50 = all_results['ComplexMLP']['mean_accuracies'][all_results['ComplexMLP']['damage_rates'].index(0.5)]
        diff = complex_50 - real_50
        print(f"  At 50% damage:")
        print(f"    Real:    {real_50:.2f}%")
        print(f"    Complex: {complex_50:.2f}%")
        print(f"    Difference: {diff:+.2f}% {'(Complex better!)' if diff > 0 else '(Real better!)'}")

    # CIFAR-10 comparison
    if 'RealCNN' in all_results and 'ComplexCNN' in all_results:
        print("\nCIFAR-10 (RealCNN vs ComplexCNN):")
        real_50 = all_results['RealCNN']['mean_accuracies'][all_results['RealCNN']['damage_rates'].index(0.5)]
        complex_50 = all_results['ComplexCNN']['mean_accuracies'][all_results['ComplexCNN']['damage_rates'].index(0.5)]
        diff = complex_50 - real_50
        print(f"  At 50% damage:")
        print(f"    Real:    {real_50:.2f}%")
        print(f"    Complex: {complex_50:.2f}%")
        print(f"    Difference: {diff:+.2f}% {'(Complex better!)' if diff > 0 else '(Real better!)'}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
