"""
Visualization utilities for plotting results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_robustness_curves(results_dict, save_path=None, title="Neuron Damage Robustness"):
    """
    Plot accuracy vs damage rate curves for multiple models

    Args:
        results_dict: Dictionary mapping model names to results
                     Each result should have 'damage_rates', 'mean_accuracies', 'std_accuracies'
        save_path: Path to save figure (optional)
        title: Plot title

    Example:
        results_dict = {
            'RealMLP': {
                'damage_rates': [0.0, 0.1, 0.2, ...],
                'mean_accuracies': [97.2, 95.1, 92.3, ...],
                'std_accuracies': [0.3, 0.4, 0.5, ...]
            },
            'ComplexMLP': { ... }
        }
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(results_dict))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, (model_name, results) in enumerate(results_dict.items()):
        damage_rates = np.array(results['damage_rates']) * 100  # Convert to percentage
        mean_acc = np.array(results['mean_accuracies'])
        std_acc = np.array(results['std_accuracies'])

        # Plot mean with error bars
        ax.plot(damage_rates, mean_acc, marker=markers[idx % len(markers)],
                label=model_name, color=colors[idx], linewidth=2, markersize=8)

        # Add confidence interval (mean ± std)
        ax.fill_between(damage_rates, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.2, color=colors[idx])

    ax.set_xlabel('Damage Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show all damage rates
    ax.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def plot_parameter_comparison(param_dict, save_path=None):
    """
    Plot bar chart comparing parameter counts

    Args:
        param_dict: Dictionary mapping model names to parameter counts
        save_path: Path to save figure (optional)

    Example:
        param_dict = {
            'RealMLP': 50890,
            'ComplexMLP': 50912,
            'RealCNN': 62410,
            'ComplexCNN': 62408
        }
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = list(param_dict.keys())
    params = list(param_dict.values())

    colors = ['#3498db' if 'Real' in m else '#e74c3c' for m in models]

    bars = ax.bar(models, params, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Count Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def plot_layer_analysis(results_dict, save_path=None):
    """
    Plot layer-wise damage analysis

    Args:
        results_dict: Dictionary mapping model names to layer-wise results
        save_path: Path to save figure (optional)

    Example:
        results_dict = {
            'RealMLP': {
                'layers': ['Layer 1', 'Layer 2', 'All Layers'],
                'accuracies': [85.2, 78.4, 65.3],
                'stds': [1.2, 1.5, 1.8]
            },
            'ComplexMLP': { ... }
        }
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(list(results_dict.values())[0]['layers']))
    width = 0.35
    colors = ['#3498db', '#e74c3c']

    for idx, (model_name, results) in enumerate(results_dict.items()):
        offset = width * (idx - 0.5)
        layers = results['layers']
        accuracies = results['accuracies']
        stds = results['stds']

        bars = ax.bar(x + offset, accuracies, width, label=model_name,
                     color=colors[idx], alpha=0.7, yerr=stds, capsize=5,
                     edgecolor='black')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Damaged Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Damage Analysis (50% Damage Rate)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def plot_training_curves(train_history, val_history, save_path=None):
    """
    Plot training and validation curves

    Args:
        train_history: Dictionary with 'loss' and 'accuracy' lists
        val_history: Dictionary with 'loss' and 'accuracy' lists
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_history['loss']) + 1)

    # Loss plot
    ax1.plot(epochs, train_history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_history['loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_history['accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def create_results_table(results_dict, save_path=None):
    """
    Create and save a results table

    Args:
        results_dict: Dictionary with results
        save_path: Path to save table (CSV or Markdown)

    Example:
        results_dict = {
            'Model': ['RealMLP', 'ComplexMLP', 'RealCNN', 'ComplexCNN'],
            'Parameters': [50890, 50912, 62410, 62408],
            'MNIST Acc (%)': [97.2, 97.1, 99.1, 99.0],
            'MNIST Std': [0.3, 0.2, 0.1, 0.1],
            'CIFAR-10 Acc (%)': [52.1, 51.8, 68.3, 67.9],
            'CIFAR-10 Std': [0.8, 0.7, 0.5, 0.6]
        }
    """
    df = pd.DataFrame(results_dict)

    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if save_path.endswith('.csv'):
            df.to_csv(save_path, index=False)
            print(f"Table saved to {save_path}")
        elif save_path.endswith('.md'):
            with open(save_path, 'w') as f:
                f.write(df.to_markdown(index=False))
            print(f"Table saved to {save_path}")
        else:
            # Default to CSV
            csv_path = save_path + '.csv'
            df.to_csv(csv_path, index=False)
            print(f"Table saved to {csv_path}")

    return df
