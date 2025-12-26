"""
Configuration file for Complex-Valued Neural Networks Robustness Study
All hyperparameters, random seeds, and experiment settings
"""

import torch

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42
NUM_TRIALS = 5  # Number of repeated experiments for statistical validation

# =============================================================================
# DATASET SETTINGS
# =============================================================================
DATASETS = ['MNIST', 'CIFAR10']
DATA_DIR = './data'
BATCH_SIZE = 128
NUM_WORKERS = 4

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
OPTIMIZER = 'Adam'

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.001

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

# MLP configurations (for MNIST)
MLP_CONFIG = {
    'real': {
        'input_size': 784,
        'hidden_sizes': [64, 64],
        'output_size': 10,
        'activation': 'relu'
    },
    'complex': {
        'input_size': 784,
        'hidden_sizes': [32, 32],  # 32 complex = 64 real params
        'output_size': 10,
        'activation': 'complex_relu'
    }
}

# CNN configurations (for CIFAR-10)
CNN_CONFIG = {
    'real': {
        'conv_channels': [32, 64],  # Conv layers
        'kernel_size': 3,
        'fc_size': 128,
        'output_size': 10,
        'activation': 'relu'
    },
    'complex': {
        'conv_channels': [16, 32],  # 16 complex = 32 real params
        'kernel_size': 3,
        'fc_size': 128,  # 128 complex neurons (256 real params) to match Real FC: 4096→128
        'output_size': 10,
        'activation': 'complex_relu'
    }
}

# =============================================================================
# DAMAGE ROBUSTNESS SETTINGS
# =============================================================================

# Damage rates to test (percentage of neurons to remove)
DAMAGE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Number of trials per damage rate (for statistical significance)
DAMAGE_TRIALS_PER_RATE = 10

# Layer-wise damage settings
LAYER_DAMAGE_RATE = 0.5  # 50% damage for layer analysis

# =============================================================================
# DEVICE SETTINGS
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# LOGGING AND SAVING
# =============================================================================
SAVE_MODELS = True
MODEL_SAVE_DIR = './saved_models'
RESULTS_DIR = './results'
LOG_INTERVAL = 100  # Log every N batches

# Visualization settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# =============================================================================
# STATISTICAL TESTING
# =============================================================================
CONFIDENCE_LEVEL = 0.95  # For confidence intervals
SIGNIFICANCE_LEVEL = 0.05  # For hypothesis testing
