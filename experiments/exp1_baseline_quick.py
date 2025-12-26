"""
Quick Test Version of Experiment 1

This is a shortened version for testing:
- Only 2 epochs per model
- Only 1 trial instead of 5
- Smaller batch size option

Use this to verify everything works before running the full experiment!
"""

import sys
sys.path.append('.')

# Temporarily modify config for quick testing
import config
config.EPOCHS = 2  # Quick test: only 2 epochs
config.NUM_TRIALS = 1  # Only 1 trial
config.PATIENCE = 50  # Disable early stopping for test

# Import the main experiment
from experiments.exp1_baseline import main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUICK TEST MODE - Experiment 1")
    print("="*70)
    print("\nSettings modified for quick testing:")
    print(f"  - Epochs: {config.EPOCHS} (instead of 50)")
    print(f"  - Trials: {config.NUM_TRIALS} (instead of 5)")
    print("  - This will take ~10-15 minutes instead of 2-3 hours")
    print("="*70 + "\n")

    main()
