"""
Quick Test Version of Experiment 2

This is a shortened version for testing:
- Only test 3 damage rates: [0%, 50%, 90%]
- Only 3 trials per damage rate (instead of 10)

Use this to verify damage robustness testing works before running full experiment!
"""

import sys
sys.path.append('.')

# Temporarily modify config for quick testing
import config
config.DAMAGE_RATES = [0.0, 0.5, 0.9]  # Only 3 damage rates
config.DAMAGE_TRIALS_PER_RATE = 3  # Only 3 trials instead of 10

# Import the main experiment
from experiments.exp2_damage import main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUICK TEST MODE - Experiment 2")
    print("="*70)
    print("\nSettings modified for quick testing:")
    print(f"  - Damage rates: {[f'{int(d*100)}%' for d in config.DAMAGE_RATES]} (instead of 0-90%)")
    print(f"  - Trials per rate: {config.DAMAGE_TRIALS_PER_RATE} (instead of 10)")
    print("  - This will take ~5-10 minutes instead of 30-60 minutes")
    print("="*70 + "\n")

    main()
