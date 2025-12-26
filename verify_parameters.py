"""
Verify that complex and real model pairs have equal parameter counts
This is critical for fair comparison
"""

import torch
import sys
sys.path.append('.')

from models import RealMLP, ComplexMLP, RealCNN, ComplexCNN
from utils.metrics import count_parameters
import config


def verify_model_pair(real_model, complex_model, model_name):
    """
    Verify that two models have similar parameter counts

    Args:
        real_model: Real-valued model
        complex_model: Complex-valued model
        model_name: Name for display

    Returns:
        True if parameters are within 5% of each other
    """
    real_params = count_parameters(real_model)
    complex_params = count_parameters(complex_model)

    diff = abs(real_params - complex_params)
    diff_pct = 100.0 * diff / real_params

    print(f"\n{'='*70}")
    print(f"{model_name} Parameter Comparison")
    print(f"{'='*70}")
    print(f"Real Model:    {real_params:,} parameters")
    print(f"Complex Model: {complex_params:,} parameters")
    print(f"Difference:    {diff:,} parameters ({diff_pct:.2f}%)")

    # Allow up to 5% difference (due to discrete neuron counts)
    if diff_pct <= 5.0:
        print(f"✓ PASS: Parameter counts are within 5%")
        print(f"{'='*70}\n")
        return True
    else:
        print(f"✗ FAIL: Parameter counts differ by more than 5%")
        print(f"{'='*70}\n")
        return False


def print_layer_breakdown(model, model_name):
    """
    Print detailed parameter breakdown by layer

    Args:
        model: PyTorch model
        model_name: Name for display
    """
    print(f"\n{model_name} - Layer-wise Parameter Count:")
    print(f"{'-'*70}")
    print(f"{'Layer Name':<30} {'Parameters':<15} {'Shape':<20}")
    print(f"{'-'*70}")

    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Count complex parameters as 2x
            if param.is_complex():
                param_count = param.numel() * 2
                shape_str = f"{param.shape} (complex)"
            else:
                param_count = param.numel()
                shape_str = str(tuple(param.shape))

            print(f"{name:<30} {param_count:<15,} {shape_str:<20}")
            total += param_count

    print(f"{'-'*70}")
    print(f"{'TOTAL':<30} {total:<15,}")
    print(f"{'-'*70}\n")


def main():
    """Main verification script"""

    print("\n" + "="*70)
    print("PARAMETER COUNT VERIFICATION")
    print("="*70)
    print("\nThis script verifies that complex and real model pairs have")
    print("equal parameter counts for fair comparison.")
    print("\nPrinciple: 1 complex neuron = 2 real neurons")
    print("           (1 for real part + 1 for imaginary part)")
    print("="*70)

    all_passed = True

    # =========================================================================
    # MLP Models (MNIST)
    # =========================================================================
    print("\n\n### Testing MLP Models (for MNIST) ###\n")

    real_mlp = RealMLP(
        input_size=config.MLP_CONFIG['real']['input_size'],
        hidden_sizes=config.MLP_CONFIG['real']['hidden_sizes'],
        output_size=config.MLP_CONFIG['real']['output_size']
    )

    complex_mlp = ComplexMLP(
        input_size=config.MLP_CONFIG['complex']['input_size'],
        hidden_sizes=config.MLP_CONFIG['complex']['hidden_sizes'],
        output_size=config.MLP_CONFIG['complex']['output_size']
    )

    print_layer_breakdown(real_mlp, "RealMLP")
    print_layer_breakdown(complex_mlp, "ComplexMLP")

    passed = verify_model_pair(real_mlp, complex_mlp, "MLP")
    all_passed = all_passed and passed

    # =========================================================================
    # CNN Models (CIFAR-10)
    # =========================================================================
    print("\n\n### Testing CNN Models (for CIFAR-10) ###\n")

    real_cnn = RealCNN(
        conv_channels=config.CNN_CONFIG['real']['conv_channels'],
        fc_size=config.CNN_CONFIG['real']['fc_size'],
        output_size=config.CNN_CONFIG['real']['output_size']
    )

    complex_cnn = ComplexCNN(
        conv_channels=config.CNN_CONFIG['complex']['conv_channels'],
        fc_size=config.CNN_CONFIG['complex']['fc_size'],
        output_size=config.CNN_CONFIG['complex']['output_size']
    )

    print_layer_breakdown(real_cnn, "RealCNN")
    print_layer_breakdown(complex_cnn, "ComplexCNN")

    passed = verify_model_pair(real_cnn, complex_cnn, "CNN")
    all_passed = all_passed and passed

    # =========================================================================
    # Final Result
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL VERIFICATION RESULT")
    print("="*70)

    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nAll model pairs have approximately equal parameter counts.")
        print("Fair comparison is ensured!")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease adjust model architectures to match parameter counts.")
        print("Recommendation: Adjust hidden layer sizes in config.py")

    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
