"""
Quick test to verify installation and model creation
"""

import torch
print(f"✓ PyTorch {torch.__version__} imported successfully")

from models import RealMLP, ComplexMLP, RealCNN, ComplexCNN
print("✓ Models imported successfully")

from utils.metrics import count_parameters
from utils.data import get_dataset_info
print("✓ Utilities imported successfully")

# Test model creation
print("\nTesting model creation...")

real_mlp = RealMLP(input_size=784, hidden_sizes=[64, 64], output_size=10)
complex_mlp = ComplexMLP(input_size=784, hidden_sizes=[32, 32], output_size=10)

print(f"✓ RealMLP created: {count_parameters(real_mlp):,} parameters")
print(f"✓ ComplexMLP created: {count_parameters(complex_mlp):,} parameters")

real_cnn = RealCNN(conv_channels=[32, 64], fc_size=128, output_size=10)
complex_cnn = ComplexCNN(conv_channels=[16, 32], fc_size=64, output_size=10)

print(f"✓ RealCNN created: {count_parameters(real_cnn):,} parameters")
print(f"✓ ComplexCNN created: {count_parameters(complex_cnn):,} parameters")

# Test forward pass
print("\nTesting forward pass...")
batch_size = 4

# MLP test
x_mnist = torch.randn(batch_size, 1, 28, 28)
out_real = real_mlp(x_mnist)
out_complex = complex_mlp(x_mnist)

print(f"✓ RealMLP forward pass: input {x_mnist.shape} -> output {out_real.shape}")
print(f"✓ ComplexMLP forward pass: input {x_mnist.shape} -> output {out_complex.shape}")

# CNN test
x_cifar = torch.randn(batch_size, 3, 32, 32)
out_real_cnn = real_cnn(x_cifar)
out_complex_cnn = complex_cnn(x_cifar)

print(f"✓ RealCNN forward pass: input {x_cifar.shape} -> output {out_real_cnn.shape}")
print(f"✓ ComplexCNN forward pass: input {x_cifar.shape} -> output {out_complex_cnn.shape}")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nSystem is ready for experiments.")
