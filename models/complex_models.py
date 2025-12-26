"""
Complex-valued neural network models
These are tested against real-valued baselines for robustness comparison
"""

import torch
import torch.nn as nn
from .complex_layers import (ComplexLinear, ComplexConv2d, complex_relu,
                              ComplexBatchNorm1d, ComplexBatchNorm2d,
                              ComplexMaxPool2d, complex_flatten, complex_modulus)


class ComplexMLP(nn.Module):
    """
    Complex-valued Multi-Layer Perceptron for MNIST

    Architecture:
    - Input: 784 (28x28 flattened) -> converted to complex
    - Hidden: [32, 32] complex neurons
    - Output: 10 classes

    Parameter count should match RealMLP with [64, 64] neurons
    since 1 complex neuron = 2 real parameters (real + imaginary)
    """

    def __init__(self, input_size=784, hidden_sizes=[32, 32], output_size=10):
        super(ComplexMLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build layers
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(ComplexLinear(prev_size, hidden_size))
            layers.append(ComplexBatchNorm1d(hidden_size))
            prev_size = hidden_size

        self.hidden_layers = nn.ModuleList(layers)

        # Output layer: complex -> real logits
        # We use the magnitude of complex output as logits
        self.output_layer = ComplexLinear(prev_size, output_size)

    def forward(self, x):
        """
        Args:
            x: Real input tensor of shape (batch, 1, 28, 28) or (batch, 784)

        Returns:
            Real logits of shape (batch, output_size)
        """
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Convert real input to complex (real part = input, imaginary = 0)
        z = torch.complex(x, torch.zeros_like(x))

        # Pass through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            z = layer(z)
            # Apply complex ReLU after Linear layers (not after BatchNorm)
            if isinstance(layer, ComplexLinear):
                z = complex_relu(z)

        # Output layer
        z = self.output_layer(z)

        # Convert complex output to real logits using magnitude
        # This is a common approach: |z| = sqrt(real^2 + imag^2)
        logits = complex_modulus(z)

        return logits

    def get_layer_activations(self, x, layer_idx):
        """
        Get activations from a specific hidden layer
        Used for layer-wise damage analysis

        Args:
            x: Input tensor
            layer_idx: Which hidden layer (0, 1, ...)

        Returns:
            Complex activations after the specified layer
        """
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Convert to complex
        z = torch.complex(x, torch.zeros_like(x))

        # Pass through layers
        current_layer = -1
        for i, layer in enumerate(self.hidden_layers):
            z = layer(z)
            if isinstance(layer, ComplexLinear):
                z = complex_relu(z)
                current_layer += 1
                if current_layer == layer_idx:
                    return z

        return z


class ComplexCNN(nn.Module):
    """
    Complex-valued Convolutional Neural Network for CIFAR-10

    Architecture:
    - Conv1: 3 -> 16 complex channels, 3x3 kernel
    - MaxPool: 2x2
    - Conv2: 16 -> 32 complex channels, 3x3 kernel
    - MaxPool: 2x2
    - FC: -> 64 complex -> 10

    Parameter count should match RealCNN with [32, 64] channels and 128 FC units
    """

    def __init__(self, conv_channels=[16, 32], fc_size=64, output_size=10):
        super(ComplexCNN, self).__init__()

        self.conv_channels = conv_channels
        self.fc_size = fc_size
        self.output_size = output_size

        # Convolutional layers
        # Input needs to be converted to complex, so input is treated as having 3 channels
        self.conv1 = ComplexConv2d(3, conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(conv_channels[0])

        self.conv2 = ComplexConv2d(conv_channels[0], conv_channels[1],
                                     kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(conv_channels[1])

        # Use standard max pooling on magnitude
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate size after convolutions
        # CIFAR-10: 32x32 -> conv1 -> 32x32 -> pool -> 16x16
        #        -> conv2 -> 16x16 -> pool -> 8x8
        self.fc_input_size = conv_channels[1] * 8 * 8

        # Fully connected layers
        self.fc1 = ComplexLinear(self.fc_input_size, fc_size)
        self.bn3 = ComplexBatchNorm1d(fc_size)
        self.fc2 = ComplexLinear(fc_size, output_size)

    def forward(self, x):
        """
        Args:
            x: Real input tensor of shape (batch, 3, 32, 32)

        Returns:
            Real logits of shape (batch, output_size)
        """
        # Convert real input to complex
        z = torch.complex(x, torch.zeros_like(x))

        # Conv block 1
        z = self.conv1(z)
        z = self.bn1(z)
        z = complex_relu(z)
        # Pool based on magnitude
        z_pooled = self._complex_maxpool(z)

        # Conv block 2
        z = self.conv2(z_pooled)
        z = self.bn2(z)
        z = complex_relu(z)
        z_pooled = self._complex_maxpool(z)

        # Flatten
        z = z_pooled.view(z_pooled.size(0), -1)

        # FC layers
        z = self.fc1(z)
        z = self.bn3(z)
        z = complex_relu(z)
        z = self.fc2(z)

        # Convert to real logits using magnitude
        logits = complex_modulus(z)

        return logits

    def _complex_maxpool(self, z):
        """
        Max pooling based on complex magnitude
        For simplicity, we pool real and imaginary parts separately
        """
        real_pooled = self.pool(z.real)
        imag_pooled = self.pool(z.imag)
        return torch.complex(real_pooled, imag_pooled)

    def get_layer_activations(self, x, layer_idx):
        """
        Get activations from a specific layer

        Args:
            x: Input tensor
            layer_idx: 0 = after conv1, 1 = after conv2, 2 = after fc1

        Returns:
            Complex activations after the specified layer
        """
        # Convert to complex
        z = torch.complex(x, torch.zeros_like(x))

        # Conv block 1
        z = self.conv1(z)
        z = self.bn1(z)
        z = complex_relu(z)
        z = self._complex_maxpool(z)

        if layer_idx == 0:
            return z

        # Conv block 2
        z = self.conv2(z)
        z = self.bn2(z)
        z = complex_relu(z)
        z = self._complex_maxpool(z)

        if layer_idx == 1:
            return z

        # Flatten
        z = z.view(z.size(0), -1)

        # FC layer
        z = self.fc1(z)
        z = self.bn3(z)
        z = complex_relu(z)

        if layer_idx == 2:
            return z

        return z


def count_complex_parameters(model):
    """
    Count parameters in a complex model

    Complex parameters are counted as 2x real parameters
    (one for real part, one for imaginary part)

    Args:
        model: Complex PyTorch model

    Returns:
        Effective number of real-valued parameters
    """
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.is_complex():
                # Complex parameter = 2 real parameters
                total += p.numel() * 2
            else:
                total += p.numel()
    return total
