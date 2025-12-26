"""
Real-valued neural network models
These serve as baselines to compare against complex-valued models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RealMLP(nn.Module):
    """
    Real-valued Multi-Layer Perceptron for MNIST

    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden: [64, 64] real neurons
    - Output: 10 classes

    This should have approximately the same number of parameters as ComplexMLP
    with [32, 32] complex neurons (since 1 complex = 2 real parameters)
    """

    def __init__(self, input_size=784, hidden_sizes=[64, 64], output_size=10):
        super(RealMLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build layers
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size

        # Output layer (no activation, will use cross-entropy loss)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784)

        Returns:
            Logits of shape (batch, output_size)
        """
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        return self.network(x)

    def get_layer_activations(self, x, layer_idx):
        """
        Get activations from a specific hidden layer
        Used for layer-wise damage analysis

        Args:
            x: Input tensor
            layer_idx: Which hidden layer (0, 1, ...)

        Returns:
            Activations before the next layer
        """
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Pass through layers up to the specified layer
        for i, module in enumerate(self.network):
            x = module(x)
            # Check if we've passed the desired layer
            # Each hidden layer = Linear + ReLU + BatchNorm = 3 modules
            if isinstance(module, nn.BatchNorm1d):
                current_layer = i // 3
                if current_layer == layer_idx:
                    return x

        return x


class RealCNN(nn.Module):
    """
    Real-valued Convolutional Neural Network for CIFAR-10

    Architecture:
    - Conv1: 3 -> 32 channels, 3x3 kernel
    - MaxPool: 2x2
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - MaxPool: 2x2
    - FC: -> 128 -> 10

    This should have approximately the same parameters as ComplexCNN
    with [16, 32] complex channels
    """

    def __init__(self, conv_channels=[32, 64], fc_size=128, output_size=10):
        super(RealCNN, self).__init__()

        self.conv_channels = conv_channels
        self.fc_size = fc_size
        self.output_size = output_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])

        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1],
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])

        self.pool = nn.MaxPool2d(2, 2)

        # Calculate size after convolutions
        # CIFAR-10: 32x32 -> conv1 -> 32x32 -> pool -> 16x16
        #        -> conv2 -> 16x16 -> pool -> 8x8
        self.fc_input_size = conv_channels[1] * 8 * 8

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, fc_size)
        self.bn3 = nn.BatchNorm1d(fc_size)
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, 3, 32, 32)

        Returns:
            Logits of shape (batch, output_size)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

    def get_layer_activations(self, x, layer_idx):
        """
        Get activations from a specific layer

        Args:
            x: Input tensor
            layer_idx: 0 = after conv1, 1 = after conv2, 2 = after fc1

        Returns:
            Activations after the specified layer
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        if layer_idx == 0:
            return x

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        if layer_idx == 1:
            return x

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layer
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)

        if layer_idx == 2:
            return x

        return x


def count_parameters(model):
    """
    Count the number of trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
