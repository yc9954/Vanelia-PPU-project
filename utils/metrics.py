"""
Evaluation metrics and neuron damage functions
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def count_parameters(model):
    """
    Count trainable parameters in a model
    For complex models, each complex parameter counts as 2 real parameters

    Args:
        model: PyTorch model

    Returns:
        Total number of effective real-valued parameters
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


def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluate model on a dataset

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on
        criterion: Loss function (optional)

    Returns:
        Dictionary with accuracy and optionally loss
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

    accuracy = 100.0 * correct / total

    result = {'accuracy': accuracy}
    if criterion is not None:
        result['loss'] = total_loss / num_batches

    return result


def apply_neuron_damage(model, damage_rate, seed=None, layer_idx=None):
    """
    Apply neuron damage to model by creating activation masks
    This returns a hook function that will mask neuron activations

    Args:
        model: PyTorch model
        damage_rate: Fraction of neurons to damage (0.0 to 1.0)
        seed: Random seed for reproducibility
        layer_idx: If specified, only damage this layer. Otherwise damage all layers.

    Returns:
        Dictionary mapping layer names to damage masks
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    masks = {}

    # Get all layers that have neurons to damage
    for name, module in model.named_modules():
        # Skip if we're only damaging a specific layer
        if layer_idx is not None:
            # Check if this is the target layer
            # For MLP: fc1, fc2, etc.
            # For CNN: conv1, conv2, fc1, fc2, etc.
            layer_num = _extract_layer_number(name)
            if layer_num != layer_idx:
                continue

        # Create masks for Linear, Conv2d, and Complex layers
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # For regular layers, mask output channels/neurons
            if isinstance(module, nn.Linear):
                num_neurons = module.out_features
                mask = (torch.rand(num_neurons) > damage_rate).float()
            else:  # Conv2d
                num_channels = module.out_channels
                mask = (torch.rand(num_channels) > damage_rate).float()

            masks[name] = mask.to(next(model.parameters()).device)

        # Handle complex layers
        elif module.__class__.__name__ in ['ComplexLinear', 'ComplexConv2d']:
            # For complex layers, mask complex neurons
            if 'Linear' in module.__class__.__name__:
                # ComplexLinear has fc_r and fc_i
                num_neurons = module.fc_r.out_features
                mask = (torch.rand(num_neurons) > damage_rate).float()
            else:  # ComplexConv2d
                num_channels = module.conv_r.out_channels
                mask = (torch.rand(num_channels) > damage_rate).float()

            masks[name] = mask.to(next(model.parameters()).device)

    return masks


def _extract_layer_number(layer_name):
    """
    Extract layer number from layer name
    e.g., 'fc1' -> 0, 'fc2' -> 1, 'conv1' -> 0, 'conv2' -> 1
    """
    # Try to extract number from name
    import re
    match = re.search(r'\d+', layer_name)
    if match:
        return int(match.group()) - 1  # Convert to 0-indexed
    return None


class DamageHook:
    """
    Hook to apply damage masks to layer activations during forward pass
    """
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, module, input, output):
        """
        Apply mask to output activations

        For Linear layers: output shape is (batch, neurons)
        For Conv2d layers: output shape is (batch, channels, height, width)
        For Complex layers: output is complex tensor
        """
        if output.is_complex():
            # For complex outputs, apply mask to both real and imaginary
            if len(output.shape) == 2:  # Linear layer
                # Shape: (batch, neurons)
                real_masked = output.real * self.mask.unsqueeze(0)
                imag_masked = output.imag * self.mask.unsqueeze(0)
            else:  # Conv layer
                # Shape: (batch, channels, height, width)
                mask_expanded = self.mask.view(1, -1, 1, 1)
                real_masked = output.real * mask_expanded
                imag_masked = output.imag * mask_expanded

            return torch.complex(real_masked, imag_masked)
        else:
            # For real outputs
            if len(output.shape) == 2:  # Linear layer
                return output * self.mask.unsqueeze(0)
            else:  # Conv layer
                # Shape: (batch, channels, height, width)
                mask_expanded = self.mask.view(1, -1, 1, 1)
                return output * mask_expanded


def evaluate_with_damage(model, dataloader, device, damage_rate, seed=None,
                          layer_idx=None, num_trials=1):
    """
    Evaluate model with neuron damage applied

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on
        damage_rate: Fraction of neurons to damage
        seed: Random seed (if None, uses different random mask each trial)
        layer_idx: Which layer to damage (None = all layers)
        num_trials: Number of trials with different random masks

    Returns:
        Dictionary with mean and std of accuracy across trials
    """
    accuracies = []

    for trial in range(num_trials):
        # Set seed for this trial
        trial_seed = seed if seed is not None else np.random.randint(0, 10000)
        if num_trials > 1:
            trial_seed = trial_seed + trial

        # Create damage masks
        masks = apply_neuron_damage(model, damage_rate, seed=trial_seed,
                                     layer_idx=layer_idx)

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in masks:
                hook = module.register_forward_hook(DamageHook(masks[name]))
                hooks.append(hook)

        # Evaluate
        result = evaluate_model(model, dataloader, device)
        accuracies.append(result['accuracy'])

        # Remove hooks
        for hook in hooks:
            hook.remove()

    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'accuracies': accuracies
    }


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train model for one epoch

    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on

    Returns:
        Dictionary with average loss and accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return {'loss': avg_loss, 'accuracy': accuracy}
