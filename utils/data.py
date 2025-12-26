"""
Data loading utilities for MNIST and CIFAR-10
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


def get_dataloaders(dataset_name='MNIST', batch_size=128, num_workers=4,
                    data_dir='./data', val_split=0.1, seed=42):
    """
    Get train, validation, and test dataloaders for specified dataset

    Args:
        dataset_name: 'MNIST' or 'CIFAR10'
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory to store/load datasets
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name.upper() == 'MNIST':
        return _get_mnist_loaders(batch_size, num_workers, data_dir, val_split, seed)
    elif dataset_name.upper() == 'CIFAR10':
        return _get_cifar10_loaders(batch_size, num_workers, data_dir, val_split, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _get_mnist_loaders(batch_size, num_workers, data_dir, val_split, seed):
    """Get MNIST dataloaders"""

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Split training into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def _get_cifar10_loaders(batch_size, num_workers, data_dir, val_split, seed):
    """Get CIFAR-10 dataloaders"""

    # Transformations for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                            (0.2023, 0.1994, 0.2010))   # CIFAR-10 std
    ])

    # Transformations for validation/test (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    train_dataset_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Split training into train and validation
    val_size = int(len(train_dataset_full) * val_split)
    train_size = len(train_dataset_full) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset_temp = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=generator
    )

    # Create validation dataset with test transforms (no augmentation)
    val_dataset_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=test_transform
    )

    # Use the same indices for validation as before
    val_indices = val_dataset_temp.indices
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_dataset_info(dataset_name):
    """
    Get information about a dataset

    Args:
        dataset_name: 'MNIST' or 'CIFAR10'

    Returns:
        Dictionary with dataset information
    """
    info = {
        'MNIST': {
            'input_shape': (1, 28, 28),
            'num_classes': 10,
            'train_size': 60000,
            'test_size': 10000,
            'mean': (0.1307,),
            'std': (0.3081,)
        },
        'CIFAR10': {
            'input_shape': (3, 32, 32),
            'num_classes': 10,
            'train_size': 50000,
            'test_size': 10000,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010)
        }
    }

    return info.get(dataset_name.upper(), None)
