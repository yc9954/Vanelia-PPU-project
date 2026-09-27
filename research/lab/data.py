"""Datasets as preloaded CPU tensors.

MNIST and Fashion-MNIST reuse the repository's split (utils/data.py): 90/10 train/validation via
random_split with seed 42, the official test set, and per-dataset mean/std normalisation. Images
are flattened to 784 features. Loading goes through the raw uint8 arrays instead of per-item
transforms, so an epoch of MLP training takes seconds on a CPU.
"""
import os

import torch
from torchvision import datasets

DATA_DIR = os.environ.get(
    "LAB_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
_STATS = {"mnist": (0.1307, 0.3081), "fashion": (0.2860, 0.3530)}
_CACHE = {}


def load(name):
    if name not in _CACHE:
        if name in _STATS:
            _CACHE[name] = _image_dataset(name)
        elif name.startswith("iq"):
            from .iq import iq_dataset
            _CACHE[name] = iq_dataset(name)
        else:
            raise ValueError(f"unknown dataset {name}")
    return _CACHE[name]


def _image_dataset(name):
    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[name]
    mean, std = _STATS[name]
    train, test = cls(DATA_DIR, train=True, download=True), cls(DATA_DIR, train=False, download=True)

    def prep(ds):
        return ((ds.data.float() / 255 - mean) / std).reshape(len(ds.data), -1), ds.targets.clone()

    X, Y = prep(train)
    Xte, Yte = prep(test)
    n_val = int(len(X) * 0.1)
    tr, va = torch.utils.data.random_split(range(len(X)), [len(X) - n_val, n_val],
                                           generator=torch.Generator().manual_seed(42))
    tr, va = torch.tensor(tr.indices), torch.tensor(va.indices)
    return dict(name=name, Xtr=X[tr], Ytr=Y[tr], Xva=X[va], Yva=Y[va], Xte=Xte, Yte=Yte,
                n_in=X.shape[1], n_classes=10, input_paired=False)
