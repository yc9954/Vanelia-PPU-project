"""Fault models. Every sampler takes an explicit torch.Generator; global RNG state is never touched.

Site faults act on hidden unit outputs through `Net.forward(site_fn=...)`:
  * unit death: exact-count masks at real-unit or pair (complex/split unit) granularity;
  * rotation coding: units are stored as v = R u (R a randomized Hadamard or Haar rotation), faults hit
    v, and u' = R^T v is decoded before the next layer; with no fault it is exact;
  * activation quantization: asymmetric per-tensor uniform with calibrated ranges, optionally in the
    rotated basis (the incoherence processing of QuaRot / TurboQuant).
Weight faults act on the stored parameters of every StructuredLinear (hidden layers and readout);
BatchNorm parameters and biases stay in full precision.
"""
import copy
import math

import torch

from .nets import StructuredLinear


# ---------------------------------------------------------------- rotations
def random_orthogonal(n, g):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()  # Haar distributed


def randomized_hadamard(n, g):
    """Sylvester Hadamard / sqrt(n) with random column signs; Haar-random when n is not a power of 2."""
    if n & (n - 1):
        return random_orthogonal(n, g)
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    signs = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return H * signs / math.sqrt(n)


# ---------------------------------------------------------------- unit death
def exact_keep_mask(n, rate, g):
    m = torch.ones(n)
    m[torch.randperm(n, generator=g)[:round(rate * n)]] = 0.0
    return m


def death_mask(width, rate, granularity, g):
    if granularity == "unit":
        return exact_keep_mask(width, rate, g)
    half = exact_keep_mask(width // 2, rate, g)  # pair (j, j + k) dies together
    return torch.cat([half, half])


def death_site_fn(masks, rotations=None, alive=None):
    """alive: optional per-site float masks; decoded outputs of units that never fire are forced back to 0
    (pruning dead units, exact on the calibration data)."""
    def fn(i, u):
        m = masks[i]
        if m is None:
            return u
        if rotations is None:
            return u * m
        R = rotations[i]
        out = ((u @ R.T) * m) @ R
        return out if alive is None else out * alive[i]
    return fn


@torch.no_grad()
def alive_masks(net, X):
    net.eval()
    _, sites = net(X, return_sites=True)
    return [(u.abs().amax(0) > 0).float() for u in sites]


# ---------------------------------------------------------------- activation quantization
def quantize_uniform(x, lo, hi, bits):
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp_min(1e-12) / levels
    return ((x - lo) / scale).round().clamp(0, levels) * scale + lo


@torch.no_grad()
def calibrate_ranges(net, X, rotations=None):
    net.eval()
    _, sites = net(X, return_sites=True)
    out = []
    for i, u in enumerate(sites):
        v = u if rotations is None else u @ rotations[i].T
        out.append((v.min(), v.max()))
    return out


def act_quant_site_fn(ranges, bits, rotations=None, alive=None):
    def fn(i, u):
        lo, hi = ranges[i]
        if rotations is None:
            return quantize_uniform(u, lo, hi, bits)
        R = rotations[i]
        out = quantize_uniform(u @ R.T, lo, hi, bits) @ R
        return out if alive is None else out * alive[i]
    return fn


# ---------------------------------------------------------------- weight faults
@torch.no_grad()
def with_weights(net, transform):
    """Deep copy of net with transform(layer, tensors) -> new tensors applied to every layer's stored weights."""
    clone = copy.deepcopy(net)
    for layer in clone.layers():
        tensors = layer.stored()
        for t, new in zip(tensors, transform(layer, [t.detach().clone() for t in tensors])):
            t.copy_(new)
    return clone


def mult_noise(sigma, g):
    return lambda layer, ts: [t * (1 + sigma * torch.randn(t.shape, generator=g)) for t in ts]


def quantize_rows(t, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = t.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    return (t / scale).round().clamp(-qmax, qmax) * scale


def weight_quant(bits):
    return lambda layer, ts: [quantize_rows(t, bits) for t in ts]


def weight_quant_rotated(bits, g):
    """Quantize W Q row-wise, then rotate back. One real rotation Q per layer acts on the input index of
    every stored tensor, which keeps complex/split structure: (A + iB) Q = AQ + i BQ."""
    def tf(layer, ts):
        Q = randomized_hadamard(ts[0].shape[1], g)
        return [quantize_rows(t @ Q, bits) @ Q.T for t in ts]
    return tf


def is_structured(module):
    return isinstance(module, StructuredLinear)
