"""Fault-robustness suite.

Every sampler has its own seeded generator keyed by (family, grid point, draw, layer), so all models
see the same masks and noise draws (common random numbers) and global RNG state is never touched.
Retained accuracy = mean over the fault grid of (accuracy under fault / clean accuracy).
"""
import numpy as np
import torch

from . import faults as F
from .nets import natural_granularity

DEATH_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEATH_MASKS = 10
NOISE_SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
NOISE_DRAWS = 5
WQ_BITS = [8, 6, 5, 4, 3, 2]
AQ_BITS = [8, 6, 4, 3, 2]
MAIN = ("death", "wnoise", "wquant", "aquant")
DEFAULT_FAMILIES = ("death", "death_unit", "death_rot", "death_rot_pruned", "wnoise", "wquant", "wquant_rot",
                    "aquant", "aquant_rot", "aquant_rot_pruned")


def gen(*key):
    s = 0
    for k in key:
        s = (s * 1_000_003 + int(k)) % (2 ** 61 - 1)
    return torch.Generator().manual_seed(s)


@torch.no_grad()
def accuracy(net, X, Y, site_fn=None, bs=5000):
    net.eval()
    hits = 0
    for s in range(0, len(X), bs):
        hits += (net(X[s:s + bs], site_fn=site_fn).argmax(1) == Y[s:s + bs]).sum().item()
    return 100.0 * hits / len(X)


def rotations(net):
    return [F.randomized_hadamard(w, gen(77, i)) for i, w in enumerate(net.widths)]


def death_curve(net, X, Y, granularity, rots=None, fam=1, alive=None):
    curve = []
    for ri, rate in enumerate(DEATH_RATES):
        accs = []
        for k in range(DEATH_MASKS):
            masks = [F.death_mask(w, rate, granularity, gen(fam, ri, k, i)) for i, w in enumerate(net.widths)]
            accs.append(accuracy(net, X, Y, F.death_site_fn(masks, rots, alive)))
        curve.append(float(np.mean(accs)))
    return curve


def noise_curve(net, X, Y):
    curve = []
    for si, sigma in enumerate(NOISE_SIGMAS):
        accs = [accuracy(F.with_weights(net, F.mult_noise(sigma, gen(3, si, k))), X, Y) for k in range(NOISE_DRAWS)]
        curve.append(float(np.mean(accs)))
    return curve


def wquant_curve(net, X, Y, rotated=False):
    return [accuracy(F.with_weights(net, F.weight_quant_rotated(b, gen(4, b)) if rotated else F.weight_quant(b)),
                     X, Y) for b in WQ_BITS]


def aquant_curve(net, X, Y, Xcal, rotated=False, alive=None):
    rots = rotations(net) if rotated else None
    ranges = F.calibrate_ranges(net, Xcal, rots)
    return [accuracy(net, X, Y, F.act_quant_site_fn(ranges, b, rots, alive)) for b in AQ_BITS]


def suite(net, data, families=DEFAULT_FAMILIES):
    X, Y = data["Xte"], data["Yte"]
    clean = accuracy(net, X, Y)
    nat = natural_granularity(net)
    Xcal = data["Xtr"][:2000]
    alive = F.alive_masks(net, Xcal)
    runners = {
        "death": lambda: death_curve(net, X, Y, nat),
        "death_unit": lambda: death_curve(net, X, Y, "unit", fam=2),
        "death_rot": lambda: death_curve(net, X, Y, nat, rotations(net)),
        "death_rot_pruned": lambda: death_curve(net, X, Y, nat, rotations(net), alive=alive),
        "wnoise": lambda: noise_curve(net, X, Y),
        "wquant": lambda: wquant_curve(net, X, Y),
        "wquant_rot": lambda: wquant_curve(net, X, Y, rotated=True),
        "aquant": lambda: aquant_curve(net, X, Y, Xcal),
        "aquant_rot": lambda: aquant_curve(net, X, Y, Xcal, rotated=True),
        "aquant_rot_pruned": lambda: aquant_curve(net, X, Y, Xcal, rotated=True, alive=alive),
    }
    curves = {f: runners[f]() for f in families}
    retained = {f: float(np.mean(c)) / clean for f, c in curves.items()}
    R = float(np.mean([retained[f] for f in MAIN if f in retained])) if any(f in retained for f in MAIN) else None
    return dict(clean=clean, curves=curves, retained=retained, R=R)
