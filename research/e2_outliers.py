"""E2: when does TurboQuant-style rotation help? Controlled channel outliers (evaluation only).

A function-preserving rescaling multiplies k units per hidden site by s: the preceding layer's rows
and bias are scaled by s (ReLU / CReLU are positively homogeneous) and the BatchNorm running
statistics are rescaled (mean * s, var * s^2), or, without BatchNorm, the next layer's columns are
divided by s. The network computes the same function, but its activations now carry LLM-style
channel outliers. Faults are then applied with and without randomized-Hadamard rotation.

Theory (README): rotation makes a site fault isotropic. For activation quantization the plain error
is set by the per-tensor range (the outlier) but skips units that are exactly 0, so the predicted
error ratio rotated / plain is
    (range_rot^2 * ||M_alive||_F^2) / (range^2 * sum_j ||M_j||^2 P(u_j != 0)),
and rotation should win once outliers inflate `range`. Unit death is unaffected by the rescaling
without rotation, while rotated death leaks the outlier's energy into every unit.

    python research/e2_outliers.py            # writes runs/e2.jsonl
"""
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lab import coherence, data, faults as F, nets, suite  # noqa: E402
from lab.train import train  # noqa: E402

SCALES = [1, 4, 16, 64]
BITS = [8, 6, 4, 3]
CONFIGS = {
    "real_lin": dict(algebra="real", readout="linear"),
    "cplx_mod": dict(algebra="complex", readout="modulus"),
    "real_lin_nobn": dict(algebra="real", readout="linear", norm="none"),
}


@torch.no_grad()
def inject_outliers(net, s, k, g):
    """Function-preserving copy of net with k units (complex nets: k complex units) per site scaled by s."""
    net = copy.deepcopy(net)
    paired = net.algebra != "real"
    for i, lin in enumerate(net.lins):
        n = net.widths[i]
        units = torch.randperm(n // 2 if paired else n, generator=g)[:k]
        rows = torch.cat([units, units + n // 2]) if paired else units
        if paired:
            lin.A[units] *= s
            lin.B[units] *= s
        else:
            lin.W[units] *= s
        if lin.bias is not None:
            lin.bias[rows] *= s
        norm = net.norms[i]
        if isinstance(norm, torch.nn.BatchNorm1d):
            norm.running_mean[rows] *= s
            norm.running_var[rows] *= s * s
        else:  # divide the next layer's input columns by s
            nxt = net.lins[i + 1] if i + 1 < len(net.lins) else net.head
            if nxt.algebra == "real":
                nxt.W[:, rows] /= s
            else:
                nxt.A[:, units] /= s
                nxt.B[:, units] /= s
    return net


@torch.no_grad()
def predicted_aquant_ratio(net, Xcal, rots, alive, bits=4):
    """First-order rotated / plain activation-quantization error at each site (see module docstring)."""
    _, sites = net(Xcal, return_sites=True)
    out = []
    for i, U in enumerate(sites):
        M, _ = coherence.downstream_affine(net, i, U.shape[1])
        colsq = (M ** 2).sum(0)
        V = U @ rots[i].T
        plain = (U.max() - U.min()) ** 2 * (colsq * (U != 0).float().mean(0)).sum()
        rotated = (V.max() - V.min()) ** 2 * (colsq * alive[i]).sum()
        out.append((rotated / plain).item())
    return out


def main():
    d = data.load("mnist")
    torch.set_num_threads(int(os.environ.get("LAB_THREADS", "4")))
    X, Y, Xcal = d["Xte"], d["Yte"], d["Xtr"][:2000]
    out_path = os.path.join(HERE, "runs", "e2.jsonl")
    for name, cfg in CONFIGS.items():
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            base = nets.Net(784, [64, 64], 10, **cfg)
            train(base, d, epochs=5, seed=seed)
            base.eval()
            for s in SCALES:
                net = inject_outliers(base, s, k=1, g=torch.Generator().manual_seed(1000 + seed))
                with torch.no_grad():
                    drift = (net(X[:2000]) - base(X[:2000])).abs().max().item()
                rots = suite.rotations(net)
                alive = F.alive_masks(net, Xcal)
                rng = F.calibrate_ranges(net, Xcal)
                rng_rot = F.calibrate_ranges(net, Xcal, rots)
                clean = suite.accuracy(net, X, Y)
                rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), tag="E2", name=name, seed=seed, scale=s,
                           function_drift=drift, clean=clean,
                           aquant={b: suite.accuracy(net, X, Y, F.act_quant_site_fn(rng, b)) for b in BITS},
                           aquant_rot={b: suite.accuracy(net, X, Y, F.act_quant_site_fn(rng_rot, b, rots, alive))
                                       for b in BITS},
                           death=suite.death_curve(net, X, Y, nets.natural_granularity(net))[:5],
                           death_rot=suite.death_curve(net, X, Y, nets.natural_granularity(net), rots,
                                                       alive=alive)[:5],
                           pred_aquant_ratio=predicted_aquant_ratio(net, Xcal, rots, alive),
                           crest2=[coherence.crest2(u) for u in net(Xcal, return_sites=True)[1]])
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{name} seed {seed} s={s:>2}: drift {drift:.1e} clean {clean:.2f} | "
                      f"aquant 4b {rec['aquant'][4]:.1f} rot {rec['aquant_rot'][4]:.1f} | "
                      f"3b {rec['aquant'][3]:.1f} rot {rec['aquant_rot'][3]:.1f} | "
                      f"pred rot/plain {[round(r, 2) for r in rec['pred_aquant_ratio']]} | "
                      f"death30% {rec['death'][2]:.1f} rot {rec['death_rot'][2]:.1f}", flush=True)


if __name__ == "__main__":
    main()
