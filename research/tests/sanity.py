"""Harness checks. Run: python research/tests/sanity.py (CPU, about a minute, downloads MNIST)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lab import coherence, data, faults, nets, suite  # noqa: E402
from lab.train import train  # noqa: E402


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        sys.exit(1)


torch.manual_seed(0)

# 1. complex structure == complex matrix multiplication
lin = nets.StructuredLinear(6, 4, "complex", paired_input=True, bias=False)
x = torch.randn(5, 6)
z = torch.complex(x[:, :3], x[:, 3:]) @ torch.complex(lin.A, lin.B).T
check("complex layer = complex matmul", torch.allclose(lin(x), torch.cat([z.real, z.imag], 1), atol=1e-6))
lin = nets.StructuredLinear(6, 4, "split", paired_input=True, bias=False)
xr, xi = x[:, :3], x[:, 3:]
ref = torch.cat([xr @ lin.A.T + xi @ lin.B.T, xr @ lin.B.T + xi @ lin.A.T], 1)
check("split-complex layer = (A + jB)(x + jy), j^2 = +1", torch.allclose(lin(x), ref, atol=1e-6))

# 2. same effective initial weight scale for every algebra
w = {a: nets.StructuredLinear(512, 512, a).weight().std().item() for a in nets.ALGEBRAS}
check("same effective init scale", max(w.values()) / min(w.values()) < 1.05, str({k: round(v, 4) for k, v in w.items()}))

# 3. quick training so BatchNorm statistics and weights are realistic
d = data.load("mnist")
models = {}
for alg, ro in (("real", "linear"), ("complex", "modulus")):
    torch.manual_seed(0)
    net = nets.Net(784, [64, 64], 10, algebra=alg, readout=ro)
    d1 = dict(d, Xtr=d["Xtr"][:12800], Ytr=d["Ytr"][:12800])
    val = train(net, d1, epochs=1, seed=0)
    models[alg] = net
    check(f"{alg}/{ro} trains", val > 85, f"val {val:.1f}%")

# 4. first-order theory: E||delta||^2 / E||S||^2 = q2 + (q1 - q2) rho for exact-count death (exact algebra)
X = d["Xte"][:1000]
for alg, net in models.items():
    net.eval()
    _, sites = net(X, return_sites=True)
    U = sites[0]
    n = U.shape[1]
    M, c = coherence.downstream_affine(net, 0, n)
    gran = nets.natural_granularity(net)
    st = coherence.site_stats(U, M, gran)
    N = n if gran == "unit" else n // 2
    for rate in (0.2, 0.5):
        k = round(rate * N)
        q1, q2 = k / N, k * (k - 1) / (N * (N - 1))
        pred = q2 + (q1 - q2) * st["rho"]
        errs = []
        for s in range(300):
            m = faults.death_mask(n, rate, gran, torch.Generator().manual_seed(s))
            errs.append((((U * (1 - m)) @ M.T) ** 2).sum().item())
        meas = np.mean(errs) / st["sig"]
        check(f"theory matches measured error ({alg}, rate {rate})", abs(meas - pred) / pred < 0.05,
              f"pred {pred:.4f} meas {meas:.4f} rho {st['rho']:.3f}")
    check(f"rho = kappa_conc * kappa_cancel ({alg})",
          abs(st["rho"] - st["kappa_conc"] * st["kappa_cancel"]) < 1e-6 * max(1, st["rho"]))

# 5. rotation coding is exact without faults and leaves kappa_cancel unchanged
for alg, net in models.items():
    rots = suite.rotations(net)
    keep = [torch.ones(w) for w in net.widths]
    with torch.no_grad():
        a, b = net(X), net(X, site_fn=faults.death_site_fn(keep, rots))
    check(f"rotation coding exact without faults ({alg})", torch.allclose(a, b, atol=1e-4))
    coh = coherence.coherence(net, X, d["Yte"][:1000], rots)
    kc, kc_rot = coh[1][coh[1]["natural"]]["kappa_cancel"], coh[1]["rot"]["kappa_cancel"]
    check(f"rotation leaves kappa_cancel unchanged ({alg})", abs(kc - kc_rot) / kc < 1e-3, f"{kc:.4f} vs {kc_rot:.4f}")

# 6. 8-bit quantization is nearly lossless; rotated weight quantization keeps complex structure valid
for alg, net in models.items():
    clean = suite.accuracy(net, d["Xte"], d["Yte"])
    wq = suite.wquant_curve(net, d["Xte"], d["Yte"])[0]
    wqr = suite.wquant_curve(net, d["Xte"], d["Yte"], rotated=True)[0]
    aq = suite.aquant_curve(net, d["Xte"], d["Yte"], d["Xtr"][:2000])[0]
    check(f"8-bit quantization ~ lossless ({alg})", min(wq, wqr, aq) > clean - 0.5,
          f"clean {clean:.2f} w8 {wq:.2f} w8rot {wqr:.2f} a8 {aq:.2f}")

# 7. no global RNG side effects from the suite
torch.manual_seed(123)
a = torch.rand(3)
torch.manual_seed(123)
suite.death_curve(models["real"], X[:200], d["Yte"][:200], "unit")
check("suite leaves global RNG untouched", torch.equal(a, torch.rand(3)))
print("all checks passed")
