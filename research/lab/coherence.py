"""Coherence of a site's contributions to the next layer: rho = kappa_conc * kappa_cancel.

For site outputs u (batch x n) and the downstream affine map s = M u + c (eval-mode BatchNorm folded
into the next linear layer), the signal is S = M u = sum_J a_J with a_J = sum_{j in J} M[:, j] u_j over
failure units J (single units or pairs). With expectations over inputs (ratio of expectations):

  rho          = E sum_J ||a_J||^2 / E ||S||^2          unit-death variance factor:
                 E||delta||^2 / E||S||^2 = q2 + (q1 - q2) rho for k of N units dead
                 (q1 = k/N, q2 = k(k-1)/(N(N-1)); Bernoulli(p): p^2 + p(1-p) rho)
  kappa_cancel = E [||M||_F^2 ||u||^2 / n] / E ||S||^2  1 = contributions add like random vectors,
                 < 1 constructive, > 1 destructive interference (cancellation)
  kappa_conc   = rho / kappa_cancel                     1 = work spread evenly over units
  *_var        the same ratios with the signal measured as its variance over inputs, E||S - E S||^2.
               Training against the raw-energy version is gameable: a large input-independent
               component inflates ||S|| without carrying information (iteration 4 in LOG.md).

kappa_cancel is a noise-to-signal gain ratio: destructive interference raises it, and so do units that
never fire but still carry downstream gain (e.g. dead ReLUs behind BatchNorm). "active" repeats the
statistics over units that fire on at least one input. Raw sums are returned so any ratio can be
recomputed later.
"""
import torch

from .nets import natural_granularity


@torch.no_grad()
def downstream_affine(net, i, n):
    net.eval()
    c = net.downstream(i, torch.zeros(1, n))
    M = (net.downstream(i, torch.eye(n)) - c).T  # (m, n); column j = effect of unit j
    return M, c


@torch.no_grad()
def site_stats(U, M, granularity, c=None, rows=None):
    """rows: optional (batch, m) bool mask selecting output rows per sample (e.g. wrong classes)."""
    n = U.shape[1]
    S = U @ M.T
    w = torch.ones_like(S) if rows is None else rows.float()
    sig = ((S ** 2) * w).sum()
    colsq = w @ (M ** 2)  # (batch, n): per sample, sum over selected rows of M[r, j]^2
    contrib = ((U ** 2) * colsq).sum()
    if granularity == "pair":
        k = n // 2
        contrib = contrib + 2 * (U[:, :k] * U[:, k:] * (w @ (M[:, :k] * M[:, k:]))).sum()
    rand = (colsq.sum(1) * (U ** 2).sum(1) / n).sum()
    Sc = S - (S * w).sum(0) / w.sum(0).clamp_min(1)  # centred over inputs (per selected row)
    sig_var = ((Sc ** 2) * w).sum()
    out = dict(sig=sig.item(), sig_var=sig_var.item(), contrib=contrib.item(), rand=rand.item(),
               rho=(contrib / sig).item(), kappa_cancel=(rand / sig).item(), kappa_conc=(contrib / rand).item(),
               rho_var=(contrib / sig_var).item(), kappa_cancel_var=(rand / sig_var).item())
    if c is not None:
        full = (((S + c) ** 2) * w).sum()
        out.update(full=full.item(), rho_full=(contrib / full).item(), kappa_cancel_full=(rand / full).item())
    return out


@torch.no_grad()
def crest2(U):
    """Per-tensor activation crest factor^2: n * max|u|^2 / E||u||^2 (sets the quantization step)."""
    return (U.shape[1] * U.abs().max() ** 2 / (U ** 2).sum(1).mean()).item()


@torch.no_grad()
def coherence(net, X, Y, rotations=None):
    net.eval()
    _, sites = net(X, return_sites=True)
    nat = natural_granularity(net)
    out = []
    for i, U in enumerate(sites):
        n = U.shape[1]
        M, c = downstream_affine(net, i, n)
        alive = U.abs().amax(0) > 0  # units that fire on at least one input
        rec = dict(site=i, n=n, natural=nat, unit=site_stats(U, M, "unit", c), pair=site_stats(U, M, "pair", c),
                   crest2=crest2(U), n_dead=int((~alive).sum()),
                   dead_gain_share=((M[:, ~alive] ** 2).sum() / (M ** 2).sum()).item(),
                   active=site_stats(U[:, alive], M[:, alive], "unit", c))
        if rotations is not None:
            R = rotations[i]
            rec["rot"] = site_stats(U @ R.T, M @ R.T, nat, c)
            rec["crest2_rot"] = crest2(U @ R.T)
        if i == len(sites) - 1:  # readout: split rows into the correct class and the wrong classes
            row_class = torch.arange(M.shape[0]) % net.n_classes
            correct = row_class[None, :] == Y[:, None]
            rec["readout_correct"] = site_stats(U, M, nat, c, rows=correct)
            rec["readout_wrong"] = site_stats(U, M, nat, c, rows=~correct)
        out.append(rec)
    return out
