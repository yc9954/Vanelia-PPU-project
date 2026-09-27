"""Differentiable coherence penalties for training (experiment E3).

On each batch and hidden site, with the eval-mode BatchNorm scale (running variance, detached) folded
into the next layer, M = W_next diag(gamma / sigma):
  "rho":    log( sum_j ||M_j||^2 u_j^2 / ||M u||^2 )             concentration x cancellation
  "cancel": log( (||M||_F^2 ||u||^2 / n) / ||M u||^2 )           cancellation only
averaged over sites and scaled by the penalty weight. These are the quantities coherence.py measures.
"""
import torch

PENALTIES = ("none", "rho", "cancel")


def _terms(net, i, u):
    nxt = net.lins[i + 1] if i + 1 < len(net.lins) else net.head
    M = nxt.weight()
    norm = net.norms[i]
    if isinstance(norm, torch.nn.BatchNorm1d):
        M = M * (norm.weight / torch.sqrt(norm.running_var.detach() + norm.eps))[None, :]
    sig = ((u @ M.T) ** 2).sum()
    contrib = ((u ** 2) @ (M ** 2).sum(0)).sum()
    rand = (M ** 2).sum() * (u ** 2).sum() / u.shape[1]
    return sig, contrib, rand


def make(kind, weight):
    if kind == "none" or weight == 0:
        return None

    def penalty(net, sites):
        total = 0.0
        for i, u in enumerate(sites):
            sig, contrib, rand = _terms(net, i, u)
            total = total + torch.log(((contrib if kind == "rho" else rand) + 1e-8) / (sig + 1e-8))
        return weight * total / len(sites)

    return penalty
