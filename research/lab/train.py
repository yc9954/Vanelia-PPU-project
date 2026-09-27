"""Training loop on preloaded tensors (the repo's optimiser settings: Adam, lr 1e-3, weight decay 1e-4)."""
import torch
import torch.nn.functional as F

from .suite import accuracy


def train(net, data, epochs=5, seed=0, lr=1e-3, weight_decay=1e-4, batch_size=128, penalty=None):
    """penalty(net, sites) -> scalar added to the loss (None for plain cross-entropy)."""
    torch.manual_seed(seed)  # dropout draws
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    X, Y = data["Xtr"], data["Ytr"]
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:
                continue
            out, sites = net(X[idx], return_sites=True)
            loss = F.cross_entropy(out, Y[idx])
            if penalty is not None:
                loss = loss + penalty(net, sites)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return accuracy(net, data["Xva"], data["Yva"])
