"""Structured MLPs: one code path for real, complex and split-complex networks.

Complex and split-complex layers act on paired real coordinates [re_1..re_k, im_1..im_k], so every
model is a real network whose weight matrices have an algebra-dependent structure:

    complex (A + iB)(x + iy):             W = [[A, -B], [B, A]]
    split-complex (A + jB)(x + jy), j^2=1: W = [[A,  B], [B, A]]

With a real-valued input (imaginary part 0) the first layer is W = [[A], [B]] for both. Every
effective weight starts from nn.Linear's default distribution U(-1/sqrt(fan_in), 1/sqrt(fan_in)) and
every bias from 0, so the algebras differ only in structure, not in initial scale (init="repo"
reproduces the original ComplexLinear initialisation, N(0, 2 / (fan_in + fan_out)) per part).

Fault injection and coherence measurements see every model through one interface: hidden *sites*
(post-activation unit outputs, where units fail) and `downstream(i, u)`, the affine map from site i
to the next pre-activation (eval-mode BatchNorm followed by the next linear layer).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ALGEBRAS = ("real", "complex", "split")
READOUTS = ("linear", "modulus", "modulus_untied")


class StructuredLinear(nn.Module):
    def __init__(self, n_in, n_out, algebra="real", paired_input=True, bias=True, init="uniform"):
        super().__init__()
        if algebra not in ALGEBRAS:
            raise ValueError(algebra)
        self.n_in, self.n_out, self.algebra, self.paired_input = n_in, n_out, algebra, paired_input
        bound = 1.0 / math.sqrt(n_in)
        if algebra == "real":
            self.W = nn.Parameter(torch.empty(n_out, n_in).uniform_(-bound, bound))
        else:
            if n_out % 2 or (paired_input and n_in % 2):
                raise ValueError("complex/split layers need even (paired) sizes")
            k_in = n_in // 2 if paired_input else n_in
            self.A = nn.Parameter(torch.empty(n_out // 2, k_in).uniform_(-bound, bound))
            self.B = nn.Parameter(torch.empty(n_out // 2, k_in).uniform_(-bound, bound))
            if init == "repo":  # models/complex_layers.py: N(0, 2 / (fan_in + fan_out)) per part, complex units
                std = math.sqrt(2.0 / (k_in + n_out // 2))
                nn.init.normal_(self.A, 0.0, std)
                nn.init.normal_(self.B, 0.0, std)
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None

    def stored(self):
        """The physically stored weight tensors: what weight faults act on."""
        return [self.W] if self.algebra == "real" else [self.A, self.B]

    def weight(self):
        if self.algebra == "real":
            return self.W
        if not self.paired_input:
            return torch.cat([self.A, self.B], 0)
        s = -1.0 if self.algebra == "complex" else 1.0
        return torch.cat([torch.cat([self.A, s * self.B], 1), torch.cat([self.B, self.A], 1)], 0)

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class Net(nn.Module):
    """MLP: [StructuredLinear -> activation (site) -> BatchNorm] x L -> readout.

    Readouts: "linear" (dense logits; for a complex network this is coherent/homodyne detection of
    one quadrature), "modulus" (algebra-structured layer to C paired outputs, logits |z|: intensity
    detection) and "modulus_untied" (dense layer to 2C outputs, then |z| of each pair).
    """

    def __init__(self, n_in, widths, n_classes, algebra="real", readout="linear", act="relu",
                 norm="bn", input_paired=False, dropout=0.0, init="uniform"):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(readout)
        paired = algebra != "real"
        if act == "modrelu" and not paired:
            raise ValueError("modReLU needs paired (complex/split) units")
        self.algebra, self.readout, self.act, self.dropout = algebra, readout, act, dropout
        self.widths, self.n_classes = list(widths), n_classes
        self.lins, self.norms, self.mod_bias = nn.ModuleList(), nn.ModuleList(), nn.ParameterList()
        prev, prev_paired = n_in, input_paired
        for w in self.widths:
            self.lins.append(StructuredLinear(prev, w, algebra, paired_input=prev_paired, init=init))
            self.norms.append(nn.BatchNorm1d(w) if norm == "bn" else nn.Identity())
            if act == "modrelu":
                self.mod_bias.append(nn.Parameter(torch.zeros(w // 2)))
            prev, prev_paired = w, paired
        C = n_classes
        if readout == "linear":
            self.head = StructuredLinear(prev, C, "real")
        elif readout == "modulus":
            self.head = StructuredLinear(prev, 2 * C, algebra, paired_input=prev_paired, init=init)
        else:
            self.head = StructuredLinear(prev, 2 * C, "real")

    def layers(self):
        return list(self.lins) + [self.head]

    def activate(self, i, z):
        if self.act == "relu":
            return F.relu(z)  # CReLU for paired units
        k = z.shape[1] // 2  # modReLU: z * relu(|z| + b) / |z|, phase preserving
        re, im = z[:, :k], z[:, k:]
        r = torch.sqrt(re * re + im * im + 1e-8)
        g = F.relu(r + self.mod_bias[i]) / r
        return torch.cat([re * g, im * g], 1)

    def logits_from(self, o):
        if self.readout == "linear":
            return o
        C = self.n_classes
        return torch.sqrt(o[:, :C] ** 2 + o[:, C:] ** 2 + 1e-8)

    def forward(self, x, site_fn=None, return_sites=False):
        h, sites = x, []
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            u = self.activate(i, lin(h))
            if self.training and self.dropout > 0:
                u = u * self._dropout_keep(u)
            if site_fn is not None:
                u = site_fn(i, u)
            sites.append(u)
            h = norm(u)
        out = self.logits_from(self.head(h))
        return (out, sites) if return_sites else out

    def _dropout_keep(self, u):
        """Inverted dropout at the natural granularity: real units, or whole complex/split units."""
        if self.algebra == "real":
            keep = (torch.rand_like(u) > self.dropout).float()
        else:
            half = (torch.rand(u.shape[0], u.shape[1] // 2) > self.dropout).float()
            keep = torch.cat([half, half], 1)
        return keep / (1 - self.dropout)

    def downstream(self, i, u):
        """Affine map from site i's unit outputs to the next pre-activation (call in eval mode)."""
        nxt = self.lins[i + 1] if i + 1 < len(self.lins) else self.head
        return nxt(self.norms[i](u))


def natural_granularity(net):
    return "unit" if net.algebra == "real" else "pair"


def count_params(net):
    return sum(p.numel() for p in net.parameters())
