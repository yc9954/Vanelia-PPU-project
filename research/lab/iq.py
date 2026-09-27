"""Synthetic I/Q modulation classification: complex-native data (experiment E4).

Each example is L complex baseband symbols from one of seven constellations (phase-coded, amplitude-
coded and mixed), multiplied by a random channel gain and a random carrier phase e^{i theta}, plus
complex white Gaussian noise at an SNR drawn uniformly from [0, 16] dB. The carrier phase is a global
U(1) symmetry: the label does not depend on it, so a network that is phase-equivariant up to a
modulus readout (complex layers without bias, modReLU, no BatchNorm, |z| readout) is exactly
invariant, while a real network must learn the invariance from data.

"iq" draws theta ~ U[0, 2 pi); "iq_fixed" sets theta = 0 (control: no symmetry to exploit).
Inputs are paired real coordinates [re_1..re_L, im_1..im_L]; the dataset is generated once from a
fixed seed.
"""
import math

import torch

CLASSES = ("BPSK", "QPSK", "8PSK", "PAM4", "16QAM", "64QAM", "OOK")


def constellation(name):
    if name == "BPSK":
        pts = torch.tensor([1.0, -1.0]).to(torch.complex64)
    elif name == "QPSK":
        pts = torch.exp(1j * (math.pi / 4 + math.pi / 2 * torch.arange(4)))
    elif name == "8PSK":
        pts = torch.exp(1j * math.pi / 4 * torch.arange(8))
    elif name == "PAM4":
        pts = torch.tensor([-3.0, -1.0, 1.0, 3.0]).to(torch.complex64)
    elif name in ("16QAM", "64QAM"):
        g = torch.arange(-3.0, 4.0, 2.0) if name == "16QAM" else torch.arange(-7.0, 8.0, 2.0)
        pts = (g[:, None] + 1j * g[None, :]).flatten()
    elif name == "OOK":
        pts = torch.tensor([0.0, 1.0]).to(torch.complex64)
    else:
        raise ValueError(name)
    pts = pts.to(torch.complex64)
    return pts / torch.sqrt((pts.abs() ** 2).mean())


def iq_dataset(name="iq", L=32, sizes=(54000, 6000, 10000), snr_db=(0.0, 16.0), seed=1234):
    g = torch.Generator().manual_seed(seed)
    consts = [constellation(c) for c in CLASSES]

    def make(n):
        y = torch.randint(0, len(CLASSES), (n,), generator=g)
        z = torch.empty(n, L, dtype=torch.complex64)
        for k, pts in enumerate(consts):
            idx = (y == k).nonzero().flatten()
            z[idx] = pts[torch.randint(0, len(pts), (len(idx), L), generator=g)]
        theta = torch.zeros(n) if name == "iq_fixed" else torch.rand(n, generator=g) * 2 * math.pi
        gain = 0.7 + 0.6 * torch.rand(n, generator=g)
        z = z * (gain * torch.exp(1j * theta))[:, None]
        snr = snr_db[0] + (snr_db[1] - snr_db[0]) * torch.rand(n, generator=g)
        std = gain / torch.sqrt(10 ** (snr / 10))
        noise = torch.complex(torch.randn(n, L, generator=g), torch.randn(n, L, generator=g)) / math.sqrt(2)
        z = z + noise * std[:, None]
        return torch.cat([z.real, z.imag], 1).float(), y, snr

    (Xtr, Ytr, _), (Xva, Yva, _), (Xte, Yte, snr_te) = (make(n) for n in sizes)
    return dict(name=name, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte, snr_te=snr_te,
                n_in=2 * L, n_classes=len(CLASSES), input_paired=True)
