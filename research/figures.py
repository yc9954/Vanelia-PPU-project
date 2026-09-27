"""Paper figures from runs/runs.jsonl and runs/e2.jsonl -> figures/*.png

    python research/figures.py
Palette: the dataviz reference categorical slots (blue, orange, aqua), validated all-pairs; every
series also has a marker shape and a direct label, so identity never depends on colour alone.
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ALGEBRA = {"real": (BLUE, "o", "real"), "complex": (ORANGE, "s", "complex"), "split": (AQUA, "^", "split-complex")}
OUT = os.path.join(HERE, "figures")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join("research", "figures", name))


def fig_rotation_crossover():
    """E2: plain vs rotated faults under function-preserving channel outliers."""
    rows = [json.loads(line) for line in open(os.path.join(HERE, "runs", "e2.jsonl"))]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["name"], r["scale"])].append(r)
    configs = [("real_lin", "real MLP, BatchNorm"), ("cplx_mod", "complex MLP, |z| readout"),
               ("real_lin_nobn", "real MLP, no BatchNorm")]
    scales = sorted({s for _, s in agg})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey="row")
    for c, (name, title) in enumerate(configs):
        for row, (plain_key, rot_key, ylabel, pick) in enumerate([
                ("aquant", "aquant_rot", "accuracy at 4-bit activations (%)", lambda r, k: r[k]["4"]),
                ("death", "death_rot", "accuracy with 30% units dead (%)", lambda r, k: r[k][2])]):
            ax = axes[row, c]
            for key, color, marker, label in ((plain_key, BLUE, "o", "plain"), (rot_key, ORANGE, "s", "rotated")):
                m = [np.mean([pick(r, key) for r in agg[(name, s)]]) for s in scales]
                sd = [np.std([pick(r, key) for r in agg[(name, s)]]) for s in scales]
                ax.errorbar(scales, m, yerr=sd, color=color, marker=marker, capsize=0, label=label)
                ax.annotate(label, (scales[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                            color=INK2, va="center", fontsize=9)
            if row == 0:  # theory: where the predicted rotated/plain error ratio (mean over sites) crosses 1
                pred = [np.mean([np.mean(r["pred_aquant_ratio"]) for r in agg[(name, s)]]) for s in scales]
                cross = next((s for s, p in zip(scales, pred) if p < 1), None)
                if cross is not None:
                    ax.axvline(cross, color=INK2, lw=1, ls=":")
                    ax.annotate("theory: rotation\nwins from here", (cross, 4), xytext=(-5, 0), ha="right",
                                textcoords="offset points", fontsize=8.5, color=INK2)
                ax.set_title(title)
            ax.set_xscale("log", base=2)
            ax.set_xticks(scales, [str(s) for s in scales])
            ax.set_ylim(0, 100)
            if c == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("outlier scale s (function unchanged)")
    axes[1, 0].legend(loc="lower left")
    fig.suptitle("Randomized-Hadamard rotation helps quantization only once channel outliers appear,\n"
                 "and always hurts unit death (mean of 3 seeds, bars = 1 sd)", fontsize=11, color=INK)
    save(fig, "rotation_crossover.png")


def fig_rho_predicts(tag="E1b"):
    """E1: a clean-data statistic predicts fault robustness across architectures."""
    recs = A.load(tag=tag) or A.load(tag="E1")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (ykey, ylabel) in zip(axes, [("ret_death", "retained accuracy, unit death"),
                                         ("ret_wnoise", "retained accuracy, weight noise")]):
        xs_all, ys_all = [], []
        for alg, (color, marker, label) in ALGEBRA.items():
            ms = [A.metrics(r) for r in recs if r["config"]["algebra"] == alg]
            xs, ys = [m["mean_log_rho"] for m in ms], [m[ykey] for m in ms]
            xs_all += xs
            ys_all += ys
            ax.scatter(xs, ys, color=color, marker=marker, s=34, label=label, edgecolor=SURFACE, linewidth=1)
        ax.set_xlabel("mean log rho over hidden sites (clean data, no faults)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Spearman {A.spearman(xs_all, ys_all):+.2f}  (n = {len(xs_all)} runs)")
    axes[0].legend(loc="lower left")
    fig.suptitle("The unit-death variance factor rho, measured without any fault sampling, ranks robustness",
                 fontsize=11)
    save(fig, "rho_predicts_robustness.png")


def fig_tones():
    """E4: clean accuracy and robustness on non-coherent tone detection, by design."""
    recs = A.load(tag="E4b") + A.load(tag="E4c")
    offsets = {"tn_real_lin_w40": (0, -15, "center"), "tn_cplx_inv_crelu": (7, -3, "left")}
    rows = [("tn_real_lin_w40", "real, 4.8k params", BLUE, "o"),
            ("tn_real_lin", "real, 9.2k params", BLUE, "o"),
            ("tn_real_lin_w128", "real, 26.6k params", BLUE, "o"),
            ("tn_real_modu_nobn", "real + untied |z| readout", BLUE, "o"),
            ("tn_cplx_mod", "complex, CReLU + BN", ORANGE, "s"),
            ("tn_cplx_inv_crelu", "complex invariant, CReLU", ORANGE, "s"),
            ("tn_cplx_inv", "complex invariant (modReLU), 4.8k", ORANGE, "s"),
            ("tn_cplx_inv_w90", "complex invariant, 7.9k", ORANGE, "s")]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for name, label, color, marker in rows:
        ms = [A.metrics(r) for r in recs if r["name"] == name]
        x, y = np.mean([m["clean"] for m in ms]), np.mean([m["R"] for m in ms])
        ax.scatter([x], [y], color=color, marker=marker, s=60, edgecolor=SURFACE, linewidth=1, zorder=3)
        dx, dy, ha = offsets.get(name, (7, -3, "left"))
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK2, ha=ha)
    ax.axvline(80.2, color=INK2, lw=1, ls=":")
    ax.annotate("matched-filter\nbank 80.2%", (80.2, 0.884), xytext=(-5, 0), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("clean accuracy (%), unknown-phase tone detection")
    ax.set_ylabel("robustness R (mean retained accuracy, 4 fault types)")
    ax.set_xlim(78.2, 83.4)
    ax.scatter([], [], color=BLUE, marker="o", label="real-valued")
    ax.scatter([], [], color=ORANGE, marker="s", label="complex-valued")
    ax.legend(loc="lower right")
    ax.set_title("Matching the phase symmetry buys accuracy and fault tolerance together")
    save(fig, "tones_design.png")


def fig_autoresearch():
    """E5: loop trajectory on search seeds vs held-out seeds."""
    def R(tag, name, seeds):
        rs = [r["R"] for r in A.load(tag=tag, seeds=seeds) if r["name"] == name]
        return (np.mean(rs), np.std(rs, ddof=1)) if rs else (np.nan, 0)
    tracks = [
        ("MNIST (real-valued data)", [
            ("real baseline", ("E1", "real_lin"), ("E5c", "real_lin")),
            ("round 1:\ndropout 0.15", ("E5", "m1_do0.15"), ("E5c", "m1_do0.15")),
            ("round 2:\nsplit + dropout", ("E5", "m2_do0.15_split_lin"), ("E5c", "m2_do0.15_split_lin")),
        ]),
        ("tones (complex-native data)", [
            ("real baseline", ("E4b", "tn_real_lin"), ("E5c", "tn_real_lin")),
            ("phase-invariant\ncomplex", ("E4b", "tn_cplx_inv"), ("E5c", "tn_cplx_inv")),
            ("+ dropout 0.2", ("E4b", "tn_cplx_inv_do"), ("E5c", "tn_cplx_inv_do")),
            ("round 2: width 90\n+ dropout 0.3", ("E5", "t2_inv_w90_do0.3"), ("E5c", "t2_inv_w90_do0.3")),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, stages) in zip(axes, tracks):
        x = np.arange(len(stages))
        for (label, color, marker, idx, seeds), dx in zip(
                [("search seeds 0-2", BLUE, "o", 1, [0, 1, 2]), ("held-out seeds 3-5", ORANGE, "s", 2, [3, 4, 5])],
                (-0.06, 0.06)):
            m, sd = zip(*[R(*st[idx], seeds) for st in stages])
            ax.errorbar(x + dx, m, yerr=sd, color=color, marker=marker, capsize=0, label=label)
        ax.set_xticks(x, [st[0] for st in stages], fontsize=8.5)
        ax.set_title(title)
        ax.set_ylim(0.76, 0.95)
    axes[0].set_ylabel("robustness R (mean retained accuracy)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Autoresearch loop: gains that replicate on held-out seeds, and one that does not", fontsize=11)
    save(fig, "autoresearch_trajectory.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["rotation", "rho", "tones", "autoresearch"]
    if "autoresearch" in which:
        fig_autoresearch()
    if "rotation" in which:
        fig_rotation_crossover()
    if "rho" in which:
        fig_rho_predicts()
    if "tones" in which:
        fig_tones()
