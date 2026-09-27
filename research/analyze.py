"""Summaries of runs/runs.jsonl.

    python research/analyze.py                 # table of every configuration (mean +- sd over seeds)
    python research/analyze.py --tag E1        # one experiment
    python research/analyze.py --corr E1       # Spearman correlations: coherence vs retained accuracy
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "runs.jsonl")


def load(path=RUNS, tag=None, seeds=None):
    recs = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    return [r for r in recs if (tag is None or r["tag"] == tag) and (seeds is None or r["seed"] in seeds)]


def site(r, i, key="rho", gran=None):
    s = r["coherence"][i]
    return s[gran or s["natural"]][key]


def metrics(r):
    """Flat per-run metrics used in tables and correlations."""
    m = {"clean": r["clean"], "R": r["R"], **{f"ret_{k}": v for k, v in r["retained"].items()}}
    last = len(r["coherence"]) - 1
    for i in range(len(r["coherence"])):
        for key in ("rho", "kappa_cancel", "kappa_conc"):
            m[f"s{i}_{key}"] = site(r, i, key)
        c = r["coherence"][i]
        if "rot" in c:
            m[f"s{i}_rho_rot"] = c["rot"]["rho"]
        m[f"s{i}_crest2"] = c["crest2"]
        if "active" in c:
            m[f"s{i}_n_dead"] = c["n_dead"]
            m[f"s{i}_dead_gain_share"] = c["dead_gain_share"]
            m[f"s{i}_kc_active"] = c["active"]["kappa_cancel"]
            m[f"s{i}_conc_active"] = c["active"]["kappa_conc"]
    m["readout_kc_wrong"] = r["coherence"][last]["readout_wrong"]["kappa_cancel"]
    m["readout_kc_correct"] = r["coherence"][last]["readout_correct"]["kappa_cancel"]
    m["mean_log_rho"] = float(np.mean([np.log(site(r, i)) for i in range(len(r["coherence"]))]))
    m["mean_log_kc"] = float(np.mean([np.log(site(r, i, "kappa_cancel")) for i in range(len(r["coherence"]))]))
    return m


def grouped(recs):
    g = defaultdict(list)
    for r in recs:
        g[(r["tag"], r["name"])].append(metrics(r))
    return g


def fmt(vals):
    vals = np.array(vals, dtype=float)
    return f"{vals.mean():.3f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.3f}"


def table(recs, cols=("clean", "R", "ret_death", "ret_wnoise", "ret_wquant", "ret_aquant", "ret_death_rot",
                      "s0_rho", "s1_rho", "s1_kappa_cancel", "readout_kc_wrong")):
    rows = ["| config | n | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for (tag, name), ms in sorted(grouped(recs).items()):
        cells = [fmt([m[c] for m in ms]) if all(c in m for m in ms) else "-" for c in cols]
        rows.append(f"| {name} | {len(ms)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def correlations(recs, xs=("mean_log_rho", "mean_log_kc", "s1_rho", "s1_kappa_cancel", "readout_kc_wrong"),
                 ys=("ret_death", "ret_wnoise", "ret_wquant", "ret_aquant", "R")):
    ms = [metrics(r) for r in recs]
    rows = ["| predictor | " + " | ".join(ys) + " |", "|---" * (len(ys) + 1) + "|"]
    for x in xs:
        cells = []
        for y in ys:
            pairs = [(m[x], m[y]) for m in ms if x in m and y in m]
            cells.append(f"{spearman(*zip(*pairs)):+.2f}" if len(pairs) > 3 else "-")
        rows.append(f"| {x} | " + " | ".join(cells) + " |")
    return f"n = {len(ms)} runs\n\n" + "\n".join(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag")
    p.add_argument("--corr")
    p.add_argument("--seeds", type=int, nargs="*")
    a = p.parse_args()
    if a.corr:
        print(correlations(load(tag=a.corr, seeds=a.seeds)))
    else:
        print(table(load(tag=a.tag, seeds=a.seeds)))
