"""Keep/discard decisions for the autoresearch loop (README, "Loop rules").

    python research/compare.py --tag E5 --incumbent r0_base --baseline-clean 97.35 [--names a b ...]

A candidate is KEPT if its mean R over seeds beats the incumbent's by more than the larger of the two
between-seed standard deviations, and its mean clean accuracy is at least baseline-clean - 0.5.
Runs with other tags can be referenced as TAG:name (e.g. E1:real_lin).
"""
import argparse

import numpy as np

import analyze as A


def stats(recs):
    R = np.array([r["R"] for r in recs])
    clean = np.array([r["clean"] for r in recs])
    fam = {k: np.mean([r["retained"][k] for r in recs]) for k in ("death", "wnoise", "wquant", "aquant")}
    return dict(n=len(recs), R=R.mean(), R_sd=R.std(ddof=1) if len(R) > 1 else 0.0, clean=clean.mean(), **fam)


def lookup(ref, tag, seeds):
    t, name = ref.split(":", 1) if ":" in ref else (tag, ref)
    recs = [r for r in A.load(tag=t, seeds=seeds) if r["name"] == name]
    if not recs:
        raise SystemExit(f"no runs for {ref}")
    return stats(recs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="E5")
    p.add_argument("--incumbent", required=True)
    p.add_argument("--baseline-clean", type=float, required=True)
    p.add_argument("--names", nargs="*")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = p.parse_args()
    inc = lookup(a.incumbent, a.tag, a.seeds)
    names = a.names or sorted({r["name"] for r in A.load(tag=a.tag, seeds=a.seeds)} - {a.incumbent})
    rows = [(n, lookup(n, a.tag, a.seeds)) for n in names]
    rows.sort(key=lambda x: -x[1]["R"])
    print(f"incumbent {a.incumbent}: R {inc['R']:.3f} +- {inc['R_sd']:.3f}, clean {inc['clean']:.2f}")
    print(f"| candidate | R | dR | clean | death | wnoise | wquant | aquant | decision |\n|---" + "|---" * 8 + "|")
    for n, s in rows:
        margin = max(inc["R_sd"], s["R_sd"])
        ok_clean = s["clean"] >= a.baseline_clean - 0.5
        keep = s["R"] - inc["R"] > margin and ok_clean
        why = "KEEP" if keep else ("discard (clean)" if not ok_clean else "discard (within noise)"
                                   if s["R"] > inc["R"] else "discard")
        print(f"| {n} | {s['R']:.3f} +- {s['R_sd']:.3f} | {s['R'] - inc['R']:+.3f} | {s['clean']:.2f} | "
              f"{s['death']:.3f} | {s['wnoise']:.3f} | {s['wquant']:.3f} | {s['aquant']:.3f} | {why} |")


if __name__ == "__main__":
    main()
