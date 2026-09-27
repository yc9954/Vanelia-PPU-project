"""Train one configuration over several seeds, run the fault suite and coherence metrics, and append
one JSON line per seed to runs/runs.jsonl.

    python research/run.py --name real_linear --algebra real --readout linear --seeds 0 1 2
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from lab import coherence, data, nets, penalties, suite  # noqa: E402
from lab.train import train  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def build(args, d):
    return nets.Net(d["n_in"], args.widths, d["n_classes"], algebra=args.algebra, readout=args.readout,
                    act=args.act, norm=args.norm, input_paired=d["input_paired"], dropout=args.dropout,
                    init=args.init)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--tag", default="", help="experiment id, e.g. E1")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--algebra", default="real", choices=nets.ALGEBRAS)
    p.add_argument("--readout", default="linear", choices=nets.READOUTS)
    p.add_argument("--act", default="relu", choices=("relu", "modrelu"))
    p.add_argument("--norm", default="bn", choices=("bn", "none"))
    p.add_argument("--widths", type=int, nargs="+", default=[64, 64])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--init", default="uniform", choices=("uniform", "repo"))
    p.add_argument("--penalty", default="none", choices=penalties.PENALTIES)
    p.add_argument("--penalty-weight", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--families", nargs="+", default=list(suite.DEFAULT_FAMILIES))
    p.add_argument("--out", default=os.path.join(HERE, "runs", "runs.jsonl"))
    args = p.parse_args()

    d = data.load(args.dataset)
    torch.set_num_threads(int(os.environ.get("LAB_THREADS", "4")))
    config = {k: v for k, v in vars(args).items() if k not in ("seeds", "out", "name", "tag")}
    for seed in args.seeds:
        torch.manual_seed(seed)
        net = build(args, d)
        t0 = time.time()
        pen = penalties.make(args.penalty, args.penalty_weight)
        val = train(net, d, epochs=args.epochs, seed=seed, lr=args.lr, penalty=pen)
        t1 = time.time()
        res = suite.suite(net, d, args.families)
        coh = coherence.coherence(net, d["Xte"][:2000], d["Yte"][:2000], suite.rotations(net))
        rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), commit=git_commit(), name=args.name, tag=args.tag,
                   seed=seed, config=config, params=nets.count_params(net), val_acc=val,
                   train_sec=round(t1 - t0, 1), eval_sec=round(time.time() - t1, 1), **res, coherence=coh)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a") as f:  # parallel batches append to the same file
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(rec) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
        ret = " ".join(f"{k}={v:.3f}" for k, v in res["retained"].items())
        last = coh[-1]
        print(f"{args.name} seed {seed}: clean {res['clean']:.2f} R {res['R']:.3f} | {ret} | "
              f"readout rho {last[last['natural']]['rho']:.3f} kc {last[last['natural']]['kappa_cancel']:.3f} "
              f"| {rec['train_sec']}s+{rec['eval_sec']}s", flush=True)


if __name__ == "__main__":
    main()
