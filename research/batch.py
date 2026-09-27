"""Run a batch file of run.py argument lines in parallel (one thread per worker).

    python research/batch.py research/batches/E1.txt --workers 4
Lines starting with # are skipped.
"""
import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def run(line):
    env = dict(os.environ, LAB_THREADS="1")
    cmd = [sys.executable, os.path.join(HERE, "run.py")] + shlex.split(line)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if "Warning" not in l and "warn(" not in l)
    return p.returncode, line, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.batch) if l.strip() and not l.lstrip().startswith("#")]
    failed = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for code, line, out in ex.map(run, lines):
            print(out.strip(), flush=True)
            if code:
                failed += 1
                print(f"FAILED ({code}): {line}", flush=True)
    print(f"done: {len(lines) - failed}/{len(lines)} ok")
