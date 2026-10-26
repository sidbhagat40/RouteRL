"""Train every model used in the report (main agent on 3 seeds + 3 ablations).

Example:
    python run_experiments.py --parallel 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

JOBS = {
    "main_seed0": ["--config", "configs/default.yaml", "--seed", "0",
                   "--model-out", "models/dqn_routerl.pt", "--log-out", "results/train_log.csv"],
    "main_seed1": ["--config", "configs/default.yaml", "--seed", "1",
                   "--model-out", "models/seeds/dqn_seed1.pt", "--log-out", "results/seeds/train_log_seed1.csv"],
    "main_seed2": ["--config", "configs/default.yaml", "--seed", "2",
                   "--model-out", "models/seeds/dqn_seed2.pt", "--log-out", "results/seeds/train_log_seed2.csv"],
    "global_reward": ["--config", "configs/ablation_global_reward.yaml",
                      "--model-out", "models/ablations/global_reward.pt",
                      "--log-out", "results/ablations/train_log_global_reward.csv"],
    "no_hold": ["--config", "configs/ablation_no_hold.yaml",
                "--model-out", "models/ablations/no_hold.pt",
                "--log-out", "results/ablations/train_log_no_hold.csv"],
    "no_confidence": ["--config", "configs/ablation_no_confidence.yaml",
                      "--model-out", "models/ablations/no_confidence.pt",
                      "--log-out", "results/ablations/train_log_no_confidence.csv"],
    "no_externality": ["--config", "configs/ablation_no_externality.yaml",
                       "--model-out", "models/ablations/no_externality.pt",
                       "--log-out", "results/ablations/train_log_no_externality.csv"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(JOBS), help="subset of jobs to run")
    p.add_argument("--parallel", type=int, default=1, help="number of concurrent training processes")
    p.add_argument("--threads", type=int, default=2, help="torch threads per process")
    p.add_argument("--total-steps", type=int, default=None)
    args = p.parse_args()

    Path("results").mkdir(exist_ok=True)
    names = args.only or list(JOBS)
    queue = list(names)
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    failed = []
    start = time.time()
    while queue or running:
        while queue and len(running) < args.parallel:
            name = queue.pop(0)
            cmd = [sys.executable, "train.py", *JOBS[name], "--threads", str(args.threads)]
            if args.total_steps:
                cmd += ["--total-steps", str(args.total_steps)]
            log = open(f"results/run_{name}.log", "w", encoding="utf-8")
            running[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            print(f"[{time.time() - start:6.0f}s] started {name}", flush=True)
        for name in list(running):
            proc, log = running[name]
            if proc.poll() is not None:
                log.close()
                del running[name]
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode != 0:
                    failed.append(name)
                print(f"[{time.time() - start:6.0f}s] finished {name}: {status}", flush=True)
        time.sleep(2)
    if failed:
        sys.exit(f"failed jobs: {failed} (see results/run_<job>.log)")


if __name__ == "__main__":
    main()
