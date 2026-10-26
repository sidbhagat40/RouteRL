"""Roll out policies on fixed seeds (common random numbers) and collect metrics."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import EnvConfig
from .env import SupportRoutingEnv


def run_episode(env: SupportRoutingEnv, policy, seed: int) -> dict:
    if hasattr(policy, "reset"):
        policy.reset()
    obs, _ = env.reset(seed=seed)
    done = False
    episode_return = 0.0
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs, env))
        episode_return += reward
        done = terminated or truncated
    summary = info["episode_summary"]
    summary["return"] = episode_return
    summary["seed"] = seed
    return summary


def evaluate_policies(policies, env_cfg: EnvConfig, seeds, load_multiplier: float | None = None,
                      collect_routing: bool = False):
    """Returns a per-episode DataFrame and (optionally) resolver matrices
    (priority x team that finally resolved the ticket) and backlog traces."""
    cfg = env_cfg if load_multiplier is None else replace(env_cfg, load_multiplier=load_multiplier)
    rows, routing, traces = [], {}, {}
    for policy in policies:
        env = SupportRoutingEnv(cfg)
        mat = np.zeros((env.n_priorities, env.n_teams), dtype=np.int64)
        for seed in seeds:
            row = run_episode(env, policy, seed)
            row["policy"] = policy.name
            rows.append(row)
            for tk in env.resolved:
                mat[tk.priority, tk.team] += 1
            if seed == seeds[0]:
                traces[policy.name] = list(env.backlog_trace)
        routing[policy.name] = mat
    df = pd.DataFrame(rows)
    if collect_routing:
        return df, routing, traces
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in ("policy", "seed")]
    agg = df.groupby("policy", sort=False)[metrics].agg(["mean", "std"])
    n = df.groupby("policy", sort=False).size()
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[m] = agg[(m, "mean")]
        out[f"{m}_ci95"] = 1.96 * agg[(m, "std")] / np.sqrt(n)
    return out
