"""Figures for the README and report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "Random": "#9e9e9e",
    "Round-Robin": "#bcaaa4",
    "Shortest-Queue": "#8d6e63",
    "Category-Rule": "#64b5f6",
    "Skill-Aware-SQ": "#1e88e5",
    "DQN": "#e53935",
}


def _color(name: str) -> str:
    return COLORS.get(name, "#43a047")


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curve(log: pd.DataFrame, baseline_val: dict[str, float], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(log["step"], log["train_cost"].rolling(10, min_periods=1).mean(), color="#ef9a9a",
            label="DQN train episodes (rolling mean, randomised load)")
    val = log.dropna(subset=["val_cost"])
    ax.plot(val["step"], val["val_cost"], color=_color("DQN"), marker="o", ms=3, label="DQN validation (greedy)")
    for name, cost in baseline_val.items():
        ax.axhline(cost, ls="--", lw=1.2, color=_color(name), label=f"{name} (validation)")
    ax.set_yscale("log")
    ax.set_xlabel("environment steps (routing decisions)")
    ax.set_ylabel("episode cost (lower is better)")
    ax.set_title("DQN learning curve")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper right")
    return _save(fig, out_dir, "training_curve")


def plot_policy_comparison(summary: pd.DataFrame, out_dir: Path) -> Path:
    panels = [
        ("total_cost", "Episode cost", True),
        ("mean_resolution_min", "Mean resolution time (min)", True),
        ("sla_breach_rate", "SLA breach rate", False),
        ("misroutes_per_ticket", "Misroutes per ticket", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    names = list(summary.index)
    for ax, (col, title, log) in zip(axes, panels):
        vals = summary[col].values
        err = summary[f"{col}_ci95"].values
        ax.bar(range(len(names)), vals, yerr=err, color=[_color(n) for n in names], capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    return _save(fig, out_dir, "policy_comparison")


def plot_sla_by_priority(summary: pd.DataFrame, priorities: list[str], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    names = list(summary.index)
    width = 0.8 / len(names)
    x = np.arange(len(priorities))
    for i, name in enumerate(names):
        vals = [summary.loc[name, f"sla_breach_rate_{p}"] for p in priorities]
        err = [summary.loc[name, f"sla_breach_rate_{p}_ci95"] for p in priorities]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, yerr=err, capsize=2,
               color=_color(name), label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(priorities)
    ax.set_ylabel("SLA breach rate")
    ax.set_title("SLA breaches by ticket priority")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, out_dir, "sla_by_priority")


def plot_load_sweep(sweep: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, grp in sweep.groupby("policy", sort=False):
        g = grp.sort_values("load")
        axes[0].errorbar(g["load"], g["total_cost"], yerr=g["total_cost_ci95"], marker="o", ms=3,
                         capsize=2, color=_color(name), label=name)
        axes[1].errorbar(g["load"], g["sla_breach_rate"], yerr=g["sla_breach_rate_ci95"], marker="o",
                         ms=3, capsize=2, color=_color(name), label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("episode cost")
    axes[1].set_ylabel("SLA breach rate")
    for ax in axes:
        ax.set_xlabel("demand multiplier (1.0 = nominal)")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("Robustness to demand level", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "load_sweep")


def plot_routing_heatmaps(routing: dict[str, np.ndarray], priorities: list[str], teams: list[str],
                          policies: list[str], out_dir: Path) -> Path:
    """Share of each priority's tickets finally resolved by each team."""
    fig, axes = plt.subplots(1, len(policies), figsize=(4.2 * len(policies), 2.4))
    axes = np.atleast_1d(axes)
    cols = teams
    shares = {n: routing[n] / np.maximum(routing[n].sum(axis=1, keepdims=True), 1) for n in policies}
    vmax = max(float(m.max()) for m in shares.values())
    for idx, (ax, name) in enumerate(zip(axes, policies)):
        mat = shares[name]
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=vmax)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > 0.7 * vmax else "black")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(priorities)))
        ax.set_yticklabels(priorities if idx == 0 else [], fontsize=8)
        if idx == 0:
            ax.set_ylabel("ticket priority")
        ax.set_title(f"{name}: team that resolved the ticket", fontsize=9)
    fig.colorbar(im, ax=list(axes), fraction=0.02)
    return _save(fig, out_dir, "routing_heatmap")


def plot_backlog_trace(traces: dict[str, list[int]], surge, shift_minutes: int, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for name, trace in traces.items():
        ax.plot(np.arange(1, len(trace) + 1), trace, color=_color(name), lw=1.3, label=name)
    if surge is not None:
        ax.axvspan(surge[0], surge[1], color="orange", alpha=0.15, label="incident surge")
    ax.axvline(shift_minutes, color="k", ls=":", lw=1, label="arrivals stop")
    ax.set_xlabel("minute")
    ax.set_ylabel("unresolved tickets")
    ax.set_title("Backlog during one test shift")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, out_dir, "backlog_trace")
