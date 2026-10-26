"""Evaluate training-seed robustness and the ablation variants on the test seeds.

Example:
    python evaluate_ablations.py --episodes 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from routerl.baselines import SkillAwareShortestQueuePolicy
from routerl.dqn import DQNAgent, DQNPolicy
from routerl.evaluation import evaluate_policies, summarize
from evaluate import TEST_SEEDS_START

VARIANTS = {
    "DQN (seed 0)": ("models/dqn_routerl.pt", "results/train_log.csv"),
    "DQN (seed 1)": ("models/seeds/dqn_seed1.pt", "results/seeds/train_log_seed1.csv"),
    "DQN (seed 2)": ("models/seeds/dqn_seed2.pt", "results/seeds/train_log_seed2.csv"),
    "Global reward": ("models/ablations/global_reward.pt", "results/ablations/train_log_global_reward.csv"),
    "No hold action": ("models/ablations/no_hold.pt", "results/ablations/train_log_no_hold.csv"),
    "No confidence": ("models/ablations/no_confidence.pt", "results/ablations/train_log_no_confidence.csv"),
    "No externality": ("models/ablations/no_externality.pt", "results/ablations/train_log_no_externality.csv"),
}
COLUMNS = ["total_cost", "mean_resolution_min", "sla_breach_rate", "sla_breach_rate_P1",
           "misroutes_per_ticket", "holds_per_ticket"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--stress-load", type=float, default=1.45,
                   help="demand multiplier for the overload comparison")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out, fig_dir = Path(args.out), Path(args.out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(TEST_SEEDS_START, TEST_SEEDS_START + args.episodes))

    def run_all(load):
        frames, env_cfg = [], None
        for name, (model, _) in VARIANTS.items():
            if not Path(model).exists():
                print(f"skipping {name}: {model} not found")
                continue
            agent, cfg = DQNAgent.load(model)
            env_cfg = env_cfg or cfg
            frames.append(evaluate_policies([DQNPolicy(agent, name)], cfg, seeds, load_multiplier=load))
        frames.append(evaluate_policies([SkillAwareShortestQueuePolicy()], env_cfg, seeds, load_multiplier=load))
        return summarize(pd.concat(frames, ignore_index=True))

    pd.set_option("display.width", 200)
    summary = run_all(None)
    summary.to_csv(out / "ablation_summary.csv")
    print("=== nominal demand ===")
    print(summary[COLUMNS].round(3).to_string())
    stress = run_all(args.stress_load)
    stress.to_csv(out / "ablation_summary_stress.csv")
    print(f"\n=== overload: demand x{args.stress_load} ===")
    print(stress[COLUMNS].round(3).to_string())

    main_runs = summary.loc[[n for n in summary.index if n.startswith("DQN (seed")]]
    print(f"\nDQN across {len(main_runs)} training seeds: total cost "
          f"{main_runs.total_cost.mean():.0f} +- {main_runs.total_cost.std():.0f} (mean +- std), "
          f"SLA breach {main_runs.sla_breach_rate.mean():.4f} +- {main_runs.sla_breach_rate.std():.4f}")

    # Bar chart: cost and breach rate per variant.
    colors = ["#e53935"] * len(main_runs) + ["#fb8c00", "#8e24aa", "#00897b", "#6d4c41", "#1e88e5"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0))
    names = list(summary.index)
    panels = [(summary, "total_cost", "Episode cost (nominal demand)"),
              (summary, "sla_breach_rate", "SLA breach rate (nominal demand)"),
              (stress, "total_cost", f"Episode cost (demand x{args.stress_load})")]
    for ax, (table, col, title) in zip(axes, panels):
        ax.bar(range(len(names)), table.loc[names, col], yerr=table.loc[names, f"{col}_ci95"], capsize=3,
               color=colors[: len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3, which="both")
        if col == "total_cost":
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(fig_dir / "ablations.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "ablations.pdf", bbox_inches="tight")
    plt.close(fig)

    # Validation curves of every variant.
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for (name, (_, log_path)), color in zip(VARIANTS.items(), colors):
        if not Path(log_path).exists():
            continue
        log = pd.read_csv(log_path).dropna(subset=["val_cost"])
        ax.plot(log["step"], log["val_cost"], color=color, lw=1.2, label=name,
                alpha=0.6 if "seed 1" in name or "seed 2" in name else 1.0)
    ax.axhline(summary.loc["Skill-Aware-SQ", "total_cost"], color="#1e88e5", ls="--", lw=1,
               label="Skill-Aware-SQ (test)")
    ax.set_yscale("log")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("validation episode cost")
    ax.set_title("Learning curves: seeds and ablations")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "ablation_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "ablation_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"written {out / 'ablation_summary.csv'} and figures in {fig_dir}/")


if __name__ == "__main__":
    main()
