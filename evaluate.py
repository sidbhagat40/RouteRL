"""Benchmark the trained DQN against the baselines and produce all figures.

Example:
    python evaluate.py --model models/dqn_routerl.pt --episodes 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from routerl.baselines import (CategoryRulePolicy, SkillAwareShortestQueuePolicy, make_baselines)
from routerl.dqn import DQNAgent, DQNPolicy
from routerl.env import SupportRoutingEnv
from routerl.evaluation import evaluate_policies, run_episode, summarize
from routerl import plotting
from train import VALIDATION_SEEDS_START

TEST_SEEDS_START = 10_000
HEADLINE_COLUMNS = ["total_cost", "mean_resolution_min", "p90_resolution_min", "sla_breach_rate",
                    "sla_breach_rate_P1", "sla_breach_rate_P2", "sla_breach_rate_P3",
                    "misroutes_per_ticket", "first_touch_correct", "holds_per_ticket"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/dqn_routerl.pt")
    p.add_argument("--episodes", type=int, default=30, help="test episodes per policy")
    p.add_argument("--sweep-episodes", type=int, default=10, help="episodes per load level")
    p.add_argument("--loads", type=float, nargs="+", default=[0.7, 0.85, 1.0, 1.15, 1.3, 1.45])
    p.add_argument("--train-log", default="results/train_log.csv")
    p.add_argument("--out", default="results")
    return p.parse_args()


def find_surge_seed(env_cfg, start: int) -> tuple[int, tuple]:
    env = SupportRoutingEnv(env_cfg)
    for seed in range(start, start + 100):
        env.reset(seed=seed)
        if env.surge is not None and env.surge[2] == 1:
            return seed, env.surge
    env.reset(seed=start)
    return start, env.surge


def main():
    args = parse_args()
    out = Path(args.out)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    agent, env_cfg = DQNAgent.load(args.model)
    dqn = DQNPolicy(agent)
    policies = make_baselines() + [dqn]
    names = [p.name for p in policies]

    # 1) Headline benchmark on held-out test seeds at nominal demand.
    seeds = list(range(TEST_SEEDS_START, TEST_SEEDS_START + args.episodes))
    df, routing, _ = evaluate_policies(policies, env_cfg, seeds, collect_routing=True)
    df.to_csv(out / "eval_episodes.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "eval_summary.csv")
    pd.set_option("display.width", 200)
    print(f"\n=== Test benchmark: {args.episodes} shifts x {len(policies)} policies (nominal load) ===")
    print(summary[HEADLINE_COLUMNS].round(3).to_string())
    for name in ("Skill-Aware-SQ", "Category-Rule", "Shortest-Queue", "Round-Robin"):
        gain = 1 - summary.loc["DQN", "total_cost"] / summary.loc[name, "total_cost"]
        print(f"DQN cost reduction vs {name}: {100 * gain:.1f}%")

    # Paired comparison against the strongest baseline on identical seeds.
    paired = (df[df.policy == "Skill-Aware-SQ"].set_index("seed")["total_cost"]
              - df[df.policy == "DQN"].set_index("seed")["total_cost"])
    print(f"DQN beats Skill-Aware-SQ on {int((paired > 0).sum())}/{len(paired)} paired test shifts")

    plotting.plot_policy_comparison(summary, fig_dir)
    plotting.plot_sla_by_priority(summary, env_cfg.priorities, fig_dir)
    plotting.plot_routing_heatmaps(routing, env_cfg.priorities, env_cfg.team_names,
                                   ["Skill-Aware-SQ", "DQN"], fig_dir)

    # 2) Demand sweep (generalisation beyond the nominal load).
    sweep_policies = [p for p in policies if p.name in ("Shortest-Queue", "Category-Rule", "Skill-Aware-SQ", "DQN")]
    sweep_rows = []
    sweep_seeds = list(range(TEST_SEEDS_START + 500, TEST_SEEDS_START + 500 + args.sweep_episodes))
    for load in args.loads:
        s = summarize(evaluate_policies(sweep_policies, env_cfg, sweep_seeds, load_multiplier=load))
        s["load"] = load
        sweep_rows.append(s.reset_index())
        print(f"load {load:.2f}: " + ", ".join(f"{n}={c:.0f}" for n, c in zip(s.index, s["total_cost"])))
    sweep = pd.concat(sweep_rows, ignore_index=True)
    sweep.to_csv(out / "load_sweep.csv", index=False)
    plotting.plot_load_sweep(sweep, fig_dir)

    # 3) Backlog trace on a shift with a technical-incident surge.
    seed, surge = find_surge_seed(env_cfg, TEST_SEEDS_START + 900)
    traces = {}
    for policy in [p for p in policies if p.name in ("Category-Rule", "Skill-Aware-SQ", "DQN")]:
        env = SupportRoutingEnv(env_cfg)
        run_episode(env, policy, seed)
        traces[policy.name] = env.backlog_trace
    plotting.plot_backlog_trace(traces, surge, env_cfg.shift_minutes, fig_dir)

    # 4) Learning curve with baseline reference levels on the validation seeds.
    log_path = Path(args.train_log)
    if log_path.exists():
        log = pd.read_csv(log_path)
        val_seeds = list(range(VALIDATION_SEEDS_START,
                               VALIDATION_SEEDS_START + int(agent.cfg.eval_episodes)))
        ref = summarize(evaluate_policies([CategoryRulePolicy(), SkillAwareShortestQueuePolicy()],
                                          env_cfg, val_seeds))
        plotting.plot_training_curve(log, ref["total_cost"].to_dict(), fig_dir)

    print(f"\nCSV files written to {out}/, figures to {fig_dir}/")


if __name__ == "__main__":
    main()
