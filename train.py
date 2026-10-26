"""Train the RouteRL DQN dispatcher.

Example:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from routerl.config import load_config
from routerl.dqn import DQNAgent, DQNPolicy, NStepAccumulator, ReplayBuffer
from routerl.env import SupportRoutingEnv
from routerl.evaluation import run_episode

VALIDATION_SEEDS_START = 1_000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--total-steps", type=int, default=None, help="override dqn.total_steps")
    p.add_argument("--seed", type=int, default=None, help="override dqn.seed")
    p.add_argument("--model-out", default="models/dqn_routerl.pt")
    p.add_argument("--log-out", default="results/train_log.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate(agent: DQNAgent, env_cfg, n_episodes: int, loads: list[float]) -> dict:
    """Nominal-demand metrics plus ``score``: the geometric mean of the mean validation cost
    over ``loads``, so model selection rewards checkpoints that also hold up under overload."""
    policy = DQNPolicy(agent)
    metrics, log_costs = {}, []
    for load in loads:
        env = SupportRoutingEnv(replace(env_cfg, load_multiplier=load))
        rows = [run_episode(env, policy, VALIDATION_SEEDS_START + i) for i in range(n_episodes)]
        log_costs.append(np.log(np.mean([r["total_cost"] for r in rows])))
        if load == env_cfg.load_multiplier:
            metrics = {k: float(np.mean([r[k] for r in rows])) for k in
                       ("total_cost", "sla_breach_rate", "misroutes_per_ticket", "mean_resolution_min")}
    metrics["score"] = float(np.exp(np.mean(log_costs)))
    return metrics


def main():
    args = parse_args()
    env_cfg, dqn_cfg = load_config(args.config)
    overrides = {k: v for k, v in {"total_steps": args.total_steps, "seed": args.seed,
                                   "device": args.device}.items() if v is not None}
    dqn_cfg = replace(dqn_cfg, **overrides)
    set_seed(dqn_cfg.seed)
    torch.set_num_threads(max(1, args.threads))

    mode = dqn_cfg.credit_assignment
    if mode not in ("ticket", "global"):
        raise ValueError(f"credit_assignment must be 'ticket' or 'global', got {mode!r}")
    env = SupportRoutingEnv(env_cfg, randomize_load=True, track_tickets=(mode == "ticket"),
                            cost_discount=dqn_cfg.gamma)
    agent = DQNAgent(env.obs_dim, env.action_space.n, dqn_cfg)
    buffer = ReplayBuffer(dqn_cfg.buffer_size, env.obs_dim, env.action_space.n)
    nstep = NStepAccumulator(dqn_cfg.n_step, dqn_cfg.gamma)
    terminal_obs = np.zeros(env.obs_dim, dtype=np.float32)
    all_legal = np.ones(env.action_space.n, dtype=bool)
    scale, gamma = env_cfg.reward_scale, dqn_cfg.gamma
    print(f"obs_dim={env.obs_dim} actions={env.action_space.n} device={agent.device} "
          f"credit_assignment={mode}")

    Path(args.log_out).parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(args.log_out, "w", newline="", encoding="utf-8")
    fields = ["episode", "step", "epsilon", "train_cost", "train_return", "train_sla_breach_rate",
              "train_misroutes_per_ticket", "train_load", "loss", "val_cost", "val_sla_breach_rate",
              "val_misroutes_per_ticket", "val_mean_resolution_min", "val_score", "elapsed_s"]
    writer = csv.DictWriter(log_fh, fieldnames=fields)
    writer.writeheader()

    decay_steps = max(1, int(dqn_cfg.eps_decay_fraction * dqn_cfg.total_steps))
    epsilon_at = lambda s: max(dqn_cfg.eps_end,
                               dqn_cfg.eps_start + (dqn_cfg.eps_end - dqn_cfg.eps_start) * s / decay_steps)

    episode, ep_return, losses = 0, 0.0, []
    best_val = float("inf")
    start = time.time()
    obs, _ = env.reset(seed=dqn_cfg.seed * 100_000 + episode)

    for step in range(1, dqn_cfg.total_steps + 1):
        eps = epsilon_at(step)
        action = agent.act(obs, eps, env.action_mask())
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += reward
        if mode == "ticket":
            for t_obs, t_act, cost, t_next, t_mask, elapsed, t_done in info["ticket_transitions"]:
                buffer.add(t_obs, t_act, -scale * cost, terminal_obs if t_next is None else t_next,
                           0.0 if t_done else gamma ** elapsed, all_legal if t_mask is None else t_mask)
        else:
            next_mask = all_legal if done else env.action_mask()
            for transition in nstep.push(obs, action, -scale * info["cost"], next_obs, done,
                                         info["elapsed"]):
                buffer.add(*transition, next_mask)
        obs = next_obs

        if step >= dqn_cfg.learning_starts and step % dqn_cfg.train_freq == 0:
            losses.append(agent.update(buffer))

        if done:
            summary = info["episode_summary"]
            row = {
                "episode": episode, "step": step, "epsilon": round(eps, 4),
                "train_cost": summary["total_cost"], "train_return": ep_return,
                "train_sla_breach_rate": summary["sla_breach_rate"],
                "train_misroutes_per_ticket": summary["misroutes_per_ticket"],
                "train_load": summary["load_multiplier"],
                "loss": float(np.mean(losses)) if losses else "",
                "elapsed_s": round(time.time() - start, 1),
            }
            if (episode + 1) % dqn_cfg.eval_every_episodes == 0 and step >= dqn_cfg.learning_starts:
                val = validate(agent, env_cfg, dqn_cfg.eval_episodes, dqn_cfg.val_loads)
                row.update({f"val_{k}": v for k, v in val.items()})
                row["val_cost"] = val["total_cost"]
                row.pop("val_total_cost", None)
                tag = ""
                if val["score"] < best_val:
                    best_val = val["score"]
                    agent.save(args.model_out, env_cfg, {"step": step, "episode": episode, "val": val})
                    tag = "  * best"
                print(f"ep {episode:4d} step {step:7d} eps {eps:.3f} train_cost {summary['total_cost']:9.0f} "
                      f"val_cost {val['total_cost']:9.0f} score {val['score']:9.0f} breach {val['sla_breach_rate']:.3f} "
                      f"misroute {val['misroutes_per_ticket']:.3f} ({row['elapsed_s']}s){tag}", flush=True)
            writer.writerow(row)
            log_fh.flush()
            losses = []
            episode += 1
            ep_return = 0.0
            obs, _ = env.reset(seed=dqn_cfg.seed * 100_000 + episode)

    final_path = Path(args.model_out).with_name(Path(args.model_out).stem + "_final.pt")
    agent.save(final_path, env_cfg, {"step": dqn_cfg.total_steps, "episode": episode})
    log_fh.close()
    print(f"done in {time.time() - start:.0f}s; best val score {best_val:.0f}; "
          f"best model -> {args.model_out}, final model -> {final_path}")


if __name__ == "__main__":
    main()
