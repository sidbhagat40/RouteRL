"""Configuration dataclasses for the RouteRL environment and the DQN learner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    # --- ticket taxonomy -------------------------------------------------
    categories: list[str] = field(
        default_factory=lambda: ["billing", "technical", "account", "logistics"]
    )
    category_mix: list[float] = field(default_factory=lambda: [0.30, 0.35, 0.15, 0.20])
    priorities: list[str] = field(default_factory=lambda: ["P1", "P2", "P3"])
    priority_mix: list[float] = field(default_factory=lambda: [0.15, 0.35, 0.50])
    sla_minutes: list[int] = field(default_factory=lambda: [30, 60, 120])

    # --- teams -----------------------------------------------------------
    team_names: list[str] = field(
        default_factory=lambda: ["Billing", "Technical", "Account", "Logistics", "Tier-1 General"]
    )
    team_agents: list[int] = field(default_factory=lambda: [3, 4, 2, 2, 4])
    # Mean handling time (minutes) of team k for category c; 0 = team lacks the skill.
    handle_time: list[list[float]] = field(
        default_factory=lambda: [
            [6, 0, 0, 0],
            [0, 10, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 6],
            [12, 25, 16, 10],
        ]
    )
    triage_minutes: int = 3          # time a wrong team spends before bouncing a ticket
    work_shape: float = 2.0          # Gamma shape of the per-ticket work multiplier (mean 1)

    # --- agent availability ---------------------------------------------
    break_prob: float = 0.01         # per-minute probability that an idle agent takes a break
    break_mean_minutes: float = 15.0

    # --- demand ----------------------------------------------------------
    shift_minutes: int = 480
    drain_minutes: int = 180         # extra minutes (no arrivals) to clear the backlog
    base_arrival_rate: float = 1.0   # tickets / minute before the daily profile
    peak_amplitude: float = 0.35     # mid-shift peak = base * (1 + amplitude)
    load_multiplier: float = 1.0
    load_randomization: list[float] = field(default_factory=lambda: [0.7, 1.5])
    surge_prob: float = 0.5          # probability of an incident surge in an episode
    surge_minutes: int = 45
    surge_multiplier: float = 2.0

    # --- upstream text classifier (noisy category signal) ----------------
    classifier_noise: float = 1.0
    clarity_range: list[float] = field(default_factory=lambda: [0.5, 3.5])

    # --- ablation switches ----------------------------------------------
    enable_hold: bool = True         # expose the "hold" action
    observe_confidence: bool = True  # False: agent only sees the classifier's top-1 label

    # --- cost / reward ---------------------------------------------------
    wait_cost: list[float] = field(default_factory=lambda: [3.0, 2.0, 1.0])   # per minute
    breach_penalty: list[float] = field(default_factory=lambda: [50.0, 25.0, 10.0])
    misroute_penalty: float = 5.0
    hold_penalty: float = 0.0
    reward_scale: float = 0.05       # reward = -reward_scale * cost (Gym reward and DQN targets)
    # Training signal only (per-ticket mode): charge a ticket for the delay its service
    # imposes on tickets already waiting at that team. Not part of the reported cost.
    externality_weight: float = 1.0


@dataclass
class DQNConfig:
    # "ticket": learn from per-ticket trajectories (decision -> next decision/resolution).
    # "global": learn from the system-wide cost stream with n-step returns.
    credit_assignment: str = "ticket"
    total_steps: int = 300_000
    learning_starts: int = 5_000
    buffer_size: int = 200_000
    batch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99              # per simulated minute (semi-MDP discounting)
    n_step: int = 3
    train_freq: int = 2
    tau: float = 0.005
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_fraction: float = 0.4
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    dueling: bool = True
    double: bool = True
    grad_clip: float = 10.0
    eval_every_episodes: int = 10
    eval_episodes: int = 3
    seed: int = 0
    device: str = "auto"


def _from_dict(cls, data: dict[str, Any] | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None) -> tuple[EnvConfig, DQNConfig]:
    if path is None:
        return EnvConfig(), DQNConfig()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(EnvConfig, raw.get("env")), _from_dict(DQNConfig, raw.get("dqn"))


def env_config_from_dict(data: dict[str, Any]) -> EnvConfig:
    return _from_dict(EnvConfig, data)


def dqn_config_from_dict(data: dict[str, Any]) -> DQNConfig:
    return _from_dict(DQNConfig, data)


def to_dict(cfg) -> dict[str, Any]:
    return asdict(cfg)
