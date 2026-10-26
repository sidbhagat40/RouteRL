"""Deep Q-Network (Double + Dueling + n-step returns) implemented from scratch in PyTorch."""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DQNConfig, EnvConfig, dqn_config_from_dict, env_config_from_dict


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        # Small MLPs train faster on CPU than with per-step host/GPU transfers.
        return torch.device("cpu")
    return torch.device(name)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int], dueling: bool = True):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(d, 1)
            self.advantage = nn.Linear(d, n_actions)
        else:
            self.head = nn.Linear(d, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        if not self.dueling:
            return self.head(z)
        a = self.advantage(z)
        return self.value(z) + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Uniform replay of n-step transitions (s, a, R^(n), s', bootstrap discount)."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, ret, next_obs, discount) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.returns[i] = ret
        self.next_obs[i] = next_obs
        self.discounts[i] = discount
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, device: torch.device):
        idx = rng.integers(0, self.size, size=batch_size)
        as_t = lambda a: torch.as_tensor(a[idx], device=device)
        return (as_t(self.obs), as_t(self.actions), as_t(self.returns),
                as_t(self.next_obs), as_t(self.discounts))


class NStepAccumulator:
    """Turns a stream of 1-step transitions into n-step transitions.

    Decisions are irregularly spaced in time, so discounting is per elapsed minute
    (semi-MDP): reward r_i is weighted by gamma**(minutes before step i), and the
    bootstrap term by gamma**(minutes spanned by the n steps). Discounting per decision
    instead would let the agent "buy" discount by generating extra hold decisions.
    """

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.window: deque = deque()

    def _emit_front(self, next_obs, done: bool):
        ret, g = 0.0, 1.0
        for _, _, r, elapsed in self.window:
            ret += g * r
            g *= self.gamma ** elapsed
        obs, action, _, _ = self.window.popleft()
        return obs, action, ret, next_obs, 0.0 if done else g

    def push(self, obs, action, reward, next_obs, done: bool, elapsed: int = 1) -> list[tuple]:
        self.window.append((obs, action, reward, elapsed))
        out = []
        if done:
            while self.window:
                out.append(self._emit_front(next_obs, True))
        elif len(self.window) == self.n:
            out.append(self._emit_front(next_obs, False))
        return out


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device: torch.device | None = None):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device or resolve_device(cfg.device)
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden, cfg.dueling).to(self.device)
        self.target = copy.deepcopy(self.q)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        self.updates = 0

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.q(x)[0].cpu().numpy()

    def greedy(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.q_values(obs)))

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)

    def update(self, buffer: ReplayBuffer) -> float:
        obs, actions, returns, next_obs, discounts = buffer.sample(self.cfg.batch_size, self.rng, self.device)
        q = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.double:
                next_a = self.q(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_a).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=1).values
            target = returns + discounts * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        with torch.no_grad():
            tau = self.cfg.tau
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.mul_(1.0 - tau).add_(p, alpha=tau)
        self.updates += 1
        return float(loss.item())

    def save(self, path: str | Path, env_cfg: EnvConfig, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.q.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "dqn_cfg": asdict(self.cfg),
            "env_cfg": asdict(env_cfg),
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> tuple["DQNAgent", EnvConfig]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = dqn_config_from_dict(ckpt["dqn_cfg"])
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], cfg, torch.device(device))
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.target.load_state_dict(ckpt["state_dict"])
        agent.q.eval()
        return agent, env_config_from_dict(ckpt["env_cfg"])


class DQNPolicy:
    """Greedy policy wrapper with the same interface as the baselines."""

    def __init__(self, agent: DQNAgent, name: str = "DQN"):
        self.agent = agent
        self.name = name

    def reset(self) -> None:
        pass

    def __call__(self, obs, env) -> int:
        return self.agent.greedy(obs)
