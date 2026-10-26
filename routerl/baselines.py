"""Non-learning dispatch policies used as baselines.

Every policy is a callable ``policy(obs, env) -> action``; heuristics read the
simulator state directly through ``env`` while the DQN policy only uses ``obs``.
"""
from __future__ import annotations

import numpy as np

from .env import SupportRoutingEnv


class Policy:
    name = "policy"

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray, env: SupportRoutingEnv) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, obs, env):
        return int(self.rng.integers(env.n_teams))


class RoundRobinPolicy(Policy):
    """Cycle through teams regardless of ticket content (category-blind)."""

    name = "Round-Robin"

    def __init__(self):
        self.next_team = 0

    def reset(self) -> None:
        self.next_team = 0

    def __call__(self, obs, env):
        k = self.next_team
        self.next_team = (k + 1) % env.n_teams
        return k


def _load(team) -> float:
    return team.queue_len - team.idle


class ShortestQueuePolicy(Policy):
    """Join-the-shortest-queue over all teams (category-blind)."""

    name = "Shortest-Queue"

    def __call__(self, obs, env):
        return int(min(range(env.n_teams), key=lambda k: (_load(env.teams[k]), k)))


class CategoryRulePolicy(Policy):
    """Static rule: send the ticket to the specialist of the classifier's top category."""

    name = "Category-Rule"

    def __call__(self, obs, env):
        c = env.current_ticket.predicted_category
        specialists = [k for k, t in enumerate(env.teams) if t.can_handle(c)]
        return min(specialists, key=lambda k: env.teams[k].handle_time[c])


class SkillAwareShortestQueuePolicy(Policy):
    """Least-loaded team (queue + busy agents per available agent) among the
    teams able to handle the predicted category."""

    name = "Skill-Aware-SQ"

    def __call__(self, obs, env):
        c = env.current_ticket.predicted_category
        candidates = [k for k, t in enumerate(env.teams) if t.can_handle(c)]

        def key(k):
            t = env.teams[k]
            return ((t.queue_len + t.busy + 1) / max(t.available, 1), t.handle_time[c], k)

        return int(min(candidates, key=key))


def make_baselines(seed: int = 0) -> list[Policy]:
    return [
        RandomPolicy(seed),
        RoundRobinPolicy(),
        ShortestQueuePolicy(),
        CategoryRulePolicy(),
        SkillAwareShortestQueuePolicy(),
    ]
