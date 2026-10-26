"""Discrete-time helpdesk simulator in which an RL dispatcher routes support tickets.

Decision epochs are tickets, not minutes: every minute, each ticket waiting at the
dispatcher (new arrivals, previously held tickets, tickets bounced back by a wrong
team) is presented to the agent one at a time. Once the last pending ticket of the
minute is decided, the clock advances and costs for that minute are charged to the
decision that triggered the advance.
"""
from __future__ import annotations

import heapq
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig

IDLE, BUSY, AWAY = 0, 1, 2

TICKET_FEATURES_EXTRA = 7   # priority one-hot (3) + slack + retagged + holds + misroutes
TEAM_FEATURES = 5
GLOBAL_FEATURES = 3


class Ticket:
    __slots__ = (
        "tid", "arrival", "category", "priority", "probs", "work", "deadline",
        "retagged", "holds", "misroutes", "breached", "resolved_at", "team",
        "last_obs", "last_action", "last_t", "acc_cost",
    )

    def __init__(self, tid, arrival, category, priority, probs, work, deadline):
        self.tid = tid
        self.arrival = arrival
        self.category = category
        self.priority = priority
        self.probs = probs
        self.work = work
        self.deadline = deadline
        self.retagged = False
        self.holds = 0
        self.misroutes = 0
        self.breached = False
        self.resolved_at = -1
        self.team = -1
        # Per-ticket credit assignment: observation/action at the latest decision and
        # the cost this ticket has accrued since then.
        self.last_obs = None
        self.last_action = -1
        self.last_t = arrival
        self.acc_cost = 0.0

    @property
    def predicted_category(self) -> int:
        return int(np.argmax(self.probs))


class Team:
    def __init__(self, idx: int, name: str, n_agents: int, handle_time: list[float], n_priorities: int):
        self.idx = idx
        self.name = name
        self.n = n_agents
        self.handle_time = [float(h) for h in handle_time]
        self.queue: list = []                       # heap of (priority, arrival, tid, ticket)
        self.qcount = [0] * n_priorities
        self.state = [IDLE] * n_agents
        self.remaining = [0] * n_agents
        self.current: list[Ticket | None] = [None] * n_agents
        self.break_left = [0] * n_agents
        self.busy_minutes = 0

    def can_handle(self, category: int) -> bool:
        return self.handle_time[category] > 0

    @property
    def queue_len(self) -> int:
        return len(self.queue)

    @property
    def idle(self) -> int:
        return self.state.count(IDLE)

    @property
    def busy(self) -> int:
        return self.state.count(BUSY)

    @property
    def available(self) -> int:
        return self.n - self.state.count(AWAY)

    def ahead_of(self, priority: int) -> int:
        return sum(self.qcount[: priority + 1])

    def next_free_in(self) -> int:
        if IDLE in self.state:
            return 0
        busy = [r for r, s in zip(self.remaining, self.state) if s == BUSY]
        return min(busy) if busy else 0


class SupportRoutingEnv(gym.Env):
    """Gymnasium environment: route the ticket at the head of the dispatcher or hold it.

    Actions 0..K-1 send the ticket to team k; action K holds it for one minute. A ticket
    can be held at most ``max_holds`` times; ``action_mask()`` reports the legal actions.

    The Gym reward is the (scaled) system-wide cost of the step. With
    ``track_tickets=True`` the env additionally reports per-ticket transitions in
    ``info["ticket_transitions"]`` as tuples
    ``(obs, action, cost, next_obs | None, next_mask | None, elapsed_minutes, done)``: each ticket's
    journey from one decision to its next decision (after a hold or a bounce) or to
    its resolution. Every cost term belongs to exactly one ticket, so these costs sum
    to the episode's total cost. With ``cost_discount=gamma < 1`` each cost term in a
    transition is discounted by gamma**(minutes since the ticket's decision), matching
    the gamma**elapsed bootstrap: otherwise a long route-to-resolution segment would be
    charged undiscounted while the same wait split by holds would be discounted, making
    holding look cheaper than routing.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, randomize_load: bool = False,
                 track_tickets: bool = False, cost_discount: float = 1.0):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.track_tickets = track_tickets
        self.cost_discount = cost_discount
        c = self.cfg
        self.n_categories = len(c.categories)
        self.n_priorities = len(c.priorities)
        self.n_teams = len(c.team_names)
        assert len(c.handle_time) == self.n_teams and all(len(r) == self.n_categories for r in c.handle_time)
        assert len(c.team_agents) == self.n_teams
        self.randomize_load = randomize_load
        self.hold_action = self.n_teams if c.enable_hold else None

        self.obs_dim = (
            self.n_categories + TICKET_FEATURES_EXTRA
            + TEAM_FEATURES * self.n_teams + GLOBAL_FEATURES
        )
        self.action_space = spaces.Discrete(self.n_teams + (1 if c.enable_hold else 0))
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.obs_dim,), dtype=np.float32)

        self._wait_cost = list(map(float, c.wait_cost))
        self._breach_penalty = list(map(float, c.breach_penalty))
        self._break_return_prob = 1.0 / max(c.break_mean_minutes, 1.0)
        self.current_ticket: Ticket | None = None

    # ------------------------------------------------------------------ setup
    def _generate_stream(self, rng: np.random.Generator) -> list[list[Ticket]]:
        c = self.cfg
        T = c.shift_minutes
        mult = c.load_multiplier
        if self.randomize_load:
            mult *= rng.uniform(*c.load_randomization)
        self.episode_load = mult

        minutes = np.arange(T)
        profile = 1.0 + c.peak_amplitude * np.sin(np.pi * (minutes + 0.5) / T)
        rates = c.base_arrival_rate * mult * profile[:, None] * np.asarray(c.category_mix)[None, :]
        self.surge = None
        if rng.random() < c.surge_prob:
            start = int(rng.integers(int(0.1 * T), int(0.8 * T)))
            cat = int(rng.integers(self.n_categories))
            rates[start:start + c.surge_minutes, cat] *= c.surge_multiplier
            self.surge = (start, start + c.surge_minutes, cat)

        counts = rng.poisson(rates)                                    # T x C
        arrival_minute = np.repeat(np.repeat(minutes, self.n_categories), counts.ravel())
        category = np.repeat(np.tile(np.arange(self.n_categories), T), counts.ravel())
        n = len(category)
        priority = rng.choice(self.n_priorities, size=n, p=c.priority_mix)
        clarity = rng.uniform(*c.clarity_range, size=n)
        logits = rng.normal(0.0, c.classifier_noise, size=(n, self.n_categories))
        logits[np.arange(n), category] += clarity
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        work = rng.gamma(c.work_shape, 1.0 / c.work_shape, size=n)

        stream: list[list[Ticket]] = [[] for _ in range(T)]
        for i in range(n):
            p = int(priority[i])
            t = int(arrival_minute[i])
            stream[t].append(Ticket(i, t, int(category[i]), p, probs[i].astype(np.float32),
                                    float(work[i]), t + c.sla_minutes[p]))
        return stream

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2**31 - 1))
        arrivals_ss, ops_ss = np.random.SeedSequence(seed).spawn(2)
        self._stream = self._generate_stream(np.random.default_rng(arrivals_ss))
        self._rng = np.random.default_rng(ops_ss)

        c = self.cfg
        self.t = 0
        self.teams = [
            Team(k, c.team_names[k], c.team_agents[k], c.handle_time[k], self.n_priorities)
            for k in range(self.n_teams)
        ]
        self.pending: list[Ticket] = []
        self.held: list[Ticket] = []
        self.in_system = [0] * self.n_priorities
        self._sla_heap: list = []
        self.n_arrived = 0
        self.resolved: list[Ticket] = []
        self.arrival_ema = 0.0
        self.cost_totals = {"wait": 0.0, "breach": 0.0, "misroute": 0.0, "hold": 0.0}
        self.backlog_trace: list[int] = []
        self.decisions = 0
        self.truncated_unresolved = 0
        self._step_cost = 0.0
        self._transitions: list[tuple] = []
        self._current_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._done = False
        self._terminated = False

        self._admit_arrivals()
        self._advance_to_next_decision()
        return self._current_obs, {}

    # ------------------------------------------------------------ dynamics
    def _charge(self, key: str, amount: float) -> None:
        self.cost_totals[key] += amount
        self._step_cost += amount

    def _accrue(self, tk: Ticket, amount: float) -> None:
        """Add a cost term to the ticket's current transition, discounted to its decision."""
        tk.acc_cost += amount * self.cost_discount ** (self.t - tk.last_t)

    def _close_ticket_segment(self, tk: Ticket, next_obs, done: bool) -> None:
        """Emit the transition from the ticket's last decision to now."""
        if not self.track_tickets or tk.last_obs is None:
            return
        elapsed = self.t - tk.last_t
        next_mask = None if next_obs is None else self.action_mask()
        g = self.cost_discount
        # Waiting in minute k after the decision is weighted by g**k.
        wait_minutes = elapsed if g == 1.0 else (1.0 - g ** elapsed) / (1.0 - g)
        cost = tk.acc_cost + self._wait_cost[tk.priority] * wait_minutes
        self._transitions.append((tk.last_obs, tk.last_action, cost, next_obs, next_mask, elapsed, done))
        tk.last_obs = None
        tk.acc_cost = 0.0

    def _admit_arrivals(self) -> None:
        if self.t < self.cfg.shift_minutes:
            new = self._stream[self.t]
            for tk in new:
                heapq.heappush(self._sla_heap, (tk.deadline, tk.tid, tk))
                self.in_system[tk.priority] += 1
            self.pending.extend(new)
            self.n_arrived += len(new)
            self.arrival_ema = 0.9 * self.arrival_ema + 0.1 * len(new)
        else:
            self.arrival_ema *= 0.9
        if self.held:
            self.pending.extend(self.held)
            self.held = []

    def _start_service(self) -> None:
        triage = self.cfg.triage_minutes
        ext_weight = self.cfg.externality_weight if self.track_tickets else 0.0
        for team in self.teams:
            if not team.queue:
                continue
            for i in range(team.n):
                if team.state[i] != IDLE or not team.queue:
                    continue
                _, _, _, tk = heapq.heappop(team.queue)
                team.qcount[tk.priority] -= 1
                h = team.handle_time[tk.category]
                minutes = max(1, int(round(tk.work * h))) if h > 0 else triage
                team.remaining[i] = minutes
                team.current[i] = tk
                team.state[i] = BUSY
                if ext_weight:
                    # Occupying one of n agents for `minutes` delays each waiting ticket
                    # by about minutes / n (marginal-cost / Pigouvian credit assignment).
                    waiting_weight = sum(q * w for q, w in zip(team.qcount, self._wait_cost))
                    self._accrue(tk, ext_weight * minutes / team.n * waiting_weight)

    def _tick(self) -> None:
        self._start_service()
        self.t += 1

        # Waiting cost for every unresolved ticket, then SLA breaches.
        self._charge("wait", sum(n * w for n, w in zip(self.in_system, self._wait_cost)))
        heap = self._sla_heap
        while heap and heap[0][0] < self.t:
            _, _, tk = heapq.heappop(heap)
            if tk.resolved_at < 0 and not tk.breached:
                tk.breached = True
                self._charge("breach", self._breach_penalty[tk.priority])
                self._accrue(tk, self._breach_penalty[tk.priority])

        # Service progress, completions and bounces.
        rng = self._rng
        for team in self.teams:
            for i in range(team.n):
                s = team.state[i]
                if s == BUSY:
                    team.busy_minutes += 1
                    team.remaining[i] -= 1
                    if team.remaining[i] <= 0:
                        tk = team.current[i]
                        team.current[i] = None
                        team.state[i] = IDLE
                        if team.can_handle(tk.category):
                            tk.resolved_at = self.t
                            self.in_system[tk.priority] -= 1
                            self.resolved.append(tk)
                            self._close_ticket_segment(tk, None, True)
                        else:
                            tk.retagged = True
                            tk.team = -1
                            onehot = np.zeros(self.n_categories, dtype=np.float32)
                            onehot[tk.category] = 1.0
                            tk.probs = onehot
                            self.pending.append(tk)
                elif s == IDLE:
                    if rng.random() < self.cfg.break_prob:
                        team.state[i] = AWAY
                else:
                    if rng.random() < self._break_return_prob:
                        team.state[i] = IDLE

        self._admit_arrivals()
        self.backlog_trace.append(sum(self.in_system))

    def _advance_to_next_decision(self) -> None:
        c = self.cfg
        while not self.pending:
            if self.t >= c.shift_minutes and sum(self.in_system) == 0:
                self._done, self._terminated = True, True
                self.current_ticket = None
                self._current_obs = np.zeros(self.obs_dim, dtype=np.float32)
                return
            if self.t >= c.shift_minutes + c.drain_minutes:
                self._truncate()
                return
            self._tick()
        self._pop_next_pending()

    def _pop_next_pending(self) -> None:
        pending = self.pending
        best = min(range(len(pending)),
                   key=lambda j: (pending[j].priority, pending[j].arrival, pending[j].tid))
        tk = pending.pop(best)
        self.current_ticket = tk
        self._current_obs = self._observe()
        self._close_ticket_segment(tk, self._current_obs, False)

    def _truncate(self) -> None:
        # Unresolved tickets at the hard horizon are charged as SLA breaches.
        for tk in (tk for minute in self._stream for tk in minute):
            if tk.resolved_at >= 0:
                continue
            if not tk.breached:
                tk.breached = True
                self._charge("breach", self._breach_penalty[tk.priority])
                self._accrue(tk, self._breach_penalty[tk.priority])
            self._close_ticket_segment(tk, None, True)
        self.truncated_unresolved = sum(self.in_system)
        self._done, self._terminated = True, False
        self.current_ticket = None
        self._current_obs = np.zeros(self.obs_dim, dtype=np.float32)

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        action = int(action)
        tk = self.current_ticket
        t_before = self.t
        self._step_cost = 0.0
        self._transitions = []
        self.decisions += 1
        if self.track_tickets:
            tk.last_obs, tk.last_action, tk.last_t, tk.acc_cost = self._current_obs, action, self.t, 0.0

        if action == self.hold_action:
            if tk.holds >= self.cfg.max_holds:
                raise ValueError(f"ticket already held {tk.holds} times (max_holds); route it")
            tk.holds += 1
            if self.cfg.hold_penalty:
                self._charge("hold", self.cfg.hold_penalty)
                self._accrue(tk, self.cfg.hold_penalty)
            self.held.append(tk)
        else:
            team = self.teams[action]
            if not team.can_handle(tk.category):
                tk.misroutes += 1
                self._charge("misroute", self.cfg.misroute_penalty)
                self._accrue(tk, self.cfg.misroute_penalty)
            tk.team = action
            heapq.heappush(team.queue, (tk.priority, tk.arrival, tk.tid, tk))
            team.qcount[tk.priority] += 1

        self.current_ticket = None
        self._advance_to_next_decision()

        reward = -self.cfg.reward_scale * self._step_cost
        # Minutes elapsed during this step; learners discount by gamma**elapsed (semi-MDP).
        info = {"cost": self._step_cost, "elapsed": self.t - t_before}
        if self.track_tickets:
            info["ticket_transitions"] = self._transitions
        terminated = self._done and self._terminated
        truncated = self._done and not self._terminated
        if self._done:
            info["episode_summary"] = self.summary()
        return self._current_obs, reward, terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        """Legal actions for the current ticket: every team, and hold while under max_holds."""
        mask = np.ones(self.action_space.n, dtype=bool)
        if self.hold_action is not None:
            tk = self.current_ticket
            mask[self.hold_action] = tk is not None and tk.holds < self.cfg.max_holds
        return mask

    # --------------------------------------------------------- observation
    def _observe(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        tk = self.current_ticket
        if tk is None:
            return obs
        C, P = self.n_categories, self.n_priorities
        if self.cfg.observe_confidence:
            obs[:C] = tk.probs
        else:
            obs[int(np.argmax(tk.probs))] = 1.0
        obs[C + tk.priority] = 1.0
        i = C + P
        sla = self.cfg.sla_minutes[tk.priority]
        obs[i] = max(-1.0, min(1.0, (tk.deadline - self.t) / sla))
        obs[i + 1] = 1.0 if tk.retagged else 0.0
        obs[i + 2] = min(tk.holds / max(self.cfg.max_holds, 1), 1.0)
        obs[i + 3] = min(tk.misroutes / 3.0, 1.0)
        i += 4
        for team in self.teams:
            n = float(team.n)
            obs[i] = min(team.queue_len / 10.0, 5.0)
            obs[i + 1] = min(team.ahead_of(tk.priority) / 10.0, 5.0)
            obs[i + 2] = team.idle / n
            obs[i + 3] = team.available / n
            obs[i + 4] = min(team.next_free_in() / 30.0, 1.0)
            i += TEAM_FEATURES
        obs[i] = min(self.t / self.cfg.shift_minutes, 1.5)
        obs[i + 1] = min(len(self.pending) / 10.0, 5.0)
        obs[i + 2] = min(self.arrival_ema / 2.0, 5.0)
        return obs

    # ------------------------------------------------------------ metrics
    def summary(self) -> dict:
        c = self.cfg
        n = max(self.n_arrived, 1)
        resolved = self.resolved
        times = np.array([tk.resolved_at - tk.arrival for tk in resolved], dtype=float)
        all_tickets = [tk for minute in self._stream for tk in minute]
        out = {
            "total_cost": sum(self.cost_totals.values()),
            **{f"cost_{k}": v for k, v in self.cost_totals.items()},
            "tickets": self.n_arrived,
            "resolved": len(resolved),
            "unresolved": self.truncated_unresolved,
            "mean_resolution_min": float(times.mean()) if len(times) else math.nan,
            "p90_resolution_min": float(np.percentile(times, 90)) if len(times) else math.nan,
            "sla_breach_rate": sum(tk.breached for tk in all_tickets) / n,
            "misroutes_per_ticket": sum(tk.misroutes for tk in all_tickets) / n,
            "first_touch_correct": sum(tk.misroutes == 0 for tk in all_tickets) / n,
            "holds_per_ticket": sum(tk.holds for tk in all_tickets) / n,
            "decisions": self.decisions,
            "episode_minutes": self.t,
            "load_multiplier": getattr(self, "episode_load", c.load_multiplier),
        }
        for p, name in enumerate(c.priorities):
            group = [tk for tk in all_tickets if tk.priority == p]
            out[f"sla_breach_rate_{name}"] = (
                sum(tk.breached for tk in group) / len(group) if group else math.nan
            )
        for team in self.teams:
            out[f"util_{team.name}"] = team.busy_minutes / max(team.n * self.t, 1)
        return out
