import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from routerl.baselines import CategoryRulePolicy, SkillAwareShortestQueuePolicy, make_baselines
from routerl.config import EnvConfig, load_config
from routerl.dqn import NStepAccumulator
from routerl.env import SupportRoutingEnv
from routerl.evaluation import run_episode


@pytest.fixture
def env():
    return SupportRoutingEnv(EnvConfig())


def test_gymnasium_api(env):
    check_env(env, skip_render_check=True)


def test_default_yaml_matches_dataclass_defaults():
    env_cfg, _ = load_config("configs/default.yaml")
    assert env_cfg == EnvConfig()


def test_observation_shape_and_bounds(env):
    obs, _ = env.reset(seed=3)
    assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32
    for _ in range(500):
        obs, _, term, trunc, _ = env.step(env.action_space.sample(mask=env.action_mask().astype(np.int8)))
        assert env.observation_space.contains(obs)
        if term or trunc:
            break


@pytest.mark.parametrize("policy", make_baselines(), ids=lambda p: p.name)
def test_every_ticket_is_accounted_for(policy):
    env = SupportRoutingEnv(EnvConfig())
    s = run_episode(env, policy, seed=11)
    assert s["resolved"] + s["unresolved"] == s["tickets"]
    assert abs(s["total_cost"] + s["return"] / env.cfg.reward_scale) < 1e-6 * max(1.0, s["total_cost"])


def _ticket_cost_sum(env, policy, seed):
    policy.reset()
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        done = term or trunc
        total += sum(tr[2] for tr in info["ticket_transitions"])
    return total, info["episode_summary"]


def test_externality_term_only_affects_training_signal():
    policy = SkillAwareShortestQueuePolicy()
    with_ext, s1 = _ticket_cost_sum(SupportRoutingEnv(EnvConfig(externality_weight=1.0), track_tickets=True), policy, 6)
    no_ext, s0 = _ticket_cost_sum(SupportRoutingEnv(EnvConfig(externality_weight=0.0), track_tickets=True), policy, 6)
    assert s1["total_cost"] == s0["total_cost"] == pytest.approx(no_ext)
    assert with_ext > no_ext


@pytest.mark.parametrize("policy", make_baselines(), ids=lambda p: p.name)
def test_ticket_transitions_decompose_total_cost(policy):
    env = SupportRoutingEnv(EnvConfig(externality_weight=0.0), track_tickets=True)
    policy.reset()
    obs, _ = env.reset(seed=4)
    n_transitions, ticket_cost, done = 0, 0.0, False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        done = term or trunc
        for t_obs, t_act, cost, t_next, t_mask, elapsed, t_done in info["ticket_transitions"]:
            assert t_obs.shape == obs.shape and elapsed >= 0 and cost >= 0
            assert (t_next is None) == t_done == (t_mask is None)
            ticket_cost += cost
            n_transitions += 1
    summary = info["episode_summary"]
    assert n_transitions == summary["decisions"]
    assert ticket_cost == pytest.approx(summary["total_cost"])


def test_ticket_transition_wait_is_discounted_per_minute():
    gamma = 0.95
    cfg = EnvConfig(externality_weight=0.0, misroute_penalty=0.0, breach_penalty=[0.0, 0.0, 0.0])
    env = SupportRoutingEnv(cfg, track_tickets=True, cost_discount=gamma)
    policy = SkillAwareShortestQueuePolicy()
    obs, _ = env.reset(seed=4)
    C, done, checked = env.n_categories, False, 0
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        done = term or trunc
        for t_obs, _, cost, _, _, elapsed, _ in info["ticket_transitions"]:
            w = cfg.wait_cost[int(np.argmax(t_obs[C:C + 3]))]
            assert cost == pytest.approx(w * (1 - gamma ** elapsed) / (1 - gamma))
            checked += 1
    assert checked > 0


def test_same_seed_same_tickets_for_all_policies():
    env_a, env_b = SupportRoutingEnv(EnvConfig()), SupportRoutingEnv(EnvConfig())
    env_a.reset(seed=5)
    env_b.reset(seed=5)
    a = [(t.arrival, t.category, t.priority, t.work) for m in env_a._stream for t in m]
    b = [(t.arrival, t.category, t.priority, t.work) for m in env_b._stream for t in m]
    assert a == b


def test_episode_is_deterministic_given_seed():
    p = SkillAwareShortestQueuePolicy()
    r1 = run_episode(SupportRoutingEnv(EnvConfig()), p, seed=9)
    r2 = run_episode(SupportRoutingEnv(EnvConfig()), p, seed=9)
    assert r1["total_cost"] == r2["total_cost"]


def test_misroute_is_penalised_and_ticket_bounces(env):
    env.reset(seed=0)
    tk = env.current_ticket
    wrong = next(k for k, t in enumerate(env.teams[:4]) if not t.can_handle(tk.category))
    _, reward, *_ = env.step(wrong)
    assert env.cost_totals["misroute"] == env.cfg.misroute_penalty
    assert tk.misroutes == 1
    while not tk.retagged:
        env.step(env.hold_action if env.current_ticket is not tk else 4)
    assert tk.probs[tk.category] == 1.0


def test_hold_returns_ticket_next_minute(env):
    env.reset(seed=1)
    tk, t0 = env.current_ticket, env.t
    env.step(env.hold_action)
    while env.current_ticket is not tk:
        assert env.current_ticket is not None
        env.step(CategoryRulePolicy()(None, env))
    assert env.t == t0 + 1 and tk.holds == 1


def test_hold_is_capped_and_masked(env):
    env.reset(seed=1)
    tk = env.current_ticket
    for _ in range(env.cfg.max_holds):
        assert env.action_mask()[env.hold_action]
        env.step(env.hold_action)
        while env.current_ticket is not tk:
            env.step(CategoryRulePolicy()(None, env))
    assert tk.holds == env.cfg.max_holds
    mask = env.action_mask()
    assert not mask[env.hold_action] and mask[:env.n_teams].all()
    with pytest.raises(ValueError):
        env.step(env.hold_action)


def test_dqn_never_picks_masked_action():
    from routerl.config import DQNConfig
    from routerl.dqn import DQNAgent
    agent = DQNAgent(4, 3, DQNConfig(hidden=[8]))
    with torch.no_grad():
        agent.q.value.bias.fill_(0.0)
        agent.q.advantage.bias.copy_(torch.tensor([0.0, 0.0, 100.0]))
    mask = np.array([True, True, False])
    obs = np.zeros(4, dtype=np.float32)
    assert agent.greedy(obs, mask) != 2
    assert all(agent.act(obs, 1.0, mask) != 2 for _ in range(50))


def test_nstep_returns():
    acc = NStepAccumulator(n=3, gamma=0.5)
    out = []
    for i, r in enumerate([1.0, 2.0, 4.0, 8.0]):
        out += acc.push(i, 0, r, i + 1, done=(i == 3))
    assert [o[0] for o in out] == [0, 1, 2, 3]
    assert out[0][2] == pytest.approx(1 + 0.5 * 2 + 0.25 * 4) and out[0][4] == pytest.approx(0.125)
    assert out[1][2] == pytest.approx(2 + 0.5 * 4 + 0.25 * 8) and out[1][4] == 0.0
    assert out[3][2] == pytest.approx(8.0) and out[3][4] == 0.0


def test_nstep_discounts_by_elapsed_minutes_not_decisions():
    acc = NStepAccumulator(n=2, gamma=0.5)
    acc.push("s0", 0, 1.0, "s1", False, elapsed=0)
    (obs, _, ret, nxt, disc), = acc.push("s1", 5, 2.0, "s2", False, elapsed=1)
    assert obs == "s0" and nxt == "s2"
    assert ret == pytest.approx(1.0 + 2.0) and disc == pytest.approx(0.5)


def test_step_reports_elapsed_minutes(env):
    env.reset(seed=2)
    start, total = env.t, 0
    for _ in range(300):
        t0 = env.t
        *_, info = env.step(CategoryRulePolicy()(None, env))
        assert info["elapsed"] == env.t - t0 >= 0
        total += info["elapsed"]
    assert total == env.t - start
