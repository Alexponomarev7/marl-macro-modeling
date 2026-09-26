"""Fast invariants of the data -> model pipeline (no Dynare needed except the last test)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lib import rewards
from lib.dataset import EconomicsDataset, Tokenizer
from lib.dynare_traj2rl_transitions import _inject_dynare_seed
from lib.evaluation import evaluate
from lib.models.transformer import AlgorithmDistillationTransformer

T, L = 120, 50  # episode length, window length
STEP = 0.01    # per-step increment of the synthetic actions
DIMS = dict(max_state_dim=30, max_action_dim=5, max_endogenous_dim=16, max_model_params_dim=36)


def _write_episode(path: Path, states: np.ndarray, actions: np.ndarray, rewards_: np.ndarray) -> None:
    pd.DataFrame({
        "state": list(states.astype(np.float32)),
        "action": list(actions.astype(np.float32)),
        "endogenous": list(np.zeros((len(states), 1), np.float32)),
        "reward": rewards_,
        "info": [{"model_params": {"beta": 0.99, "alpha": 0.33}, "env_group": "Hansen_1985"}] * len(states),
        "state_description": [["Capital", "LoggedProductivity"]] * len(states),
        "action_description": [["Consumption", "HoursWorked"]] * len(states),
        "endogenous_description": [["Output"]] * len(states),
    }).to_parquet(path)


@pytest.fixture
def episode_dir(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "interim"
    data.mkdir()
    for k in range(3):
        states = rng.normal(size=(T, 2)).cumsum(0) * 0.01 + [10.0, 0.0]
        # a_t encodes t (kept well inside the model's +-1000 input clamp)
        actions = np.stack([1.0 + k + STEP * np.arange(T), -1.0 - STEP * np.arange(T)], 1)
        _write_episode(data / f"Hansen_1985_config_{k}.parquet", states, actions, np.arange(T) + 0.5)
    return data


def _index(data_dir: Path, tmp_path: Path) -> Path:
    from lib.generate_dataset import run_generation_batch_dynare
    index = tmp_path / "index"
    index.mkdir(exist_ok=True)
    run_generation_batch_dynare(data_dir, index)
    return index


def test_dataset_alignment(episode_dir, tmp_path):
    ds = EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False)
    for idx in range(len(ds)):
        item = ds[idx]
        a, pa, r, pr = item["actions"][:, 0], item["prev_actions"][:, 0], item["reward"][:, 0], item["prev_reward"][:, 0]
        start = int(item["window_start"])
        # a_t = 1 + k + STEP*t: the window is the contiguous slice starting at `start`
        assert torch.allclose(a - a[0], STEP * torch.arange(L, dtype=torch.float32), atol=1e-5)
        # step t sees a_{t-1} and r_{t-1}; the episode's first step sees 0
        assert torch.allclose(pa[1:], a[:-1]) and torch.allclose(pr[1:], r[:-1])
        if start == 0:
            assert pa[0] == 0 and pr[0] == 0
        else:
            assert torch.isclose(pa[0], a[0] - STEP, atol=1e-6)
        assert item["attention_mask"].all()
        assert item["actions_info"][2:].eq(0).all() and item["actions_info"][:2].ne(0).all()
        assert (item["action_scale"][:, 2:] == 1).all()  # padded action slots


def test_state_dropout_hides_variables(episode_dir, tmp_path):
    torch.manual_seed(0)
    ds = EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False, state_dropout=0.9)
    for _ in range(20):
        item = ds[0]
        shown = item["states_info"][:2] != 0
        assert shown.any()  # at least one variable stays
        assert (item["states"][:, :2][:, ~shown] == 0).all()


def test_hide_latent_hides_only_latent_states(episode_dir, tmp_path):
    item = EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False, hide_latent=1.0)[0]
    assert item["states_info"][0] != 0 and item["states_info"][1] == 0  # Capital kept, LoggedProductivity hidden
    assert (item["states"][:, 1] == 0).all() and (item["states"][:, 0] != 0).any()


def test_random_windows_cover_the_episode(episode_dir, tmp_path):
    ds = EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=True)
    torch.manual_seed(0)
    starts = {int(ds[0]["window_start"]) for _ in range(200)}
    assert 0 in starts and max(starts) > T - L - 10  # episode start oversampled, tail reachable


def _model(**kw):
    return AlgorithmDistillationTransformer(
        state_dim=30, action_dim=5, num_tasks=Tokenizer().num_tasks, d_model=48, nhead=4, num_layers=2,
        max_seq_len=L, model_params_dim=36, pinn_output_dim=16, has_pinn=kw.pop("has_pinn", True), **kw,
    ).eval()


def _batch(ds, n=3):
    return {k: torch.stack([ds[i][k] for i in range(n)]) for k in ds[0]}


def _forward(model, b, out=0, **override):
    args = dict(states=b["states"], states_info=b["states_info"], actions=b["prev_actions"],
                actions_info=b["actions_info"], rewards=b["prev_reward"], task_ids=b["task_id"],
                model_params=b["model_params"])
    args.update(override)
    with torch.no_grad():
        return model(**args)[out]


def test_residual_head_starts_at_persistence(episode_dir, tmp_path):
    b = _batch(EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False))
    assert torch.allclose(_forward(_model(), b), b["prev_actions"])


def test_model_is_causal(episode_dir, tmp_path):
    """Perturbing inputs at step >= t must not change predictions at steps < t."""
    torch.manual_seed(0)
    model = _model()
    torch.nn.init.normal_(model.action_head.weight, std=0.1)  # non-trivial predictions
    b = _batch(EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False))
    base = _forward(model, b)
    t = 30
    states = b["states"].clone(); states[:, t:] += 5.0
    actions = b["prev_actions"].clone(); actions[:, t:] += 5.0
    rewards_ = b["prev_reward"].clone(); rewards_[:, t:] += 5.0
    moved = _forward(model, b, states=states, actions=actions, rewards=rewards_)
    assert torch.allclose(moved[:, :t], base[:, :t], atol=1e-5)
    assert not torch.allclose(moved[:, t:], base[:, t:])


def _causal_model():
    torch.manual_seed(0)
    model = _model(input_normalization="causal")
    torch.nn.init.normal_(model.action_head.weight, std=0.1)  # non-trivial predictions
    return model


def test_causal_mode_starts_at_persistence(episode_dir, tmp_path):
    b = _batch(EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False))
    assert torch.allclose(_forward(_model(input_normalization="causal"), b), b["prev_actions"])


def test_causal_mode_is_scale_free(episode_dir, tmp_path):
    """Rescaling a state leaves the predictions unchanged; rescaling an action rescales its own."""
    model = _causal_model()
    b = _batch(EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False))
    base = _forward(model, b)
    states = b["states"].clone(); states[..., 0] *= 1000.0
    actions = b["prev_actions"].clone(); actions[..., 1] *= 0.01
    moved = _forward(model, b, states=states, actions=actions)
    assert not torch.allclose(base, b["prev_actions"])
    assert torch.allclose(moved[..., 0], base[..., 0], rtol=1e-4, atol=1e-6)
    assert torch.allclose(moved[..., 1], 0.01 * base[..., 1], rtol=1e-4, atol=1e-8)


def test_causal_mode_is_causal_and_ignores_placeholders(episode_dir, tmp_path):
    model = _causal_model()
    b = _batch(EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False))
    start = torch.ones(len(b["states"]), dtype=torch.bool)
    base = _forward(model, b, first_step=start)
    t = 30
    states = b["states"].clone(); states[:, t:] += 5.0
    actions = b["prev_actions"].clone(); actions[:, t:] *= 3.0
    moved = _forward(model, b, states=states, actions=actions, first_step=start)
    assert torch.allclose(moved[:, :t], base[:, :t], atol=1e-6) and not torch.allclose(moved[:, t:], base[:, t:])
    # an episode's first step carries placeholder a_{-1}, r_{-1}: they must not matter
    actions, rewards_ = b["prev_actions"].clone(), b["prev_reward"].clone()
    actions[:, 0], rewards_[:, 0] = 123.0, -7.0
    moved = _forward(model, b, actions=actions, rewards=rewards_, first_step=start)
    assert torch.allclose(moved[:, 1:], base[:, 1:], atol=1e-6)


def test_dynamics_head_predicts_state_changes_in_state_units(episode_dir, tmp_path):
    torch.manual_seed(0)
    model = _model(input_normalization="causal", dynamics_head=True)
    torch.nn.init.normal_(model.dynamics_head[-1].weight, std=0.1)
    b = _batch(EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False))
    base = _forward(model, b, out=1)["dynamics"]
    states = b["states"].clone(); states[..., 0] *= 1000.0
    moved = _forward(model, b, out=1, states=states)["dynamics"]
    assert torch.allclose(moved[..., 0], 1000.0 * base[..., 0], rtol=1e-4, atol=1e-6)
    assert torch.allclose(moved[..., 1], base[..., 1], rtol=1e-4, atol=1e-8)


def test_next_state_loss_scores_only_real_next_steps():
    from pipeline.run_pipeline import next_state_loss
    g = torch.Generator().manual_seed(0)
    states = torch.randn(2, L, 3, generator=g).cumsum(1)
    states[..., 2] = 0.0
    info = torch.tensor([[5, 6, 0], [5, 6, 0]])  # the third slot is padding
    mask = torch.ones(2, L, dtype=torch.bool)
    mask[1, :10] = False  # left padding in the second window
    perfect = torch.cat([states[:, 1:] - states[:, :-1], torch.zeros(2, 1, 3)], 1)
    wrong = perfect.clone()
    wrong[:, -1], wrong[..., 2], wrong[1, :10] = 99.0, 99.0, 99.0  # no next step, padded slot, padded steps
    assert next_state_loss(perfect, states, mask, info) < 1e-10
    assert next_state_loss(wrong, states, mask, info) < 1e-10
    assert next_state_loss(perfect + 0.1, states, mask, info) > 0


def test_ridge_channel_matches_the_oracle_and_the_model_stays_causal(episode_dir, tmp_path):
    from lib.evaluation import _in_context_ridge
    from lib.models.transformer import in_context_ridge_change
    g = torch.Generator().manual_seed(0)
    s = torch.randn(4, L, 3, generator=g).cumsum(1)
    a = 5.0 + torch.einsum("bls,bsa->bla", s, torch.randn(4, 3, 2, generator=g)) + 0.01 * torch.randn(4, L, 2, generator=g)
    pa = torch.cat([a[:, :1] - 0.1, a[:, :-1]], 1)  # a window inside an episode: a_{-1} is real
    oracle = _in_context_ridge(s, pa, a, torch.ones(4, L, dtype=torch.bool))
    first = torch.arange(L).view(1, L, 1) > 0  # no difference at the window's first position
    change = in_context_ridge_change(s, pa, first.expand(4, L, 1), first.expand(4, L, 1))
    assert torch.allclose((pa + change)[:, 10:], oracle[:, 10:], atol=1e-4)

    torch.manual_seed(0)
    model = _model(input_normalization="causal", ridge_channel=True)
    torch.nn.init.normal_(model.action_head.weight, std=0.1)
    b = _batch(EconomicsDataset(_index(episode_dir, tmp_path), **DIMS, max_seq_len=L, random_window=False))
    base = _forward(model, b)
    t = 30
    states = b["states"].clone(); states[:, t:] += 5.0
    actions = b["prev_actions"].clone(); actions[:, t:] *= 3.0
    moved = _forward(model, b, states=states, actions=actions)
    assert torch.allclose(moved[:, :t], base[:, :t], atol=1e-6) and not torch.allclose(moved[:, t:], base[:, t:])


def test_windowed_ridge_uses_only_the_last_pairs():
    from lib.models.transformer import in_context_ridge_change
    g = torch.Generator().manual_seed(1)
    s, pa = torch.randn(2, L, 3, generator=g).cumsum(1), torch.randn(2, L, 2, generator=g).cumsum(1)
    ok = lambda n: (torch.arange(n).view(1, n, 1) > 0).expand(2, n, 1)
    w, t = 12, 40
    windowed = in_context_ridge_change(s, pa, ok(L), ok(L), window=w)[:, t]
    cut = slice(t - w - 1, t + 1)
    assert torch.allclose(windowed, in_context_ridge_change(s[:, cut], pa[:, cut], ok(w + 2), ok(w + 2))[:, -1], atol=1e-5)


def test_ridge_skip_can_output_the_ridge_estimate():
    from lib.models.transformer import in_context_ridge_change
    g = torch.Generator().manual_seed(0)
    s = torch.randn(2, L, 3, generator=g).cumsum(1)
    a = 5.0 + torch.einsum("bls,bsa->bla", s, torch.randn(2, 3, 2, generator=g))
    pa = torch.cat([a[:, :1] - 0.1, a[:, :-1]], 1)
    model = _model(input_normalization="causal", ridge_channel=True, ridge_skip=True)
    torch.manual_seed(0)
    torch.nn.init.normal_(model.action_head.weight, std=0.1)
    with torch.no_grad():
        model.ridge_gate.bias.view(-1, 3)[:, 1] = 30.0   # the full-window estimate
        model.ridge_gate.bias.view(-1, 3)[:, 2] = -30.0  # no correction
        pad = lambda x, d: torch.nn.functional.pad(x, (0, d - x.shape[-1]))
        pred = model(states=pad(s, 30), states_info=torch.ones(2, 30, dtype=torch.long), actions=pad(pa, 5),
                     actions_info=torch.ones(2, 5, dtype=torch.long), rewards=torch.zeros(2, L, 1),
                     task_ids=torch.zeros(2, dtype=torch.long), model_params=torch.zeros(2, 36))[0]
    first = (torch.arange(L).view(1, L, 1) > 0).expand(2, L, 1)
    assert torch.allclose(pred[..., :2], pa + in_context_ridge_change(s, pa, first, first), atol=1e-4)


def test_weight_decay_spares_biases_norms_and_embeddings():
    import hydra
    from omegaconf import OmegaConf
    from pipeline.run_pipeline import decay_param_groups
    model = _model(input_normalization="causal", ridge_channel=True, ridge_skip=True)
    decay, no_decay = ({id(p) for p in g["params"]} for g in decay_param_groups(model))
    norms = [p for m in model.modules() if isinstance(m, torch.nn.LayerNorm) for p in m.parameters()]
    assert all(id(p) in no_decay for p in norms + [model.state_embedding.weight, model.ridge_gate.bias])
    assert id(model.transformer.layers[0].linear1.weight) in decay and id(model.ridge_gate.weight) in decay
    assert len(decay) + len(no_decay) == len(list(model.parameters()))
    cfg = OmegaConf.load(Path(__file__).parents[1] / "pipeline/configs/train/default.yaml").optimizer
    optimizer = hydra.utils.instantiate(cfg, _partial_=True)(decay_param_groups(model))  # as in configure_optimizers
    assert [g["weight_decay"] for g in optimizer.param_groups] == [cfg.weight_decay, 0.0]


def test_observed_steps_skip_padding_and_the_episode_start_placeholder():
    mask = torch.tensor([[False, False, True, True], [True, True, True, True], [True, True, True, True]])
    first = torch.tensor([True, True, False])
    valid, prev_valid = AlgorithmDistillationTransformer._observed_steps(torch.zeros(3, 4, 2), first, mask)
    assert valid[..., 0].equal(mask)
    # the placeholder a_{-1}, r_{-1} sits at the first valid position of a window that opens an episode
    assert prev_valid[..., 0].equal(torch.tensor([[False, False, False, True], [False, True, True, True], [True] * 4]))


def test_load_policy_after_tokens_were_appended(tmp_path):
    """A checkpoint saved before new Tokenizer tokens were appended still loads, rows unchanged."""
    from lib.evaluation import load_policy
    cfg = dict(state_dim=30, action_dim=5, num_tasks=Tokenizer().num_tasks, d_model=48, nhead=4, num_layers=2,
               max_seq_len=L, model_params_dim=36, pinn_output_dim=16, has_pinn=True, context_only=True)
    state = {f"model.{k}": v for k, v in AlgorithmDistillationTransformer(**cfg).state_dict().items()}
    for name in ("model.state_embedding.weight", "model.action_embedding.weight"):
        state[name] = state[name][:-2].clone()  # as saved before the last two tokens existed
    torch.save({"hyper_parameters": {"model_cfg": cfg}, "state_dict": state}, tmp_path / "old.ckpt")
    model = load_policy(tmp_path / "old.ckpt")
    rows = state["model.state_embedding.weight"]
    assert torch.equal(model.state_embedding.weight[:len(rows)], rows)


def test_lq_tasks(tmp_path):
    """Optimal actions are linear in the state; with behavior noise the model's a_{t-1} is the
    executed action while the target stays optimal."""
    from lib.generate_dataset import run_generation_batch_dynare
    from lib.lq_tasks import random_task, simulate
    rng = np.random.default_rng(0)
    data = tmp_path / "lq"
    data.mkdir()
    for k in range(3):
        simulate(random_task(rng), T, rng, behavior_noise=0.5).to_parquet(data / f"LQ_random_config_{k}.parquet")
    df = pd.read_parquet(data / "LQ_random_config_0.parquet")
    S, A = np.stack(df.state), np.stack(df.action)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(T), S]), A, rcond=None)
    assert np.allclose(np.column_stack([np.ones(T), S]) @ coef, A, atol=1e-8)
    index = tmp_path / "index"
    index.mkdir()
    run_generation_batch_dynare(data, index)
    item = EconomicsDataset(index, **DIMS, max_seq_len=L, random_window=False)[0]
    executed = np.stack(df.behavior_action)
    start, m = int(item["window_start"]), A.shape[1]
    assert np.allclose(item["prev_actions"][1:, :m].numpy(), executed[start:start + L - 1], atol=1e-5)
    assert np.allclose(item["actions"][:, :m].numpy(), A[start:start + L], atol=1e-5)


def test_dataset_stage_adds_lq_tasks(episode_dir, tmp_path):
    from lib.generate_dataset import run_generation_batch_dynare
    index = tmp_path / "index"
    index.mkdir()
    run_generation_batch_dynare(episode_dir, index, lq_tasks={"episodes": 2, "regimes": 2, "periods": T})
    meta = json.loads((index / "metadata.json").read_text())
    assert [m["env_group"] for m in meta] == ["Hansen_1985"] * 3 + ["LQ_random"] * 2
    assert "regime" in pd.read_parquet(meta[-1]["output_dir"]).columns
    item = EconomicsDataset(index, **DIMS, max_seq_len=L, random_window=False)[4]
    assert item["states_info"].ne(0).any() and torch.isfinite(item["states"]).all()


def test_separable_utility_reward():
    d = pd.DataFrame({"c": [1.0, 2.0], "h": [0.3, 0.4], "m": [2.0, 3.0], "p": [1.0, 1.5]})
    params = {"B": 2.0, "chi": 0.5, "phi": 1.5, "D": 0.1, "sigma": 2.0}
    r = rewards.separable_utility_reward(d, params, "c", "h", labor_form="linear", labor_weight_column="B")
    assert np.allclose(r, np.log(d.c) - 2.0 * d.h)
    r = rewards.separable_utility_reward(d, params, "c", "h", labor_form="linear", labor_weight_column="B",
                                         labor_weight_scale=-1.0, money_column="m", price_column="p",
                                         money_weight_column="D")
    assert np.allclose(r, np.log(d.c) + 2.0 * d.h + 0.1 * np.log(d.m / d.p))
    r = rewards.separable_utility_reward(d, params, "c", "h", sigma_column="sigma", labor_form="power",
                                         labor_weight_column="chi", frisch_column="phi")
    assert np.allclose(r, (d.c ** -1.0 - 1) / -1.0 - 0.5 * d.h ** 2.5 / 2.5)


def test_crra_is_normalized():
    d = pd.DataFrame({"c": [1.0, 1.0]})
    for sigma in (0.5, 1.0, 2.0, 5.0):
        assert np.allclose(rewards.crra_reward(d, {}, target_column="c", sigma_default=sigma), 0.0)


def test_olg_reward_is_the_savers_lifetime_utility():
    d = pd.DataFrame({"y": [1.0, 2.0, 3.0], "o": [4.0, 5.0, 6.0]})
    r = rewards.olg_lifetime_utility_reward(d, {"beta": 0.5}, "y", "o", beta_column="beta")
    assert np.allclose(r, np.log(d.y) + 0.5 * np.log([5.0, 6.0, 6.0]))


def test_seed_injection():
    mod = "% stoch_simul( in a comment\nmodel;\nend;\n  stoch_simul(order=1);\nstoch_simul(order=2);\n"
    out = _inject_dynare_seed(mod, 123)
    assert out.count("set_dynare_seed(123);") == 1
    assert "  set_dynare_seed(123);\n  stoch_simul(order=1);" in out
    assert "discretionary_policy" in _inject_dynare_seed("discretionary_policy(instruments=(r));", 7)
    assert _inject_dynare_seed("discretionary_policy(instruments=(r));", 7).startswith("set_dynare_seed(7);")


def test_evaluation_persistence_baseline(episode_dir, tmp_path):
    """A fresh (zero-residual) model IS persistence, so model and baseline NMSE must coincide."""
    table = evaluate(_model(), episode_dir, windows_per_episode=2, seed=0)
    row = table.iloc[0]
    assert row.env == "Hansen_1985" and row.episodes == 3
    assert np.isclose(row.model_nmse, row.persistence_nmse)
    hidden = evaluate(_model(), episode_dir, windows_per_episode=2, seed=0, hidden_states=("Capital",)).iloc[0]
    assert np.isclose(hidden.model_nmse, hidden.persistence_nmse)


def test_evaluation_without_pinn_head(episode_dir):
    for f in episode_dir.glob("*.parquet"):
        df = pd.read_parquet(f)
        df["endogenous"] = [np.zeros(3, np.float32)] * len(df)
        df["endogenous_description"] = [["Output", "Investment", "Wage"]] * len(df)
        df.to_parquet(f)
    row = evaluate(_model(has_pinn=False), episode_dir, windows_per_episode=2, seed=0).iloc[0]
    assert np.isclose(row.model_nmse, row.persistence_nmse)


def test_in_context_ridge_recovers_a_linear_policy():
    from lib.evaluation import _in_context_ridge
    g = torch.Generator().manual_seed(0)
    s = torch.randn(4, L, 3, generator=g).cumsum(1)
    a = 5.0 + torch.einsum("bls,bsa->bla", s, torch.randn(4, 3, 2, generator=g))  # a_t = c + G s_t
    pa = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], 1)
    steps = torch.ones(4, L, dtype=torch.bool)
    steps[:, 0] = False  # cold start
    pred = _in_context_ridge(s, pa, a, steps)
    assert torch.allclose(pred[:, :2], pa[:, :2])  # no (Delta s, Delta a) pair yet: persistence
    assert (((pred - a) ** 2).mean(-1)[:, 10:] / a.var()).max() < 1e-3


def test_in_context_ridge_ignores_rounding_noise():
    """A state that so far moved only by rounding error must not blow up the estimate when it first moves."""
    from lib.evaluation import _in_context_ridge
    from lib.models.transformer import in_context_ridge_change
    g = torch.Generator().manual_seed(0)
    s = torch.randn(1, L, 2, generator=g, dtype=torch.float64).cumsum(1)
    s[..., 1] = 1e-18 * torch.randn(1, L, generator=g, dtype=torch.float64)
    s[:, 10:, 1] += 0.5
    a = 5.0 + 2.0 * s[..., :1]
    pa = torch.cat([a[:, :1], a[:, :-1]], 1)
    first = torch.arange(L).view(1, L, 1) > 0
    for pred in (_in_context_ridge(s, pa, a, torch.ones(1, L, dtype=torch.bool)).double(),
                 pa + in_context_ridge_change(s, pa, first, first)):
        assert (pred - a)[:, 5:].abs().max() < 0.1


def test_oracle_change_is_clamped():
    from lib.evaluation import ORACLE_CLAMP, _in_context_ridge
    g = torch.Generator().manual_seed(0)
    s = torch.randn(1, L, 2, generator=g, dtype=torch.float64).cumsum(1)
    s[:, 30:, 0] += 1e3  # far outside the context
    a = 5.0 + 2.0 * s[..., :1] + s[..., 1:]
    pa = torch.cat([a[:, :1], a[:, :-1]], 1)
    pred = _in_context_ridge(s, pa, a, torch.ones(1, L, dtype=torch.bool)).double()
    rms = ((a - pa)[:, 1:30] ** 2).mean().sqrt()
    assert (pred[:, 30] - pa[:, 30]).abs().max() <= ORACLE_CLAMP * rms * (1 + 1e-4)


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="needs Octave + Dynare")
def test_mit_shock_solver_matches_dynare_expectation_errors(tmp_path):
    models = Path(__file__).resolve().parent.parent / "dynare" / "docker" / "dynare_models"
    mod = """var c k a; varexo e; parameters alpha beta delta rho;
alpha=0.33; beta=0.99; delta=0.025; rho=0.9;
model;
c^(-1) = beta*c(+1)^(-1)*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
k = exp(a)*k(-1)^alpha + (1-delta)*k(-1) - c;
a = rho*a(-1) + e;
end;
steady_state_model; a = 0; k = (alpha/(1/beta - 1 + delta))^(1/(1-alpha)); c = k^alpha - delta*k; end;
initval; a = 0; k = 0.9*(alpha/(1/beta - 1 + delta))^(1/(1-alpha)); c = 2; end;
endval; a = 0; k = (alpha/(1/beta - 1 + delta))^(1/(1-alpha)); c = k^alpha - delta*k; end;
"""
    (tmp_path / "ee.mod").write_text(mod + """shocks(learnt_in=10); var e; periods 10; values 0.05; end;
shocks(learnt_in=25); var e; periods 25; values -0.03; end;
perfect_foresight_with_expectation_errors_setup(periods=80);
perfect_foresight_with_expectation_errors_solver;
""")
    (tmp_path / "mit.mod").write_text(mod + """shocks; var e; periods 10 25; values 0.05 -0.03; end;
perfect_foresight_setup(periods=80);
oo_ = mit_shock_solver(M_, options_, oo_);
""")
    script = f"""addpath /opt/homebrew/opt/dynare/lib/dynare/matlab; addpath {models};
dynare ee.mod noclearall nolog; a = oo_.endo_simul;
dynare mit.mod noclearall nolog; b = oo_.endo_simul;
fprintf('MAXDIFF %.3e\\n', max(abs(a(:) - b(:))));"""
    out = subprocess.run(["octave-cli", "--eval", script], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    diff = [l for l in out.stdout.splitlines() if l.startswith("MAXDIFF")]
    assert diff, out.stdout[-2000:] + out.stderr[-2000:]
    assert float(diff[0].split()[1]) < 1e-8


def test_dataset_model_selection(episode_dir, tmp_path):
    import json
    from lib.generate_dataset import run_generation_batch_dynare
    rng = np.random.default_rng(1)
    _write_episode(episode_dir / "SGU_2004_config_0.parquet", rng.normal(size=(T, 2)), np.ones((T, 2)), np.zeros(T))

    def names(**sel):
        out = tmp_path / f"idx_{len(list(tmp_path.iterdir()))}"
        out.mkdir()
        run_generation_batch_dynare(episode_dir, out, **sel)
        return sorted({m["env_name"].rsplit("_config_", 1)[0] for m in json.load(open(out / "metadata.json"))})

    assert names() == ["Hansen_1985", "SGU_2004"]
    assert names(exclude_models=["Hansen_1985"]) == ["SGU_2004"]
    assert names(include_models=["Hansen_1985"]) == ["Hansen_1985"]
    with pytest.raises(KeyError):
        names(include_models=["NotAModel"])
