"""Procedurally generated linear-quadratic control tasks in the processed-episode format.

Each task draws a stable system x' = A x + B u + C e and a discounted loss x'Qx + u'Ru; the
optimal policy u = -K x solves the discounted Riccati equation. With behavior_noise > 0 the
executed actions (column behavior_action) deviate from the optimal ones (column action). With
regimes > 1 the system and loss are redrawn at random dates of the episode (column regime).

    python -m lib.lq_tasks --out data/interim --episodes 2000 [--periods 1000] [--behavior-noise 0.5] [--regimes 3]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are

MAX_STATES, MAX_ACTIONS = 8, 3
ENV_NAME = "LQ_random"


def random_task(rng: np.random.Generator, n: int | None = None, m: int | None = None) -> dict:
    n = int(rng.integers(1, MAX_STATES + 1)) if n is None else n
    m = int(rng.integers(1, min(MAX_ACTIONS, n) + 1)) if m is None else m
    radius = rng.uniform(0.5, 0.99)
    A = rng.normal(size=(n, n))
    A *= radius / max(abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(n, m))
    G, H = rng.normal(size=(n, n)), rng.normal(size=(m, m))
    Q, R = G @ G.T / n + 0.1 * np.eye(n), H @ H.T / m + 0.1 * np.eye(m)
    beta = rng.uniform(0.9, 0.99)
    P = solve_discrete_are(np.sqrt(beta) * A, np.sqrt(beta) * B, Q, R)
    K = beta * np.linalg.solve(R + beta * B.T @ P @ B, B.T @ P @ A)
    return {
        "A": A, "B": B, "C": np.diag(rng.uniform(0.2, 1.0, n)), "Q": Q, "R": R, "K": K, "beta": beta,
        "radius": radius, "shock_std": rng.uniform(0.005, 0.05),
        "x_bar": rng.lognormal(0.0, 1.0, n), "u_bar": rng.lognormal(0.0, 1.0, m),
    }


def simulate(task: dict, periods: int, rng: np.random.Generator, behavior_noise: float = 0.0, burn_in: int = 100,
             regimes: int = 1) -> pd.DataFrame:
    n, m = task["B"].shape
    tasks = [task] + [random_task(rng, n, m) for _ in range(regimes - 1)]
    starts = np.sort(rng.choice(np.arange(1, periods), regimes - 1, replace=False)) + burn_in
    regime_of = lambda t: int(np.searchsorted(starts, t, side="right"))
    step = lambda k, x, u: tasks[k]["A"] @ x + tasks[k]["B"] @ u + tasks[k]["shock_std"] * tasks[k]["C"] @ rng.normal(size=n)
    u_std = np.zeros(m)
    if behavior_noise > 0:  # scale of the optimal actions, from a pilot run without noise
        x, pilot = np.zeros(n), []
        for _ in range(burn_in + 200):
            pilot.append(-task["K"] @ x)
            x = step(0, x, pilot[-1])
        u_std = np.std(pilot[burn_in:], axis=0) + 1e-12
    x, rows = np.zeros(n), []
    for t in range(burn_in + periods):
        k = regime_of(t)
        u_opt = -tasks[k]["K"] @ x
        u = u_opt + behavior_noise * u_std * rng.normal(size=m)
        x_next = step(k, x, u)
        if t >= burn_in:
            rows.append((x, u_opt, u, x_next, -(x @ tasks[k]["Q"] @ x + u @ tasks[k]["R"] @ u), k))
        x = x_next
    xs, u_opts, us, xns, rewards, regime = map(np.array, zip(*rows))
    params = {"beta": task["beta"], "spectral_radius": task["radius"], "shock_std": task["shock_std"],
              "n_states": float(n), "n_actions": float(m)}
    df = pd.DataFrame({
        "state": list(task["x_bar"] + xs),
        "action": list(task["u_bar"] + u_opts),
        "reward": rewards,
        "endogenous": list(task["x_bar"] + xns),
        "info": [{"model_params": params, "env_group": ENV_NAME}] * periods,
        "state_description": [[f"LQState{i + 1}" for i in range(n)]] * periods,
        "action_description": [[f"LQAction{j + 1}" for j in range(m)]] * periods,
        "endogenous_description": [[f"LQState{i + 1}" for i in range(n)]] * periods,
    })
    if behavior_noise > 0:
        df["behavior_action"] = list(task["u_bar"] + us)
    if regimes > 1:
        df["regime"] = regime
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--periods", type=int, default=1000)
    parser.add_argument("--behavior-noise", type=float, default=0.0, help="std of executed-action noise, relative to the optimal actions' std")
    parser.add_argument("--regimes", type=int, default=1, help="systems per episode, switching at random dates")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for k in range(args.episodes):
        episode = simulate(random_task(rng), args.periods, rng, args.behavior_noise, regimes=args.regimes)
        episode.to_parquet(args.out / f"{ENV_NAME}_config_{k}.parquet")


if __name__ == "__main__":
    main()
