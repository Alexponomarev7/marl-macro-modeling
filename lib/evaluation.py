"""Per-environment evaluation of a checkpoint: NMSE of a_t given (s_t, a_{t-1}, r_{t-1}) against
persistence (a_t = a_{t-1}) and an in-context ridge oracle, by window position.

--context-diagnostics re-scores the last position with the history shuffled in time and with the
context cut to TRUNCATED steps; --bootstrap N adds per-environment CIs of model/persistence;
--observables-only hides LATENT_STATES from the model and the oracle.

    python -m lib.evaluation --checkpoint path/to/model.ckpt --data data/interim [--windows 8]
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.dataset import LATENT_STATES, EconomicsDataset, Tokenizer, latent_token_ids
from lib.generate_dataset import run_generation_batch_dynare
from lib.models.transformer import AlgorithmDistillationTransformer

EARLY, LATE = 5, 10  # first / last window positions
TRUNCATED = 2
RIDGE = 1e-2
ORACLE_CLAMP = 10.0  # bound on the oracle's |a_t - a_{t-1}|, in running RMS of past action changes


def load_policy(checkpoint: Path) -> AlgorithmDistillationTransformer:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in ckpt["hyper_parameters"]["model_cfg"].items() if k != "_target_"}
    model = AlgorithmDistillationTransformer(**cfg)
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    # tokens are only appended: keep the checkpoint's rows, new tokens keep their initialization
    for name in ("state_embedding.weight", "action_embedding.weight"):
        new, old = model.state_dict()[name], state[name]
        if old.shape[0] < new.shape[0] and old.shape[1:] == new.shape[1:]:
            state[name] = torch.cat([old, new[old.shape[0]:]])
    model.load_state_dict(state)
    return model.eval()


def _predict(model, batch, **inputs):
    """`inputs` override the batch's states, prev_actions, prev_reward, first_step or attention_mask."""
    inputs = {
        "states": batch["states"], "prev_actions": batch["prev_actions"], "prev_reward": batch["prev_reward"],
        "first_step": batch["window_start"] == 0, "attention_mask": batch["attention_mask"], **inputs,
    }
    with torch.no_grad():
        pred, _ = model(
            states=inputs["states"],
            states_info=batch["states_info"],
            actions=inputs["prev_actions"],
            actions_info=batch["actions_info"],
            rewards=inputs["prev_reward"],
            task_ids=batch["task_id"],
            model_params=batch["model_params"],
            first_step=inputs["first_step"],
            attention_mask=inputs["attention_mask"],
        )
    return pred


def _shuffle_history(batch, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Permute positions 0..L-3 of each sequence, keeping each step's (s, a, r) together."""
    out = {k: batch[k].clone() for k in ("states", "prev_actions", "prev_reward")}
    for i in range(out["states"].shape[0]):
        perm = torch.randperm(out["states"].shape[1] - 2, generator=generator)
        for x in out.values():
            x[i, :-2] = x[i, perm]
    return out


def _in_context_ridge(states, prev_actions, actions, steps, lam=RIDGE):
    """At position t, ridge-regress a_tau - a_{tau-1} on s_tau - s_{tau-1} over tau < t and predict
    a_t = a_{t-1} + G_hat (s_t - s_{t-1}), clamped to ORACLE_CLAMP running RMS of past action changes;
    persistence until there are two more pairs than states.

    states: [B, L, S]; prev_actions, actions: [B, L, A]; steps: [B, L] bool, usable positions."""
    s, pa, a = states.double(), prev_actions.double(), actions.double()
    x = torch.zeros_like(s)
    x[:, 1:] = s[:, 1:] - s[:, :-1]
    use = steps & torch.cat([torch.zeros_like(steps[:, :1]), steps[:, :-1]], 1)
    xm, ym = x * use.unsqueeze(-1), (a - pa) * use.unsqueeze(-1)
    exclusive = lambda z: torch.cat([torch.zeros_like(z[:, :1]), torch.cumsum(z, 1)[:, :-1]], 1)
    xtx = exclusive(xm.unsqueeze(-1) * xm.unsqueeze(-2))  # [B, L, S, S], pairs tau < t
    xty = exclusive(xm.unsqueeze(-1) * ym.unsqueeze(-2))  # [B, L, S, A]
    d = torch.sqrt(torch.diagonal(xtx, dim1=-2, dim2=-1)) + 1e-30  # [B, L, S]
    gram = xtx / (d.unsqueeze(-1) * d.unsqueeze(-2)) + lam * torch.eye(s.shape[-1], dtype=s.dtype)
    coef = torch.linalg.solve(gram, xty / d.unsqueeze(-1))  # standardized G_hat, [B, L, S, A]
    has_pairs = d > 1e-30
    step = torch.einsum("bls,blsa->bla", torch.where(has_pairs, x / d, 0.0), coef)
    n_pairs = exclusive(use.double())  # [B, L]
    n_features = (x.abs().sum(1) > 0).sum(-1, keepdim=True)  # state slots that ever move, [B, 1]
    bound = ORACLE_CLAMP * torch.sqrt(exclusive(ym * ym) / n_pairs.clamp(min=1).unsqueeze(-1))
    step = torch.maximum(torch.minimum(step, bound), -bound)
    return (pa + torch.where((n_pairs >= n_features + 2).unsqueeze(-1), step, 0.0)).float()


def evaluate(
    model: AlgorithmDistillationTransformer,
    data_dir: Path,
    windows_per_episode: int = 8,
    seed: int = 0,
    context_diagnostics: bool = False,
    bootstrap: int = 0,
    batch_size: int = 64,
    hidden_states: tuple[str, ...] = (),
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as index_dir:
        run_generation_batch_dynare(Path(data_dir), Path(index_dir))
        dataset = EconomicsDataset(
            Path(index_dir), model.state_dim, model.action_dim, model.pinn_head[-1].out_features if model.has_pinn else 1,
            model.model_params_dim, model.max_seq_len, random_window=True,
        )
        env_of_task = {v: k for k, v in Tokenizer.ENV_MAPPING.items()}
        hidden_ids = torch.tensor(sorted(latent_token_ids(hidden_states)), dtype=torch.long)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        # env -> episode -> per-window arrays; diagnostics hold last-position errors
        records: dict[str, dict[int, dict[str, list]]] = {}
        for _ in range(windows_per_episode):
            for batch_start, batch in enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
                if len(hidden_ids):
                    hide = torch.isin(batch["states_info"], hidden_ids)
                    batch["states"] = batch["states"] * (~hide).unsqueeze(1)
                    batch["states_info"] = torch.where(hide, 0, batch["states_info"])
                pred = _predict(model, batch)
                # scored steps: no padding, no episode-start placeholder
                steps = model._observed_steps(batch["states"], batch["window_start"] == 0, batch["attention_mask"])[1][..., 0]
                oracle = _in_context_ridge(batch["states"], batch["prev_actions"], batch["actions"], steps)
                diag = {}
                if context_diagnostics:
                    # diagnostics use full windows that do not start an episode
                    no_start = torch.zeros_like(batch["window_start"], dtype=torch.bool)
                    diag["shuffled"] = _predict(model, batch, first_step=no_start, **_shuffle_history(batch, generator))[:, -1]
                    tail = {k: batch[k][:, -TRUNCATED:] for k in ("states", "prev_actions", "prev_reward", "attention_mask")}
                    diag["truncated"] = _predict(model, batch, first_step=no_start, **tail)[:, -1]
                    diag_ok = batch["attention_mask"].all(1) & (batch["window_start"] > 0)
                valid = (batch["actions_info"] != 0).unsqueeze(1) & steps.unsqueeze(-1)
                for i in range(len(batch["task_id"])):
                    env = env_of_task[int(batch["task_id"][i])]
                    episode = batch_start * batch_size + i
                    na = int((batch["actions_info"][i] != 0).sum())
                    a = batch["actions"][i, :, :na]
                    rec = records.setdefault(env, {}).setdefault(episode, {
                        "model": [], "persist": [], "oracle": [], "act": [], "mask": [],
                        "shuffled": [], "truncated": [], "diag_ok": [],
                    })
                    rec["model"].append(((pred[i, :, :na] - a) ** 2).numpy())
                    rec["persist"].append(((batch["prev_actions"][i, :, :na] - a) ** 2).numpy())
                    rec["oracle"].append(((oracle[i, :, :na] - a) ** 2).numpy())
                    rec["act"].append(a.numpy())
                    rec["mask"].append(valid[i, :, :na].numpy())
                    for key, last in diag.items():
                        rec[key].append(((last[i, :na] - a[-1]) ** 2).numpy())
                    if diag:
                        rec["diag_ok"].append(bool(diag_ok[i]))

    rows = []
    rng = np.random.default_rng(seed)
    for env, episodes in sorted(records.items()):
        eps = list(episodes.values())

        def scores(sample):
            cat = lambda key: np.concatenate([np.stack(e[key]) for e in sample])  # [W, L, na] or [W, na]
            mask, act = cat("mask"), cat("act")
            var = np.array([act[..., j][mask[..., j]].var() for j in range(act.shape[-1])]) + 1e-12
            nmse = lambda se, sl=slice(None): np.mean([
                se[:, sl, j][mask[:, sl, j]].mean() / var[j] for j in range(act.shape[-1])
            ])
            se_m, se_o = cat("model"), cat("oracle")
            out = {
                "model_nmse": nmse(se_m), "persistence_nmse": nmse(cat("persist")), "oracle_nmse": nmse(se_o),
                "model_early": nmse(se_m, slice(0, EARLY)), "model_late": nmse(se_m, slice(-LATE, None)),
                "oracle_early": nmse(se_o, slice(0, EARLY)), "oracle_late": nmse(se_o, slice(-LATE, None)),
            }
            if sample[0]["shuffled"]:
                last = mask[:, -1, :] & np.concatenate([e["diag_ok"] for e in sample])[:, None]
                at_last = lambda se: np.mean([se[:, j][last[:, j]].mean() / var[j] for j in range(act.shape[-1])])
                out["model_last"] = at_last(se_m[:, -1])
                out["shuffled_last"] = at_last(cat("shuffled"))
                out["truncated_last"] = at_last(cat("truncated"))
            return out

        row = {"env": env, "episodes": len(eps), **scores(eps)}
        if bootstrap:
            ratios = [
                (s := scores([eps[k] for k in rng.integers(0, len(eps), len(eps))]))["model_nmse"]
                / (s["persistence_nmse"] + 1e-12)
                for _ in range(bootstrap)
            ]
            row["ratio_ci_low"], row["ratio_ci_high"] = np.percentile(ratios, [2.5, 97.5])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.insert(4, "ratio", table.model_nmse / (table.persistence_nmse + 1e-12))
    table.attrs["position_curve"] = _position_curve(records)
    return table


def _position_curve(records) -> pd.DataFrame:
    """Median over environments of the NMSE at each window position (model, persistence, oracle)."""
    per_env = []
    for episodes in records.values():
        cat = lambda key: np.concatenate([np.stack(e[key]) for e in episodes.values()])  # [W, L, na]
        mask, act = cat("mask"), cat("act")
        var = np.array([act[..., j][mask[..., j]].var() for j in range(act.shape[-1])]) + 1e-12
        curves = {}
        for name, key in [("model", "model"), ("persistence", "persist"), ("oracle", "oracle")]:
            se = cat(key)
            with np.errstate(invalid="ignore"):  # positions without valid steps give NaN
                curves[name] = np.nanmean([
                    (se[..., j] * mask[..., j]).sum(0) / mask[..., j].sum(0) / var[j] for j in range(act.shape[-1])
                ], axis=0)
        per_env.append(curves)
    curve = pd.DataFrame({
        name: np.nanmedian(np.stack([c[name] for c in per_env]), axis=0) for name in ("model", "persistence", "oracle")
    })
    curve.index.name = "position"
    return curve


def summarize(table: pd.DataFrame) -> str:
    ratio = (table.model_nmse + 1e-6) / (table.persistence_nmse + 1e-6)
    lines = [
        f"envs: {len(table)} | median model NMSE {table.model_nmse.median():.4f} vs persistence "
        f"{table.persistence_nmse.median():.4f} | geomean(model/persistence) {np.exp(np.log(ratio).mean()):.3f} | "
        f"better than persistence in {(table.model_nmse < table.persistence_nmse).sum()}/{len(table)}",
        f"in-context ridge oracle: median NMSE {table.oracle_nmse.median():.4f} | model better than the "
        f"oracle in {(table.model_nmse < table.oracle_nmse).sum()}/{len(table)}",
        f"window position, median NMSE early (first {EARLY}) -> late (last {LATE}): model "
        f"{table.model_early.median():.4f} -> {table.model_late.median():.4f} | oracle "
        f"{table.oracle_early.median():.4f} -> {table.oracle_late.median():.4f}",
    ]
    if "shuffled_last" in table:
        lines.append(
            f"context use (last step): median NMSE {table.model_last.median():.4f} with the full context vs "
            f"{table.shuffled_last.median():.4f} time-shuffled vs {table.truncated_last.median():.4f} with only "
            f"the last {TRUNCATED} steps"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="directory of processed episode parquets")
    parser.add_argument("--windows", type=int, default=8, help="random windows per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-diagnostics", action="store_true", help="time-shuffle and truncation tests")
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for CIs (0 = off)")
    parser.add_argument("--observables-only", action="store_true", help="hide LATENT_STATES from model and oracle")
    parser.add_argument("--out", type=Path, help="optional CSV path for the per-env table")
    args = parser.parse_args()

    table = evaluate(load_policy(args.checkpoint), args.data, args.windows, args.seed, args.context_diagnostics,
                     args.bootstrap, hidden_states=LATENT_STATES if args.observables_only else ())
    pd.set_option("display.width", 250)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(summarize(table))
    curve = table.attrs["position_curve"]
    print("\nmedian NMSE by window position:")
    shown = sorted({p for p in (1, 2, 3, 5, 10, 20, 30) if p < len(curve)} | {len(curve) - 1})
    print(curve.iloc[shown].to_string(float_format=lambda x: f"{x:.4f}"))
    if args.out:
        table.to_csv(args.out, index=False)
        curve.to_csv(args.out.with_name(args.out.stem + "_curve.csv"))


if __name__ == "__main__":
    main()
