"""Summary of the env-count ablation: one row per finished run, plus the curve of test error against the
number of training envs (mean and std over seeds).

    python cluster/env_ablation/summarize.py <OUT_ROOT>

Writes OUT_ROOT/results.csv (per run), results_by_step.csv (per step), ratio_by_env.csv (per run and
test env) and curve.png. Safe to run while the ablation is still going.
"""
import json
from pathlib import Path
import sys

import pandas as pd

TABLES = ("test", "test_observables", "seen_val")


def main(out: Path) -> None:
    results = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*/result.json"))]
    if not results:
        print("no finished runs yet")
        return
    order = json.loads((out / "order.json").read_text())["train_envs"]
    rows, per_env = [], []
    for r in results:
        row = {"step": r["step"], "seed": r["seed"], "n_envs": r["n_envs"], "added_env": r["envs"][-1] if r["envs"] else "(LQ only)",
               "train_episodes": r["train_episodes"], "best_val_action_loss": r["best_val_action_loss"]}
        for t in TABLES:
            if t in r:
                row |= {f"{t}_geomean_ratio": r[t]["geomean_ratio"], f"{t}_median_nmse": r[t]["median_model_nmse"],
                        f"{t}_better_than_persistence": r[t]["better_than_persistence"]}
        for k in ("model_last", "shuffled_last", "truncated_last"):
            row[f"test_{k}"] = r["test"].get(k)
        rows.append(row)
        per_env += [{"step": r["step"], "seed": r["seed"], "env": env, "ratio": ratio}
                    for env, ratio in r["test"]["ratio_by_env"].items()]
    runs = pd.DataFrame(rows).sort_values(["step", "seed"])
    runs.to_csv(out / "results.csv", index=False)
    pd.DataFrame(per_env).to_csv(out / "ratio_by_env.csv", index=False)

    metrics = [c for c in runs if c.endswith("_geomean_ratio")]
    by_step = runs.groupby(["step", "added_env"], sort=False)[metrics].agg(["mean", "std", "count"])
    by_step.columns = ["_".join(c) for c in by_step.columns]
    by_step = by_step.reset_index().sort_values("step")
    by_step.to_csv(out / "results_by_step.csv", index=False)
    pd.set_option("display.width", 250)
    print(by_step.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n{len(runs)} runs; env order: {order}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t, label in (("test", "held-out families"), ("test_observables", "held-out families, latent states hidden"),
                     ("seen_val", "training envs, fresh draws")):
        if f"{t}_geomean_ratio_mean" in by_step:
            ax.errorbar(by_step.step, by_step[f"{t}_geomean_ratio_mean"], yerr=by_step[f"{t}_geomean_ratio_std"],
                        marker="o", capsize=3, label=label)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("training envs (0 = LQ tasks only)")
    ax.set_ylabel("geomean NMSE(model) / NMSE(persistence)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curve.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
