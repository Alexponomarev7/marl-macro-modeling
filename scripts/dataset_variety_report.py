from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import typer


def _load_dataset_module() -> Any:
    """
    Load `lib/dataset.py` as a module without importing the `lib` package.

    This avoids side effects from `lib/__init__.py` (which imports `lib.config` and calls dotenv).
    """
    import importlib.util

    dataset_py = Path(__file__).resolve().parents[1] / "lib" / "dataset.py"
    spec = importlib.util.spec_from_file_location("dataset_mod", dataset_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for: {dataset_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_markdown_report(report: dict, out_path: Path, dataset_path: Path) -> None:
    """
    Write a Markdown report with an overall summary and a per-environment section.

    Args:
        report: Output of DatasetDiversityScorer.generate_report()
        out_path: Markdown file path to write.
        dataset_path: Dataset root that was scored (for display only).
    """
    overall = report["overall"]
    per_env = report["per_env"]  # pandas DataFrame

    lines: list[str] = []
    lines.append("## Dataset variety report")
    lines.append("")
    lines.append(f"- **Dataset path**: `{dataset_path}`")
    lines.append(f"- **Generated at**: `{datetime.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Metrics Reference")
    lines.append("")
    lines.append("This report measures dataset variety using several metrics:")
    lines.append("")
    lines.append("### Within-Model Metrics (Episode-to-Episode Variety)")
    lines.append("")
    lines.append("These metrics measure diversity between episodes of the same economic model:")
    lines.append("")
    lines.append("- **mean_pairwise_coverage**: Measures how well episodes cover the state-action space.")
    lines.append("  - For each episode and each state-action pair (s_i, a_j), a 2D grid is constructed using quantile binning.")
    lines.append("  - Coverage for a pair = (number of occupied cells) / (total cells in grid)")
    lines.append("  - Episode-level coverage = mean over all state-action pairs")
    lines.append("  - Model-level coverage = mean over all episodes")
    lines.append("  - Range: [0, 1], where 1 = full coverage, 0 = minimal coverage")
    lines.append("")
    lines.append("- **mean_episode_embedding_knn**: Measures diversity of episode trajectories using k-nearest neighbor distances.")
    lines.append("  - Each episode is embedded by extracting time-series features (mean, std, quantiles, skewness, kurtosis, autocorrelation, trend, energy) for each state variable")
    lines.append("  - For each episode, compute average distance to its k nearest neighbors (excluding itself) in the standardized embedding space")
    lines.append("  - Model-level metric = mean over all episodes")
    lines.append("  - Range: [0, ∞), where larger values indicate more diverse episodes")
    lines.append("")
    lines.append("### Cross-Model Metrics (Environment-to-Environment Variety)")
    lines.append("")
    lines.append("These metrics measure similarity and redundancy between different economic models:")
    lines.append("")
    lines.append("- **shared_state_frac**: Fraction of state variables shared with the nearest other model.")
    lines.append("  - Formula: |S_i ∩ S_j| / |S_i|, where S_i is the state space of model i and S_j is the state space of the nearest model j")
    lines.append("  - The nearest model is identified as the one with the highest shared_state_frac")
    lines.append("  - Range: [0, 1], where 1 = all states shared, 0 = no shared states")
    lines.append("")
    lines.append("- **intersection_over_union**: Jaccard similarity of state spaces with the nearest model.")
    lines.append("  - Formula: |S_i ∩ S_j| / |S_i ∪ S_j|")
    lines.append("  - Range: [0, 1], where 1 = identical state spaces, 0 = no overlap")
    lines.append("")
    lines.append("- **intra_over_inter**: Ratio of within-model diversity to between-model diversity (restricted to shared states).")
    lines.append("  - intra = average k-NN distance within each model's episodes (averaged across both models)")
    lines.append("  - inter = average distance from episodes in one model to their k nearest neighbors in the other model (symmetrized)")
    lines.append("  - Formula: intra / inter")
    lines.append("  - Range: [0, ∞), where smaller values indicate well-separated models (more unique), larger values indicate overlapping models")
    lines.append("")
    lines.append("### Overall Summary Metrics")
    lines.append("")
    lines.append("The overall summary shows averages of the above metrics across all models in the dataset.")
    lines.append("")

    lines.append("## Overall summary")
    lines.append("")
    for k, v in overall.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    lines.append("## Per-environment report")
    lines.append("")

    name_col = "env_group" if "env_group" in per_env.columns else "env_name"

    if "mean_pairwise_coverage" in per_env.columns and name_col in per_env.columns:
        per_env = per_env.sort_values(
            ["mean_pairwise_coverage", name_col], ascending=[False, True]
        )
    elif name_col in per_env.columns:
        per_env = per_env.sort_values([name_col], ascending=[True])

    # Define the nearest environment comparison metrics
    nearest_env_metrics = {"nearest_env", "shared_state_frac", "intersection_over_union", "intra_over_inter", "intra", "inter"}

    for _, row in per_env.iterrows():
        env = row.get(name_col, "unknown")
        lines.append(f"### {env}")
        lines.append("")

        # First, output all non-nearest-env metrics
        for col in per_env.columns:
            if col == name_col or col in nearest_env_metrics:
                continue
            val = row[col]
            # Make list-like fields readable
            if isinstance(val, (list, tuple)):
                lines.append(f"- **{col}**: `{len(val)}` items")
                if len(val) <= 50:
                    lines.append("")
                    lines.append("```")
                    lines.append(", ".join(map(str, val)))
                    lines.append("```")
                continue
            lines.append(f"- **{col}**: `{val}`")

        # Then, create a subsection for nearest environment comparison
        has_nearest_env_metrics = any(col in per_env.columns for col in {"nearest_env", "shared_state_frac", "intersection_over_union", "intra_over_inter"})
        if has_nearest_env_metrics:
            lines.append("")
            lines.append("#### Nearest Environment Comparison")
            lines.append("")

            # Display nearest_env
            if "nearest_env" in per_env.columns:
                val = row["nearest_env"]
                lines.append(f"- **nearest_env**: `{val}`")

            # Display shared_state_frac
            if "shared_state_frac" in per_env.columns:
                val = row["shared_state_frac"]
                lines.append(f"- **shared_state_frac**: `{val}`")

            # Display intersection_over_union
            if "intersection_over_union" in per_env.columns:
                val = row["intersection_over_union"]
                lines.append(f"- **intersection_over_union**: `{val}`")

            # Display intra_over_inter with breakdown
            if "intra_over_inter" in per_env.columns:
                intra_val = row.get("intra")
                inter_val = row.get("inter")
                ratio_val = row["intra_over_inter"]

                if intra_val is not None and inter_val is not None and ratio_val is not None:
                    lines.append(f"- **intra_over_inter**: `{intra_val}` / `{inter_val}` = `{ratio_val}`")
                elif ratio_val is not None:
                    lines.append(f"- **intra_over_inter**: `{ratio_val}`")
                else:
                    lines.append(f"- **intra_over_inter**: `None`")

        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    dataset_path: Path = typer.Argument(..., exists=True, dir_okay=True, file_okay=False),
    quantile_bins: int = typer.Option(10, "--quantile-bins", "-b", min=2),
    knn_k: int = typer.Option(5, "--knn-k", "-k", min=1),
    cache_parquets: bool = typer.Option(True, "--cache-parquets/--no-cache-parquets"),
    report_md: Path = typer.Option(
        Path("reports/dataset_variety_report.md"),
        "--report-md",
        help="Where to write the markdown report.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="If set, also write per_env.csv and overall.json to this folder.",
    ),
) -> None:
    """
    Generate dataset variety report using `DatasetDiversityScorer`.
    """
    mod = _load_dataset_module()
    scorer = mod.DatasetDiversityScorer(
        dataset_path=dataset_path,
        quantile_bins=quantile_bins,
        knn_k=knn_k,
        cache_parquets=cache_parquets,
    )
    report = scorer.generate_report()

    _write_markdown_report(report, report_md, dataset_path)
    typer.echo(f"Wrote markdown report: {report_md}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report["per_env"].to_csv(out_dir / "per_env.csv", index=False)
        (out_dir / "overall.json").write_text(json.dumps(report["overall"], indent=2, default=str))
        typer.echo(f"Wrote: {out_dir / 'per_env.csv'}")
        typer.echo(f"Wrote: {out_dir / 'overall.json'}")


if __name__ == "__main__":
    typer.run(main)


