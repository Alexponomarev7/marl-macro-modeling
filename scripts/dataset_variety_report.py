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

    for _, row in per_env.iterrows():
        env = row.get(name_col, "unknown")
        lines.append(f"### {env}")
        lines.append("")
        for col in per_env.columns:
            if col == name_col:
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


