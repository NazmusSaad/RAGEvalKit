from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.evaluators.compare import RunComparison, compare_runs
from rageval.evaluators.summary import build_run_summary
from rageval.storage.duckdb_dao import (
    get_connection,
    get_metric_scores_for_run,
    get_root_cause_distribution,
    get_run_by_id,
    get_run_items_basic,
    get_run_metric_means,
)

_ALL_CAUSES = [
    "none",
    "retrieval_failure",
    "grounding_failure",
    "answer_relevance_failure",
    "judge_uncertain",
    "missing_metric",
]

_DIRECTION_STYLE: dict[str, str] = {
    "improved": "[green]improved[/green]",
    "regressed": "[red]regressed[/red]",
    "unchanged": "unchanged",
    "n/a": "N/A",
}


def _compute_label_counts(con, run_id: str) -> dict[str, int]:
    """Compute overall item label counts using summary logic (no summarize-run needed)."""
    items_data = get_run_items_basic(con, run_id)
    scores_data = get_metric_scores_for_run(con, run_id)
    run_summary = build_run_summary(run_id, items_data, scores_data)
    return {
        "pass": run_summary.pass_count,
        "fail": run_summary.fail_count,
        "unknown": run_summary.unknown_count,
    }


def compare(
    baseline: Annotated[str, typer.Option("--baseline", "-b", help="Baseline run ID")],
    candidate: Annotated[str, typer.Option("--candidate", "-c", help="Candidate run ID")],
) -> None:
    """Compare two completed runs and print metric deltas."""
    project_dir = Path.cwd()

    if not (project_dir / ".rageval").exists():
        console.print(
            "[red]Error:[/red] .rageval/ not found.\n"
            "  Run [bold]rageval init[/bold] first."
        )
        raise typer.Exit(1)

    db_path = project_dir / ".rageval" / "runs.db"
    con = get_connection(db_path)

    try:
        baseline_row = get_run_by_id(con, baseline)
        if baseline_row is None:
            console.print(f"[red]Error:[/red] Baseline run not found: {baseline!r}")
            raise typer.Exit(1)

        candidate_row = get_run_by_id(con, candidate)
        if candidate_row is None:
            console.print(f"[red]Error:[/red] Candidate run not found: {candidate!r}")
            raise typer.Exit(1)

        baseline_means = get_run_metric_means(con, baseline)
        candidate_means = get_run_metric_means(con, candidate)

        baseline_labels = _compute_label_counts(con, baseline)
        candidate_labels = _compute_label_counts(con, candidate)

        baseline_causes = get_root_cause_distribution(con, baseline)
        candidate_causes = get_root_cause_distribution(con, candidate)

    finally:
        con.close()

    comparison = compare_runs(
        baseline_metric_means=baseline_means,
        candidate_metric_means=candidate_means,
        baseline_label_counts=baseline_labels,
        candidate_label_counts=candidate_labels,
        baseline_root_causes=baseline_causes,
        candidate_root_causes=candidate_causes,
        baseline_run_id=baseline,
        candidate_run_id=candidate,
    )

    # --- run info header ---
    console.print(
        f"\n[bold]Baseline:[/bold]  {baseline[:16]}...  tag=[bold]{baseline_row['tag']}[/bold]"
    )
    console.print(
        f"[bold]Candidate:[/bold] {candidate[:16]}...  tag=[bold]{candidate_row['tag']}[/bold]\n"
    )

    # --- metric delta table ---
    metric_table = Table(title="Metric Comparison", show_header=True)
    metric_table.add_column("Metric", style="bold")
    metric_table.add_column("Baseline", justify="right")
    metric_table.add_column("Candidate", justify="right")
    metric_table.add_column("Delta", justify="right")
    metric_table.add_column("Status")

    for delta in comparison.metric_deltas:
        b_str = f"{delta.baseline_mean:.3f}" if delta.baseline_mean is not None else "N/A"
        c_str = f"{delta.candidate_mean:.3f}" if delta.candidate_mean is not None else "N/A"
        d_str = f"{delta.absolute_delta:+.3f}" if delta.absolute_delta is not None else "N/A"
        s_str = _DIRECTION_STYLE.get(delta.direction, delta.direction)
        metric_table.add_row(delta.metric, b_str, c_str, d_str, s_str)

    console.print(metric_table)

    # --- label counts table ---
    label_table = Table(title="Overall Item Labels", show_header=True)
    label_table.add_column("Label", style="bold")
    label_table.add_column("Baseline", justify="right")
    label_table.add_column("Candidate", justify="right")
    for label in ("pass", "fail", "unknown"):
        label_table.add_row(
            label,
            str(baseline_labels.get(label, 0)),
            str(candidate_labels.get(label, 0)),
        )
    console.print(label_table)

    # --- root-cause distribution ---
    if not baseline_causes and not candidate_causes:
        console.print(
            "\n[yellow]Root causes not found.[/yellow] "
            "Run [bold]rageval summarize-run[/bold] for each run to populate "
            "root-cause distributions."
        )
    else:
        all_cause_keys = sorted(
            {c for d in (baseline_causes, candidate_causes) for c in d},
            key=lambda c: _ALL_CAUSES.index(c) if c in _ALL_CAUSES else 99,
        )
        cause_table = Table(title="Root-Cause Distribution", show_header=True)
        cause_table.add_column("Cause", style="bold")
        cause_table.add_column("Baseline", justify="right")
        cause_table.add_column("Candidate", justify="right")
        for cause in all_cause_keys:
            cause_table.add_row(
                cause,
                str(baseline_causes.get(cause, 0)),
                str(candidate_causes.get(cause, 0)),
            )
        console.print(cause_table)
