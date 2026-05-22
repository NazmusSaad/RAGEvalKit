from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.core.config import PipelineConfig
from rageval.core.llm import MockLLMClient, create_llm_client
from rageval.evaluators.answer_relevance import evaluate_answer_relevance_for_item

# Task-specific mock response used when judge.provider == "mock".
# Produces score=0.75 (pass) so `rageval evaluate-answer-relevance` gives meaningful
# output in dev mode instead of all-unknown from the default question-gen response.
_MOCK_JUDGE_RESPONSE = '{"score": 3, "reason": "Mock answer addresses the question."}'


def _build_judge_client(judge_config):
    """Build the judge LLM client.

    When ``provider == "mock"``, returns a :class:`MockLLMClient` pre-loaded with
    answer-relevance JSON so manual dev runs produce real scores.  All other
    providers delegate to :func:`create_llm_client`.
    """
    if judge_config.provider == "mock":
        return MockLLMClient(response_text=_MOCK_JUDGE_RESPONSE, model=judge_config.model)
    return create_llm_client(judge_config)
from rageval.storage.duckdb_dao import (
    get_connection,
    get_run_by_id,
    get_run_items_for_evaluation,
    insert_metric_score,
)


def evaluate_answer_relevance(
    run: Annotated[str, typer.Option("--run", "-r", help="Run ID to evaluate")],
    config: Annotated[str, typer.Option("--config", "-c", help="Pipeline config YAML")] = "configs/baseline.yaml",
) -> None:
    """Score answer relevance for each item in a completed run using the judge LLM."""
    project_dir = Path.cwd()

    if not (project_dir / ".rageval").exists():
        console.print(
            "[red]Error:[/red] .rageval/ not found.\n"
            "  Run [bold]rageval init[/bold] first."
        )
        raise typer.Exit(1)

    config_path = project_dir / config
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    try:
        pipeline_config = PipelineConfig.from_yaml(config_path)
    except Exception as exc:
        console.print(f"[red]Error loading config:[/red] {exc}")
        raise typer.Exit(1)

    llm_client = _build_judge_client(pipeline_config.judge)

    db_path = project_dir / ".rageval" / "runs.db"
    con = get_connection(db_path)

    try:
        run_row = get_run_by_id(con, run)
        if run_row is None:
            console.print(f"[red]Error:[/red] Run not found: {run!r}")
            raise typer.Exit(1)

        items = get_run_items_for_evaluation(con, run)
        if not items:
            console.print("[yellow]No run items found for this run.[/yellow]")
            return

        pass_count = 0
        fail_count = 0
        unknown_count = 0
        scored: list[float] = []

        for item in items:
            result = evaluate_answer_relevance_for_item(
                question=item["question"],
                generated_answer=item["generated_answer"],
                llm_client=llm_client,
            )
            raw_json_str = json.dumps(result.raw_json) if result.raw_json else None
            insert_metric_score(
                con,
                item_id=item["item_id"],
                metric=result.metric,
                score=result.score,
                label=result.label,
                reason=result.reason,
                judge_model=pipeline_config.judge.model,
                raw_json=raw_json_str,
            )
            if result.label == "pass":
                pass_count += 1
                scored.append(result.score)
            elif result.label == "fail":
                fail_count += 1
                scored.append(result.score)
            else:
                unknown_count += 1

    finally:
        con.close()

    n_total = len(items)
    mean_str = f"{sum(scored) / len(scored):.3f}" if scored else "N/A (all unknown)"

    table = Table(title="Answer Relevance Evaluation Summary", show_header=True)
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Run ID", run[:16] + "...")
    table.add_row("Judge model", pipeline_config.judge.model)
    table.add_row("Items evaluated", str(n_total))
    table.add_row("Pass (score >= 0.75)", str(pass_count))
    table.add_row("Fail (score < 0.75)", str(fail_count))
    table.add_row("Unknown", str(unknown_count))
    table.add_row("Mean answer relevance", mean_str)
    console.print(table)
