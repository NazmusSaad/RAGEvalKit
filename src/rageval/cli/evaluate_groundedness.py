from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from rageval.cli._console import console
from rageval.core.config import PipelineConfig
from rageval.core.llm import MockLLMClient, create_llm_client
from rageval.evaluators.groundedness import evaluate_groundedness_for_item
from rageval.storage.duckdb_dao import (
    get_claims_for_item,
    get_connection,
    get_retrieved_contexts_for_item,
    get_run_by_id,
    get_run_items_for_evaluation,
    insert_metric_score,
    update_claim_evaluation,
)

# Task-specific mock: all claims "supported" → faithfulness=1.0 in dev mode.
_MOCK_JUDGE_RESPONSE = json.dumps({
    "verdict": "supported",
    "supporting_indices": [0],
    "rationale": "Mock claim is treated as supported by the top context.",
})


def _build_judge_client(judge_config):
    """Build the judge LLM client.

    When ``provider == "mock"``, returns a :class:`MockLLMClient` pre-loaded
    with a groundedness-appropriate response so dev runs produce faithfulness=1.0.
    """
    if judge_config.provider == "mock":
        return MockLLMClient(response_text=_MOCK_JUDGE_RESPONSE, model=judge_config.model)
    return create_llm_client(judge_config)


def evaluate_groundedness(
    run: Annotated[str, typer.Option("--run", "-r", help="Run ID to evaluate")],
    config: Annotated[str, typer.Option("--config", "-c", help="Pipeline config YAML")] = "configs/baseline.yaml",
) -> None:
    """Judge groundedness of extracted claims against retrieved context."""
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

        # Guard: require claims to exist
        total_claims_in_run = sum(
            len(get_claims_for_item(con, item["item_id"])) for item in items
        )
        if total_claims_in_run == 0:
            console.print(
                "[yellow]No claims found for this run.[/yellow]\n"
                "  Run [bold]rageval extract-claims[/bold] first."
            )
            raise typer.Exit(1)

        # Accumulators
        claim_supported = 0
        claim_contradicted = 0
        claim_not_enough_info = 0
        claim_unknown_verdict = 0
        total_claims_judged = 0
        item_unknown_count = 0
        scored_faithfulness: list[float] = []

        for item in items:
            claims = get_claims_for_item(con, item["item_id"])
            contexts = get_retrieved_contexts_for_item(con, item["item_id"])

            item_result = evaluate_groundedness_for_item(claims, contexts, llm_client)

            # Update each claim row with verdict / supporting_chunk_ids / rationale
            for claim, cr in zip(claims, item_result.claim_results):
                supporting_chunk_ids = [
                    contexts[idx]["chunk_id"]
                    for idx in cr.supporting_indices
                    if 0 <= idx < len(contexts)
                ]
                update_claim_evaluation(
                    con,
                    item_id=item["item_id"],
                    claim_idx=claim["claim_idx"],
                    verdict=cr.verdict,
                    supporting_chunk_ids=json.dumps(supporting_chunk_ids),
                    rationale=cr.rationale,
                )
                if cr.verdict == "supported":
                    claim_supported += 1
                elif cr.verdict == "contradicted":
                    claim_contradicted += 1
                elif cr.verdict == "not_enough_info":
                    claim_not_enough_info += 1
                else:
                    claim_unknown_verdict += 1
                total_claims_judged += 1

            insert_metric_score(
                con,
                item_id=item["item_id"],
                metric="faithfulness",
                score=item_result.faithfulness,
                label=item_result.label,
                reason=item_result.reason,
                judge_model=pipeline_config.judge.model,
            )

            if item_result.label == "unknown":
                item_unknown_count += 1
            else:
                scored_faithfulness.append(item_result.faithfulness)

    finally:
        con.close()

    n_total = len(items)
    mean_str = (
        f"{sum(scored_faithfulness) / len(scored_faithfulness):.3f}"
        if scored_faithfulness
        else "N/A (all unknown)"
    )

    table = Table(title="Groundedness Evaluation Summary", show_header=True)
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Run ID", run[:16] + "...")
    table.add_row("Judge model", pipeline_config.judge.model)
    table.add_row("Items evaluated", str(n_total))
    table.add_row("Total claims judged", str(total_claims_judged))
    table.add_row("Supported", str(claim_supported))
    table.add_row("Contradicted", str(claim_contradicted))
    table.add_row("Not enough info", str(claim_not_enough_info))
    table.add_row("Unknown verdict", str(claim_unknown_verdict))
    table.add_row("Items unknown", str(item_unknown_count))
    table.add_row("Mean faithfulness", mean_str)
    console.print(table)
