from __future__ import annotations

from typing import Annotated

import typer

from rageval.cli._console import console


def generate_evalset(
    docs_path: Annotated[str, typer.Argument(help="Path to document corpus")],
    num_questions: Annotated[int, typer.Option("--num-questions", "-n", help="Number of questions to generate")] = 50,
    output: Annotated[str, typer.Option("--output", "-o", help="Output JSONL file")] = "evalsets/auto.jsonl",
    diversity: Annotated[str, typer.Option("--diversity", help="Question diversity mode")] = "mixed",
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
) -> None:
    """Synthesize an eval set from a document corpus using an LLM."""
    console.print("[yellow]not yet implemented[/yellow]")
