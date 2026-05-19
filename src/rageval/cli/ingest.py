from __future__ import annotations

from typing import Annotated

import typer

from rageval.cli._console import console


def ingest(
    path: Annotated[str, typer.Argument(help="Path to file or directory to ingest")],
    config: Annotated[str, typer.Option("--config", "-c", help="Pipeline config YAML")] = "configs/baseline.yaml",
    collection: Annotated[str | None, typer.Option("--collection", help="ChromaDB collection name")] = None,
    chunk_size: Annotated[int | None, typer.Option("--chunk-size", help="Override chunk size")] = None,
    chunk_overlap: Annotated[int | None, typer.Option("--chunk-overlap", help="Override chunk overlap")] = None,
) -> None:
    """Load documents, chunk, embed, and persist to ChromaDB."""
    console.print("[yellow]not yet implemented[/yellow]")
