from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from rageval.cli._console import console
from rageval.storage.duckdb_dao import init_db

_BASELINE_YAML = """\
version: 1
name: baseline
seed: 42

corpus:
  path: ./docs
  glob: "**/*.{md,pdf,txt}"

chunking:
  strategy: recursive
  chunk_size: 512
  chunk_overlap: 64

embedding:
  provider: sentence_transformers
  model: BAAI/bge-small-en-v1.5
  batch_size: 64

vector_store:
  type: chroma
  path: .rageval/chroma
  collection: docs_v1
  distance: cosine

retrieval:
  top_k: 5
  rerank: null
  filter: null

generation:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 512
  system_prompt_path: prompts/system.txt
  prompt_template_path: prompts/rag.j2

judge:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_concurrent: 4

evaluators:
  - retrieval_relevance
  - groundedness
  - answer_relevance

evalset:
  path: evalsets/v1.jsonl

cost:
  input_per_1k: 0.015
  output_per_1k: 0.060
"""

_EXPERIMENT_YAML = """\
version: 1
name: experiment
seed: 42

corpus:
  path: ./docs
  glob: "**/*.{md,pdf,txt}"

chunking:
  strategy: recursive
  chunk_size: 256
  chunk_overlap: 32

embedding:
  provider: sentence_transformers
  model: BAAI/bge-small-en-v1.5
  batch_size: 64

vector_store:
  type: chroma
  path: .rageval/chroma
  collection: docs_v1
  distance: cosine

retrieval:
  top_k: 3
  rerank: null
  filter: null

generation:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 512
  system_prompt_path: prompts/system.txt
  prompt_template_path: prompts/rag.j2

judge:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_concurrent: 4

evaluators:
  - retrieval_relevance
  - groundedness
  - answer_relevance

evalset:
  path: evalsets/v1.jsonl

cost:
  input_per_1k: 0.015
  output_per_1k: 0.060
"""

_RAGEVAL_YAML = """\
version: 1

absolute:
  faithfulness_min: 0.80
  retrieval_relevance_min: 0.70

relative:
  faithfulness_drop_max: 0.05
  retrieval_relevance_drop_max: 0.05
  recall_at_k_drop_max: 0.05
  answer_relevance_drop_max: 0.05
  p50_latency_increase_max: 0.30
  cost_per_query_increase_max: 0.30

policy:
  require_all_absolute: true
  require_all_relative: true
  allow_unknown_as_pass: false
"""

_SYSTEM_TXT = """\
You are a helpful AI assistant. Answer questions accurately and concisely based on the provided context.
If the context does not contain enough information to answer the question, say so clearly.
Do not make up information that is not in the context.
"""

_RAG_J2 = """\
{% for chunk in chunks %}
[{{ loop.index }}] {{ chunk.text }}
{% endfor %}

Question: {{ question }}

Answer:
"""


def _update_gitignore(project_dir: Path) -> None:
    gitignore = project_dir / ".gitignore"
    entry = ".rageval/"
    if gitignore.exists():
        content = gitignore.read_text()
        if not any(line.strip() == entry for line in content.splitlines()):
            with gitignore.open("a") as f:
                f.write(f"\n{entry}\n")
    else:
        gitignore.write_text(f"{entry}\n")


def init(
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing .rageval/ directory")] = False,
) -> None:
    """Scaffold a new RAGEvalKit project in the current directory."""
    project_dir = Path.cwd()
    rageval_dir = project_dir / ".rageval"

    if rageval_dir.exists() and not force:
        console.print(
            "[yellow]Warning:[/yellow] .rageval/ already exists.\n"
            "  Run [bold]rageval init --force[/bold] to reinitialize."
        )
        raise typer.Exit(1)

    created: list[str] = []

    rageval_dir.mkdir(exist_ok=True)
    db_path = rageval_dir / "runs.db"
    init_db(db_path)
    created.append(db_path.relative_to(project_dir).as_posix())

    configs_dir = project_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    for filename, content in [("baseline.yaml", _BASELINE_YAML), ("experiment.yaml", _EXPERIMENT_YAML)]:
        p = configs_dir / filename
        p.write_text(content)
        created.append(p.relative_to(project_dir).as_posix())

    thresholds_path = project_dir / "rageval.yaml"
    thresholds_path.write_text(_RAGEVAL_YAML)
    created.append("rageval.yaml")

    prompts_dir = project_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    for filename, content in [("system.txt", _SYSTEM_TXT), ("rag.j2", _RAG_J2)]:
        p = prompts_dir / filename
        p.write_text(content)
        created.append(p.relative_to(project_dir).as_posix())

    _update_gitignore(project_dir)
    created.append(".gitignore")

    lines = "\n".join(f"  [green]+[/green] {p}" for p in created)
    console.print(Panel(lines, title="[bold green]rageval init[/bold green]", expand=False))
