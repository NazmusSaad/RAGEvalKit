# RAGEvalKit Architecture

## Overview

RAGEvalKit is a CLI-first evaluation framework. Every component is a standalone Python module with a corresponding CLI command. There is no backend server; all state is stored locally in `.rageval/`.

## Directory Layout

```
RAGEvalKit/
├── src/rageval/
│   ├── cli/                  Typer command handlers
│   ├── core/                 Domain logic (config, ingestion, pipeline, embedder, etc.)
│   ├── evaluators/           Metric computation (retrieval, answer relevance, claims, groundedness)
│   ├── evalset/              Evalset generation and loading
│   ├── ci/                   CI threshold gate logic
│   ├── report/               HTML report rendering (Jinja2)
│   └── storage/              DuckDB DAO and Chroma DAO
├── examples/
│   ├── tiny-corpus/          3-doc dev corpus (mock/dev mode)
│   ├── demo-corpus/          3-doc realistic corpus (live OpenAI demo)
│   └── configs/              YAML pipeline configs
├── prompts/                  RAG system prompt and Jinja2 answer template
├── tests/
│   ├── unit/                 Fast, no I/O, no model downloads
│   └── integration/          CLI-level tests using tmp_path
└── .rageval/                 Runtime workspace (created by rageval init)
    ├── runs.db               DuckDB database
    └── chroma/               ChromaDB persistent client data
```

## Storage

### DuckDB (`runs.db`)

All structured data is stored in a single DuckDB file. The schema (defined in `storage/schema.sql`) contains:

| Table | Purpose |
|-------|---------|
| `documents` | One row per ingested source file |
| `chunks` | One row per text chunk (FK → documents) |
| `eval_sets` | Named evalset metadata |
| `eval_questions` | One row per question (FK → eval_sets) |
| `runs` | One row per pipeline run |
| `run_items` | One row per evaluated question in a run |
| `retrieved_contexts` | Ranked chunks retrieved for each run item |
| `metric_scores` | One score row per (run_item, metric) |
| `claim_evaluations` | One row per atomic claim extracted from an answer |
| `root_causes` | Primary and secondary root cause per run item |

All writes are idempotent: existing rows are deleted and re-inserted on re-evaluation. FK constraints are enforced.

### ChromaDB

One `PersistentClient` per config, stored under `.rageval/chroma/`. Each pipeline config names a distinct collection (e.g. `demo_openai`, `docs_dev`) to prevent cross-contamination between different ingestion runs.

## Evaluation Pipeline

```
ingest → generate-evalset → run → evaluate-retrieval
                                 → evaluate-answer-relevance
                                 → extract-claims → evaluate-groundedness
                                 → summarize-run
                                 → compare / ci-check / report
```

Each step is independent and idempotent. You can re-run any evaluator without re-running the full pipeline.

## LLM Abstraction

`core/llm.py` defines an `LLMClient` protocol with a single `complete(prompt) -> str` method. Implementations:

| Client | Usage |
|--------|-------|
| `MockLLMClient` | Tests and dev mode; returns configurable canned JSON |
| `OpenAIClient` | Real calls via `openai` SDK; reads `OPENAI_API_KEY` from env |

Each CLI evaluator module exposes a `_build_judge_client(config)` helper that returns `MockLLMClient` when `provider="mock"` and `OpenAIClient` otherwise.

## Embedding Abstraction

`core/embedder.py` defines an `Embedder` protocol with `embed(texts) -> list[list[float]]`. Implementations:

| Embedder | Usage |
|----------|-------|
| `DummyEmbedder` | Tests and dev mode; SHA-256-based deterministic 16-dim unit-normalized vectors |
| `SentenceTransformerEmbedder` | Real local embeddings via `sentence-transformers` |

## Config

All pipeline configuration lives in a YAML file validated by Pydantic v2 (`core/config.py`). The top-level model is `PipelineConfig`. Threshold configuration for CI checks lives in a separate `ThresholdsConfig` (also YAML). All threshold fields default to `None` (opt-in).

## CLI Layer

Each command is a thin Typer handler in `cli/`. Heavy imports (chromadb, sentence-transformers, openai) are deferred to function bodies to keep `rageval --help` and `rageval init` fast.

## Report

`report/render.py` collects data from DuckDB into `ReportData` and renders it through the Jinja2 template `report/templates/run_report.html.j2`. The output is a single self-contained HTML file with inline CSS. No JavaScript, no external CDN.
