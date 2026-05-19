# RAGEvalKit MVP Spec

## Product
CLI-first RAG evaluation and regression-testing framework.
Goal: evaluate RAG pipeline quality, diagnose failures, compare versions, and fail CI on regressions.

## MVP Architecture Decisions
- Python 3.11+
- CLI-first, no backend server
- Typer + Rich CLI
- Pydantic v2 config validation
- DuckDB for run/eval storage
- ChromaDB for vectors starting Milestone 2
- LlamaIndex or lightweight custom ingestion/retrieval later
- Static HTML report later
- No Streamlit/FastAPI/Postgres/auth/model training in MVP

## MVP Commands
- rageval init
- rageval ingest
- rageval generate-evalset
- rageval run
- rageval compare
- rageval report
- rageval ci-check
- rageval inspect

## Milestones
1. Skeleton CLI + config + DuckDB schema + init
2. Ingest/chunk/embed/retrieve
3. Generate eval set + run RAG + store traces
4. Evaluators: retrieval relevance, claim extraction, groundedness, answer relevance
5. Compare + ci-check
6. Static HTML report
7. README, Docker, GitHub Action, demo

## Milestone 1 Scope
Implement only:
- installable Python package
- CLI stubs for all commands
- working rageval init
- DuckDB schema
- Pydantic config models
- starter configs/prompts
- tests for CLI help, init files, DB tables, config validation

Do not implement:
- Chroma
- embeddings
- LLM calls
- document ingestion
- eval generation
- reports
- GitHub Action