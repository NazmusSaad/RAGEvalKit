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
- Lightweight custom ingestion/retrieval for MVP; avoid LlamaIndex unless needed later
- Static HTML report later
- No Streamlit/FastAPI/Postgres/auth/model training in MVP

## MVP Commands
- rageval init
- rageval ingest
- rageval retrieve
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

## Progress

### Completed: Milestone 1
- Installable Python package
- Typer CLI with command stubs
- Working `rageval init`
- DuckDB schema initialized
- Pydantic config validation
- Unit tests for help/init/schema/config

### Completed: Milestone 2A
- Stable document/chunk ID helpers
- `.md` / `.txt` document loader
- Recursive directory loading
- Deterministic `SimpleChunker`
- Tiny example corpus
- Unit tests for IDs, loading, and chunking

### Completed: Milestone 2B
- DuckDB DAO methods for documents and chunks
- Idempotent document/chunk upserts
- `ingest_documents_to_duckdb()` service
- Integration tests for tiny-corpus ingestion
- No Chroma or embeddings in this step

### Completed: Milestone 2C
- `Embedder` protocol
- `DummyEmbedder` for tests/dev
- `SentenceTransformerEmbedder` for real local embeddings
- `create_embedder(config)` factory
- Chroma vector storage DAO
- `retrieve_top_k()` retrieval primitive
- Unit tests for embeddings, Chroma upsert/query, and retrieval
- Tests use `DummyEmbedder` and `tmp_path`; no model downloads

### Current: Milestone 2D
Goal: wire ingestion and retrieval into real CLI commands.

Scope:
- Implement `rageval ingest <path> --config configs/baseline.yaml`
- Load documents with existing loader
- Chunk with `SimpleChunker`
- Store documents/chunks in DuckDB
- Embed chunks with configured embedder
- Store vectors in Chroma
- Implement `rageval retrieve "<query>" --top-k 3 --config configs/baseline.yaml`
- Print ranked retrieved chunks with scores and text previews
- Add CLI integration tests using `DummyEmbedder`
- Ensure tests use `tmp_path` and do not write to the real `.rageval/`

Do not implement yet:
- LLM answer generation
- evalset generation
- judge/evaluator prompts
- `rageval run`
- report generation
- CI threshold logic
- GitHub Actions
- Docker