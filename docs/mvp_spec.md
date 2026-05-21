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

### Completed: Milestone 2D
- Implemented `rageval ingest`
- Implemented `rageval retrieve`
- Added full ingestion pipeline: load → chunk → embed → DuckDB → Chroma
- Added `dummy` embedding provider for tests/dev
- Added CLI integration tests for ingest/retrieve
- Verified idempotent ingestion and retrieval from Chroma

### Completed: Cleanup after Milestone 2
- Moved Chroma import to lazy import path
- Kept `rageval init` and `rageval --help` lightweight
- Updated SentenceTransformer embedding dimension lookup to avoid deprecated API warning

### Current: Milestone 3A
Goal: add LLM client abstraction and synthetic evalset generation.

Scope:
- Add OpenAI-compatible LLM client abstraction
- Add `MockLLMClient` for tests
- Add prompt template for synthetic eval question generation
- Implement `rageval generate-evalset`
- Generate JSONL eval sets from already-ingested chunks
- Store eval sets/questions in DuckDB
- Add tests using mocked LLM responses only

Do not implement yet:
- RAG answer generation
- `rageval run`
- retrieval/generation evaluators
- claim extraction
- groundedness judging
- reports
- compare/ci-check logic
- GitHub Actions
- Docker