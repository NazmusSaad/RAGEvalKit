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

### Completed: Milestone 3A
- Added LLM client abstraction
- Added `MockLLMClient` for tests
- Added lazy OpenAI client path
- Implemented synthetic evalset generation from ingested chunks
- Implemented `rageval generate-evalset`
- Added JSONL evalset output
- Stored eval_sets and eval_questions in DuckDB
- Added tests with no real API calls

### Completed: Milestone 3B
- Implemented `rageval run`
- Added evalset JSONL loader
- Added `RAGPipeline` and `QueryTrace`
- Added RAG prompt rendering with retrieved contexts
- Stored runs, run_items, and retrieved_contexts in DuckDB
- Added deterministic run item IDs
- Added `--limit` for cheap local runs
- Added per-question error capture
- Verified manual dev flow with zero errors

### Completed: Milestone 4A
- Added deterministic retrieval metrics
- Implemented recall@k
- Implemented MRR
- Used evalset `source_chunk_ids` as retrieval ground truth
- Added `rageval evaluate-retrieval`
- Stored retrieval metric scores in DuckDB
- Added idempotent metric insertion into `metric_scores`
- Verified manual retrieval evaluation flow

### Completed: Milestone 4B
- Added LLM-as-judge answer relevance evaluator
- Implemented `rageval evaluate-answer-relevance`
- Stored answer relevance scores in `metric_scores`
- Added pass/fail/unknown label policy
- Added mock judge path for manual dev testing
- Verified manual answer relevance evaluation flow

### Current: Milestone 4C
Goal: implement claim extraction from generated answers.

Scope:
- Extract atomic factual claims from each generated answer
- Store extracted claims in DuckDB `claim_evaluations` or a dedicated claim table path
- Use `MockLLMClient` in tests/dev
- Add CLI command to extract claims for a run

Do not implement yet:
- Groundedness / faithfulness judging
- Per-claim support classification
- Completeness
- Safety
- Reports
- compare/ci-check
- GitHub Actions
- Docker