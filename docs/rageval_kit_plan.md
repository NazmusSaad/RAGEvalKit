# RAGEvalKit — Architecture & Build Plan

> A CLI-first, open-source RAG evaluation and regression-testing framework.
> Built to be feasible in 2–4 weeks while looking production-grade on a resume.

---

## 1. Product Vision

RAGEvalKit is a developer tool that does for RAG pipelines what `pytest` + `coverage` did for Python code: it gives you fast, reproducible, CI-gated feedback on whether your retrieval-augmented system is actually getting better or quietly regressing. You point it at a document corpus, it auto-generates an eval set, runs your RAG config end-to-end, scores retrieval and generation with property-based evaluators (relevance, groundedness, completeness, safety), and then **diagnoses *why* failing answers failed** — retrieval miss, hallucination, incomplete answer, or unsafe content. Two runs can be diffed, and a single `rageval ci-check` step fails a pull request when faithfulness drops, recall@k slips, or latency/cost regress past thresholds.

The hiring signal is the part most student RAG projects miss: anyone can ship a chatbot, but very few undergrads ship a **measurement system around a chatbot**. This project demonstrates evaluation design, LLM-as-judge engineering, CI/CD integration, data modeling, and CLI/UX craft — the exact bundle ML platform, LLM infra, and applied ML teams hire for.

---

## 2. Recommended MVP Scope

### Must-have (Weeks 1–3)

- `rageval` CLI with: `init`, `ingest`, `generate-evalset`, `run`, `compare`, `report`, `ci-check`
- Local-only stack: DuckDB for runs/metrics, ChromaDB for vectors, no server
- One reference RAG pipeline (chunk → embed → retrieve → generate) configurable via YAML
- Synthetic eval-set generation from documents via LLM
- **Three core evaluators**: retrieval relevance, groundedness (claim extraction + per-claim NLI-style judging), answer relevance
- Per-query trace storage (retrieved chunks, generated answer, claims, scores, latency, tokens, cost)
- Static HTML report via Jinja2 + Chart.js (zero-server)
- `compare` produces a delta table; `ci-check` exits non-zero on threshold breach
- GitHub Actions workflow + Dockerfile + docker-compose
- README, architecture diagram (Mermaid), 60–90s demo video

### Stretch (Week 4)

- Completeness evaluator
- Safety/PII/toxicity evaluator (use `presidio` + a small judge prompt; do **not** train anything)
- Expected-output similarity (when reference answer exists) using embedding cosine + LLM equivalence judge
- Cohen's kappa calibration against 30–50 hand-labeled examples to *report* judge reliability — this is a huge resume bullet
- Optional Streamlit "inspector" for browsing failed traces

### Explicitly do NOT build

- Auth, multi-tenant features, user management
- A hosted web app, deployed dashboard, or SaaS landing page
- A custom embedding model or any model training/fine-tuning
- Postgres/pgvector — overkill for MVP
- A second RAG framework adapter — pick one and go
- Realtime/streaming monitoring
- A "universal" plugin system — hardcode two retrievers and two generators and call it a day
- Pretty animations or marketing site

The single biggest risk is over-scoping the dashboard. Aim for ugly-but-correct, then polish.

---

## 3. Final Recommended Tech Stack

| Concern | Choice | Why | Alternatives Considered |
|---|---|---|---|
| Language | **Python 3.11+** | Ecosystem fit, type hints mature | — |
| CLI | **Typer** + **Rich** | Type-hint native, beautiful help/output, less boilerplate than Click | Click (more verbose), argparse (ugly) |
| Config | **Pydantic v2** + YAML | Validated configs, great errors, dataclass-like ergonomics | dataclasses (no validation), attrs |
| Run storage | **DuckDB** | Single-file, columnar, blazing for `GROUP BY run_id` analytics, no server, can read Parquet directly | SQLite (slower aggregates, less "platform-y"), Postgres (server overhead) |
| Vector store | **ChromaDB** (persistent client) | Local persistent file, zero ops, embeddings + metadata in one place | FAISS (no metadata), Qdrant (server), pgvector (server) |
| RAG framework | **LlamaIndex** core only | Better retrieval/eval primitives than LangChain in 2026; minimal deps | LangChain (too sprawling), custom (more code) |
| Embeddings | **`BAAI/bge-small-en-v1.5`** via sentence-transformers (default), OpenAI `text-embedding-3-small` as toggle | Free local default for dev; API option for fairness when comparing | E5, Cohere (paid) |
| Generator LLM | **OpenAI `gpt-4o-mini`** default, Anthropic adapter | Cheap, deterministic at temp=0, easy to demo | Local Llama (slower demos, harder to reproduce) |
| Judge LLM | **`gpt-4o-mini`** (same), optional `gpt-4o` for "trusted" mode | Same provider keeps demo simple; document this as a known bias | Cross-provider judging (stretch) |
| Reporting | **Jinja2 + Chart.js** → static HTML | Zero infra, ships as CI artifact, opens anywhere, looks pro | Streamlit (server, slower iteration), React (too much work) |
| Optional dashboard | Streamlit (stretch only) | Cheapest path to interactive inspector | — |
| Tests | **pytest** + **VCR.py** (cassette-based LLM mocking) | Real first run, replays after — no repeated spend | Manual mocks (brittle) |
| Packaging | **`pyproject.toml`** + `uv` (or `pip`) | Modern, fast | poetry (heavier) |
| Container | **Dockerfile** + `docker-compose.yml` | Reproducibility, plays well with CI | — |
| CI | **GitHub Actions** | Where the recruiters look | — |
| Diagrams | **Mermaid** in README | Renders in GitHub natively | excalidraw export (extra step) |

**Key tradeoff to call out:** DuckDB instead of SQLite. DuckDB is materially better for cross-run analytical queries (`SELECT AVG(faithfulness) GROUP BY run_id, config_hash`) and signals platform-engineering taste. It's a single-file embedded DB so there's no operational cost.

---

## 4. High-Level Architecture

**CLI-first.** No backend server in MVP. Everything is a Python process invoked by `rageval`. The HTML report is a static artifact. The optional Streamlit dashboard (stretch) reads the same DuckDB file and is the only thing that could be called a "server."

```
                        ┌───────────────────────────────────┐
                        │           rageval CLI             │
                        │   (Typer entrypoint + Rich UI)    │
                        └────────────────┬──────────────────┘
                                         │
            ┌────────────────────────────┼─────────────────────────────┐
            ▼                            ▼                             ▼
   ┌─────────────────┐         ┌──────────────────┐          ┌───────────────────┐
   │  Ingest module  │         │   Eval Runner    │          │  Report builder   │
   │  - loaders      │         │  - executes RAG  │          │  - Jinja2 + HTML  │
   │  - chunkers     │         │  - logs traces   │          │  - comparison     │
   │  - embedder     │         │  - calls judges  │          │  - root-cause     │
   └────────┬────────┘         └────────┬─────────┘          └─────────┬─────────┘
            │                           │                              │
            ▼                           ▼                              ▼
   ┌────────────────┐         ┌─────────────────┐            ┌──────────────────┐
   │  ChromaDB      │◄────────┤  RAG pipeline   │            │  DuckDB          │
   │  (vectors +    │         │  (LlamaIndex    │            │  (runs, items,   │
   │   chunk meta)  │         │   thin wrapper) │            │   metrics, costs)│
   └────────────────┘         └────────┬────────┘            └──────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  LLM providers  │
                              │ (OpenAI/Anthr.) │
                              └─────────────────┘
                                       ▲
                                       │
                              ┌────────┴────────┐
                              │  Evaluators     │
                              │  - retrieval    │
                              │  - claim extr.  │
                              │  - groundedness │
                              │  - relevance    │
                              │  - safety       │
                              └─────────────────┘
```

**Component responsibilities:**

- **CLI layer (`rageval/cli/`)**: argument parsing only. No business logic.
- **Core (`rageval/core/`)**: config models, run orchestration, RAG pipeline abstraction, judge clients.
- **Storage (`rageval/storage/`)**: DuckDB DAO + ChromaDB DAO. Single source of truth for "what did run X produce."
- **Evaluators (`rageval/evaluators/`)**: each is a pure function `(QueryTrace) -> EvalResult`. Composable. Independently testable with cassettes.
- **Reporting (`rageval/report/`)**: Jinja2 templates + a comparison engine.
- **CI (`rageval/ci/`)**: threshold logic + exit-code policy.

**No backend server in MVP.** Everything is local. The only network calls are to the LLM provider. This is a deliberate design choice — it makes the project trivially `docker run`-able by a recruiter.

---

## 5. CLI Design

All commands write to a project-local `.rageval/` directory (similar to `.git/`). DuckDB lives at `.rageval/runs.db`. Chroma at `.rageval/chroma/`. Run artifacts under `.rageval/runs/<run_id>/`.

### `rageval init`
- **Purpose:** scaffold a project: create `.rageval/`, `configs/baseline.yaml`, `rageval.yaml` (thresholds), `.gitignore` entries.
- **Inputs:** none (optional `--force`).
- **Outputs:** directory + starter files printed to stdout.
- **Example:** `rageval init`

### `rageval ingest <path>`
- **Purpose:** load documents, chunk, embed, persist to Chroma. Idempotent on file hash.
- **Inputs:** path to file or directory; flags: `--config`, `--collection`, `--chunk-size`, `--chunk-overlap`.
- **Outputs:** count of docs/chunks ingested, collection name, embedding model used.
- **Example:** `rageval ingest ./docs --config configs/baseline.yaml`

### `rageval generate-evalset <docs_path>`
- **Purpose:** synthesize an eval set (question + reference answer + source chunk_id) from the corpus using an LLM.
- **Inputs:** docs path; `--num-questions` (default 50), `--output` (default `evalsets/auto.jsonl`), `--diversity` (single-hop|multi-hop|mixed), `--seed`.
- **Outputs:** JSONL file with one question per line, each with `question`, `reference_answer`, `source_chunk_ids`, `difficulty`, `type`.
- **Example:** `rageval generate-evalset ./docs --num-questions 50 --output evalsets/v1.jsonl`

### `rageval run --config <yaml>`
- **Purpose:** run the full RAG pipeline against an eval set and execute all configured evaluators. Stores everything in DuckDB.
- **Inputs:** `--config` (required), `--evalset` (default from config), `--name` (run label), `--tag baseline|candidate|...`.
- **Outputs:** `run_id` (ULID), summary table to stdout (Rich), full traces in DuckDB.
- **Example:** `rageval run --config configs/experiment.yaml --tag candidate`

### `rageval compare <baseline_id> <candidate_id>`
- **Purpose:** print a side-by-side delta of metrics + flag regressions per `rageval.yaml` thresholds.
- **Inputs:** two run IDs (or names like `latest:baseline`); optional `--thresholds`.
- **Outputs:** Rich-rendered delta table; exit 0 always (informational; use `ci-check` for gating).
- **Example:** `rageval compare runs/baseline runs/experiment`

### `rageval report --run <run_id>`
- **Purpose:** generate a self-contained HTML report.
- **Inputs:** `--run` (or `--compare <a> <b>`), `--output` (default `.rageval/reports/<run_id>.html`), `--open`.
- **Outputs:** single HTML file with embedded CSS/JS, inline charts.
- **Example:** `rageval report --compare baseline candidate --output report.html --open`

### `rageval ci-check --baseline <id> --candidate <id> --thresholds <yaml>`
- **Purpose:** the CI gate. Exits 1 if any threshold breached.
- **Inputs:** baseline + candidate run IDs (or refs), thresholds file.
- **Outputs:** machine-readable JSON to stdout + human summary to stderr; non-zero exit on failure.
- **Example:** `rageval ci-check --baseline last_main --candidate last_pr --thresholds rageval.yaml`

### `rageval inspect <run_id>` (nice-to-have, low cost)
- **Purpose:** print N worst-failing queries with root cause and unsupported claims, for terminal debugging.
- **Example:** `rageval inspect latest --top 5 --by faithfulness`

---

## 6. Configuration Design

Two YAML files: **pipeline config** (versionable, per-experiment) and **thresholds** (CI policy, usually one per repo).

### Pipeline config schema (`configs/baseline.yaml`)

```yaml
version: 1
name: baseline
seed: 42

corpus:
  path: ./docs
  glob: "**/*.{md,pdf,txt}"

chunking:
  strategy: recursive          # recursive | sentence | semantic
  chunk_size: 512
  chunk_overlap: 64

embedding:
  provider: sentence_transformers   # sentence_transformers | openai
  model: BAAI/bge-small-en-v1.5
  batch_size: 64

vector_store:
  type: chroma
  path: .rageval/chroma
  collection: docs_v1
  distance: cosine

retrieval:
  top_k: 5
  rerank: null                 # null | bge-reranker-base (stretch)
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
  # stretch: completeness, safety, expected_output_similarity

evalset:
  path: evalsets/v1.jsonl

cost:
  # cents per 1k tokens for bookkeeping (override via env)
  input_per_1k:  0.015
  output_per_1k: 0.060
```

### Thresholds config (`rageval.yaml`)

```yaml
version: 1
# Absolute floors (candidate must clear regardless of baseline)
absolute:
  faithfulness_min: 0.80
  retrieval_relevance_min: 0.70

# Relative regressions (candidate vs baseline)
relative:
  faithfulness_drop_max: 0.05         # fail if drops >5 pts
  retrieval_relevance_drop_max: 0.05
  recall_at_k_drop_max: 0.05
  answer_relevance_drop_max: 0.05
  p50_latency_increase_max: 0.30      # fail if >30% slower
  cost_per_query_increase_max: 0.30

# Pass/fail policy
policy:
  require_all_absolute: true
  require_all_relative: true
  allow_unknown_as_pass: false        # judge-uncertain items count as fail
```

Both files validated by Pydantic on load. Unknown fields fail loudly. `seed` is propagated everywhere reproducibility matters.

---

## 7. Data / Storage Design

**Use DuckDB for the MVP.** Single file (`.rageval/runs.db`), no server, fast aggregates, queryable from the CLI or a notebook. SQLite would work but DuckDB is materially better for analytical queries across runs and shows engineering taste. Postgres is overkill until you have multiple users.

ChromaDB holds vectors + chunk text + chunk metadata (it's already a DB). DuckDB holds everything else.

### Schema

```sql
-- Documents in the corpus
CREATE TABLE documents (
  doc_id          TEXT PRIMARY KEY,        -- sha256(content)
  source_path     TEXT NOT NULL,
  title           TEXT,
  num_chars       INTEGER,
  ingested_at     TIMESTAMP DEFAULT now()
);

-- Chunks produced by a chunking strategy
CREATE TABLE chunks (
  chunk_id        TEXT PRIMARY KEY,        -- sha256(doc_id||span)
  doc_id          TEXT REFERENCES documents(doc_id),
  ordinal         INTEGER,
  text            TEXT NOT NULL,
  num_tokens      INTEGER,
  chunking_config_hash TEXT                -- ties chunk to a chunking strategy
);

-- Eval sets (collections of questions)
CREATE TABLE eval_sets (
  evalset_id      TEXT PRIMARY KEY,
  name            TEXT,
  created_at      TIMESTAMP DEFAULT now(),
  generated_by    TEXT,                    -- 'synthetic' | 'manual' | 'imported'
  config_json     JSON
);

-- Individual eval questions
CREATE TABLE eval_questions (
  question_id     TEXT PRIMARY KEY,
  evalset_id      TEXT REFERENCES eval_sets(evalset_id),
  question        TEXT NOT NULL,
  reference_answer TEXT,                   -- nullable
  source_chunk_ids JSON,                   -- list[chunk_id], for recall@k
  difficulty      TEXT,                    -- 'easy'|'medium'|'hard'
  qtype           TEXT                     -- 'factoid'|'multi_hop'|'reasoning'
);

-- One row per `rageval run`
CREATE TABLE runs (
  run_id          TEXT PRIMARY KEY,        -- ULID
  name            TEXT,
  tag             TEXT,                    -- 'baseline'|'candidate'|...
  config_hash     TEXT,                    -- sha256(full resolved config)
  config_json     JSON,                    -- full resolved config
  evalset_id      TEXT REFERENCES eval_sets(evalset_id),
  git_sha         TEXT,
  started_at      TIMESTAMP,
  finished_at     TIMESTAMP,
  status          TEXT                     -- 'running'|'completed'|'failed'
);

-- One row per (run, question)
CREATE TABLE run_items (
  item_id         TEXT PRIMARY KEY,
  run_id          TEXT REFERENCES runs(run_id),
  question_id     TEXT REFERENCES eval_questions(question_id),
  generated_answer TEXT,
  prompt_tokens   INTEGER,
  completion_tokens INTEGER,
  total_cost_usd  DOUBLE,
  latency_ms      INTEGER,
  model           TEXT,
  error           TEXT                     -- nullable
);

-- Retrieved chunks per run_item, with rank and score
CREATE TABLE retrieved_contexts (
  item_id         TEXT REFERENCES run_items(item_id),
  rank            INTEGER,
  chunk_id        TEXT,                    -- may not FK if corpus changed
  chunk_text      TEXT,                    -- snapshot for trace fidelity
  score           DOUBLE,
  PRIMARY KEY (item_id, rank)
);

-- One row per (run_item, evaluator)
CREATE TABLE metric_scores (
  item_id         TEXT REFERENCES run_items(item_id),
  metric          TEXT,                    -- 'faithfulness','retrieval_relevance',...
  score           DOUBLE,                  -- 0.0–1.0
  label           TEXT,                    -- 'pass'|'fail'|'unknown'
  reason          TEXT,                    -- judge's rationale
  judge_model     TEXT,
  raw_json        JSON,
  PRIMARY KEY (item_id, metric)
);

-- Atomic claims extracted from answer, per-claim verdicts
CREATE TABLE claim_evaluations (
  item_id         TEXT REFERENCES run_items(item_id),
  claim_idx       INTEGER,
  claim_text      TEXT,
  verdict         TEXT,                    -- 'supported'|'contradicted'|'not_enough_info'
  supporting_chunk_ids JSON,
  rationale       TEXT,
  PRIMARY KEY (item_id, claim_idx)
);

-- Optional: cost/latency time-series for charts (denormalized from run_items)
-- Skip unless you want monitoring-style charts.

-- Root cause assignments
CREATE TABLE root_causes (
  item_id         TEXT REFERENCES run_items(item_id),
  primary_cause   TEXT,                    -- 'retrieval'|'grounding'|'incomplete'|'irrelevant'|'unsafe'|'judge_uncertain'|'latency_cost'
  secondary_causes JSON,
  suggested_fix   TEXT,
  PRIMARY KEY (item_id)
);
```

### Run comparison

Comparison is a single SQL `JOIN` on `(metric, question_id)` between two runs. Sample query the `compare` command builds:

```sql
SELECT
  m.metric,
  AVG(CASE WHEN ri.run_id = ? THEN m.score END) AS baseline_score,
  AVG(CASE WHEN ri.run_id = ? THEN m.score END) AS candidate_score
FROM metric_scores m
JOIN run_items ri USING (item_id)
WHERE ri.run_id IN (?, ?)
GROUP BY m.metric;
```

Per-question diffs (for the "biggest regressions" view in the report) are the same join without aggregation. DuckDB makes this trivially fast even on tens of thousands of items.

---

## 8. RAG Pipeline Design

Keep this layer **boring and replaceable**. The interesting code is the evaluation engine.

- **Document loading**: LlamaIndex `SimpleDirectoryReader` + a tiny `pdf` fallback (`pypdf`). Md, txt, pdf only for MVP.
- **Chunking**: `RecursiveCharacterTextSplitter`-style (also from LlamaIndex). Hash the chunking config into `chunking_config_hash`.
- **Embedding**: pluggable provider; for the default config, `BAAI/bge-small-en-v1.5` via sentence-transformers. Batched.
- **Vector search**: Chroma persistent client, cosine, `top_k` from config. Return `(chunk_id, score, text, metadata)`.
- **Generation**: a single Jinja2 prompt template — system + context-injection + question. Provider abstracted behind a `LLMClient` protocol with `complete(messages) -> CompletionResult`.
- **Citation/context tracking**: every `run_item` stores the exact retrieved chunks (ranked, scored, text-snapshot) into `retrieved_contexts`. Don't rely on the live vector store for traces — the corpus might change. Snapshot at retrieval time.

### Versioning RAG configs

A run's identity is `config_hash = sha256(canonicalize(resolved_config))`. Two runs with the same config_hash on the same evalset_id should yield deterministic results at temperature=0. Comparison is then "diff by config field" + "diff by metric." Show the config diff at the top of every comparison report.

---

## 9. Evaluation Engine Design

Every evaluator implements:

```python
class Evaluator(Protocol):
    name: str
    def evaluate(self, item: RunItem) -> list[MetricScore]: ...
```

Evaluators are independently testable, parallelizable, and cassette-mocked in tests. The runner gathers items, fans out evaluators, writes results.

### 9.1 Retrieval Relevance

- **Measures:** Are the retrieved chunks relevant to the question? Reference-free.
- **Input:** `question`, `retrieved_contexts` (top-k).
- **Output:** per-chunk `relevant: bool` + a normalized `retrieval_relevance` score (= fraction of retrieved chunks judged relevant).
- **MVP:** LLM judge, batched, one call scoring up to 5 chunks at a time. Returns JSON.
- **Later:** open-source cross-encoder reranker score (e.g., `bge-reranker-base`) as a cheaper backstop and disagreement signal.

### 9.2 Recall@k / MRR (reference-required)

- **Measures:** Did retrieval surface the chunk(s) that the synthetic question was generated from?
- **Input:** `source_chunk_ids` (from eval question), `retrieved_contexts`.
- **Output:** `recall_at_k` ∈ {0, 1}, `mrr` ∈ [0, 1].
- **MVP:** pure set logic + rank lookup. Cheap, deterministic, no LLM.
- **Note:** only meaningful when questions are synthetic-from-chunks or otherwise labeled.

### 9.3 Claim Extractor

- **Measures:** Decompose the generated answer into atomic verifiable claims.
- **Input:** `generated_answer`.
- **Output:** list of `claim_text` strings.
- **MVP:** LLM call, JSON output. Prompt forces atomicity ("one fact per claim, no compound sentences").
- **Later:** small fine-tuned T5 (out of scope), or rule-based sentence splitting fallback.

### 9.4 Groundedness / Faithfulness

- **Measures:** Is each claim supported by retrieved context? Aggregate per item.
- **Input:** `claims`, `retrieved_contexts`.
- **Output:** per-claim `verdict ∈ {supported, contradicted, not_enough_info}` and a `faithfulness` score = supported / total.
- **MVP:** for each claim, LLM judge prompted with claim + all retrieved chunks; structured JSON verdict + supporting chunk indices + rationale. Cache by `(claim_hash, context_hash)`.
- **Later:** NLI model (`cross-encoder/nli-deberta-v3-base`) as a second opinion; flag disagreements as `judge_uncertain`.

### 9.5 Answer Relevance

- **Measures:** Does the answer address the question (ignoring whether it's true)? Reference-free.
- **Input:** `question`, `generated_answer`.
- **Output:** `answer_relevance ∈ [0, 1]` + label + rationale.
- **MVP:** LLM judge with a 0–4 rubric, normalized.
- **Later:** Ragas-style "reverse generation" — have an LLM generate N questions from the answer and measure their similarity to the original question.

### 9.6 Completeness (stretch)

- **Measures:** If the question has multiple parts, does the answer cover all of them?
- **Input:** `question`, `generated_answer`, optional `reference_answer`.
- **Output:** `completeness ∈ [0, 1]`, list of `missing_aspects`.
- **MVP:** LLM judge that first decomposes the question into required sub-aspects, then checks coverage.

### 9.7 Safety / PII (stretch)

- **Measures:** Toxicity, PII leakage, jailbreak compliance.
- **Input:** `question`, `generated_answer`.
- **Output:** booleans: `contains_pii`, `is_toxic`, `complied_with_jailbreak` + `safety_pass`.
- **MVP:** Microsoft Presidio for PII (no LLM cost) + a lightweight LLM judge for the rest.

### 9.8 Expected Output Similarity (stretch)

- **Measures:** When a reference answer exists, how close is the generated answer?
- **Input:** `generated_answer`, `reference_answer`.
- **Output:** `similarity ∈ [0, 1]` (embedding cosine) + `equivalence ∈ {equivalent, partial, different}` (LLM judge).
- **MVP:** cosine first; LLM judge only when cosine is in the ambiguous middle band.

### 9.9 Overall Aggregator

- **Measures:** Per-item pass/fail/unknown decision and per-run summary.
- **MVP rule:** item passes iff all enabled metrics meet absolute thresholds and none returned `unknown`. Aggregate to run-level percentages.
- **Output:** `metric_scores` row for `overall` + `root_causes` row.

---

## 10. LLM-as-Judge Prompt Templates

All prompts request **structured JSON only**. Wrap each in a Pydantic model and validate; retry once on parse failure with a "your previous output failed to parse; emit valid JSON" follow-up. Always set `temperature=0`, request no preamble.

### 10.1 Synthetic eval question generation

```
SYSTEM: You generate evaluation questions for a RAG system. Output JSON only.

USER:
Given the following passage, generate {N} questions that test whether a RAG
system can correctly answer them using ONLY this passage.

Rules:
- Each question must be answerable from the passage alone.
- Vary difficulty: include factoid, multi-hop, and reasoning questions.
- Provide a concise reference answer drawn verbatim or paraphrased from the
  passage.
- Do not invent facts.

PASSAGE (chunk_id={chunk_id}):
"""
{chunk_text}
"""

Return JSON matching this schema exactly:
{
  "questions": [
    {
      "question": "string",
      "reference_answer": "string",
      "qtype": "factoid" | "multi_hop" | "reasoning",
      "difficulty": "easy" | "medium" | "hard"
    }
  ]
}
```

### 10.2 Retrieval relevance

```
SYSTEM: You are a strict relevance judge. Output JSON only.

USER:
Question: {question}

For each retrieved passage below, decide whether it contains information that
would help answer the question. Be strict: tangentially related ≠ relevant.

Passages:
[0] {chunk_0}
[1] {chunk_1}
...
[k] {chunk_k}

Return JSON:
{
  "judgments": [
    {"index": int, "relevant": true|false, "reason": "string (<=20 words)"}
  ]
}
```

### 10.3 Atomic claim extraction

```
SYSTEM: You decompose answers into atomic factual claims. Output JSON only.

USER:
Decompose the following answer into a list of atomic claims. Rules:
- One distinct fact per claim.
- Rewrite pronouns and references to be self-contained.
- Exclude meta-commentary, hedges, or non-factual sentences.

ANSWER:
"""
{generated_answer}
"""

Return JSON:
{ "claims": ["string", ...] }
```

### 10.4 Groundedness (per claim)

```
SYSTEM: You verify whether claims are supported by source passages. Output JSON only.

USER:
Claim: {claim}

Sources:
[0] {chunk_0}
[1] {chunk_1}
...
[k] {chunk_k}

Decide:
- "supported"          if at least one source directly supports the claim
- "contradicted"       if at least one source directly contradicts the claim
- "not_enough_info"    otherwise (including partial overlap)

Return JSON:
{
  "verdict": "supported" | "contradicted" | "not_enough_info",
  "supporting_indices": [int, ...],
  "rationale": "string (<=30 words)"
}
```

### 10.5 Answer relevance

```
SYSTEM: You score answer relevance. Output JSON only.

USER:
Question: {question}
Answer:   {generated_answer}

Score 0–4:
4 = Directly and fully addresses the question.
3 = Addresses the question but with minor irrelevant content.
2 = Partially addresses; significant off-topic content.
1 = Mostly off-topic but mentions the question subject.
0 = Off-topic / non-answer / refusal.

Return JSON:
{ "score": 0|1|2|3|4, "reason": "string (<=25 words)" }
```

### 10.6 Completeness

```
SYSTEM: You judge whether answers cover all parts of a question. Output JSON only.

USER:
Question: {question}
Answer:   {generated_answer}

Step 1: List the distinct aspects the question is asking about.
Step 2: For each aspect, mark whether the answer addresses it.
Step 3: Compute completeness = covered / total.

Return JSON:
{
  "aspects": [{"aspect": "string", "covered": true|false}],
  "completeness": float
}
```

### 10.7 Root-cause diagnosis

Used only when the item failed; the prompt gets all metric scores as context.

```
SYSTEM: You diagnose RAG failures. Output JSON only.

USER:
Question: {question}
Generated answer: {generated_answer}
Retrieved chunks (top-k): {chunks}
Reference answer (may be empty): {reference_answer}

Metric scores:
- retrieval_relevance: {retrieval_relevance}
- faithfulness:       {faithfulness}
- answer_relevance:   {answer_relevance}
- completeness:       {completeness}

Unsupported claims: {unsupported_claims}

Choose the SINGLE primary cause from:
["retrieval", "grounding", "incomplete", "irrelevant", "unsafe", "judge_uncertain"]

Also list up to two secondary causes and a one-sentence suggested fix.

Return JSON:
{
  "primary_cause": "string",
  "secondary_causes": ["string", ...],
  "suggested_fix": "string"
}
```

### 10.8 Final overall judgment (optional safety net)

```
SYSTEM: You produce a final pass/fail judgment for a RAG response. Output JSON only.

USER:
Given these metric scores: {scores_json}
And these thresholds:      {thresholds_json}

Return JSON:
{ "overall": "pass" | "fail" | "unknown", "reason": "string" }
```

Programmatic aggregation is preferred for the MVP — this prompt exists for the cases where you want an LLM tiebreaker. Don't ship it as the default.

---

## 11. Root-Cause Analysis Design

Two-stage approach: **rules first, LLM second** (only when rules are ambiguous). Rules are cheaper and reproducible; the LLM diagnosis is for human-readable explanation and suggested fixes.

### Failure categories

| Code | Meaning |
|---|---|
| `retrieval` | Right answer wasn't retrievable from the top-k chunks |
| `grounding` | Retrieval was fine but the answer hallucinated or added unsupported claims |
| `incomplete` | Answer left out required aspects |
| `irrelevant` | Answer doesn't address the question (off-topic) |
| `unsafe` | Toxic/PII/jailbreak issue |
| `judge_uncertain` | Judges disagreed or returned low-confidence verdicts |
| `latency_cost` | Met all quality bars but blew latency or cost budget (only flagged for system-level regressions, not per-item) |

### Rule-based assignment (deterministic)

Run these in order; first match wins for `primary_cause`. Secondary causes are appended for any others that also trip.

```
if safety_pass == False:
    primary = "unsafe"
elif retrieval_relevance < 0.5 OR (has_labels AND recall_at_k == 0):
    primary = "retrieval"
elif faithfulness < 0.7:
    primary = "grounding"
elif answer_relevance < 0.5:
    primary = "irrelevant"
elif completeness < 0.7:
    primary = "incomplete"
elif any(metric.label == "unknown" for metric in scores):
    primary = "judge_uncertain"
else:
    primary = None   # item passed
```

Thresholds are configurable in `rageval.yaml` under a `root_cause:` block; defaults above. After rule assignment, run the LLM diagnosis prompt (10.7) **only for failing items** to get a human-readable suggested fix. Cap this at the top N failures per run to control cost (e.g., `--max-diagnoses 50`).

### Multiple failures

Real failures often co-occur (bad retrieval → ungrounded answer → low completeness). The rule above picks the most "upstream" cause first (retrieval > grounding > completeness > relevance) because fixing upstream usually fixes downstream. Secondary causes get listed in the report so users see the full picture.

### Debugging info in the report

For each failing item the report shows:
- Question + generated answer (with **unsupported claims highlighted in red**)
- Top-k retrieved chunks with relevance verdicts (✓/✗) and scores
- Per-claim groundedness table
- Metric breakdown (bar chart with thresholds overlaid)
- Suggested fix from the LLM diagnosis
- Direct link to the next/prev failing item

This is the part recruiters will screenshot. Make it good.

---

## 12. Version Comparison Design

### What a "version" is

A version = `config_hash` (sha256 of the canonicalized resolved config) tied to a specific evalset_id. Two runs are *comparable* iff their `evalset_id` matches; the comparator refuses to compare across evalsets.

### Comparison output

For each metric, compute:
- `baseline_mean`, `candidate_mean`, `absolute_delta`, `relative_delta`
- `n_regressed` (items where candidate < baseline by more than noise threshold)
- `n_improved`
- Per-question worst regressions (sorted by `baseline_score - candidate_score`)

### Regression thresholds (defaults)

```yaml
relative:
  faithfulness_drop_max:        0.05    # >5pt drop = fail
  retrieval_relevance_drop_max: 0.05
  recall_at_k_drop_max:         0.05
  answer_relevance_drop_max:    0.05
  completeness_drop_max:        0.05
  p50_latency_increase_max:     0.30    # >30% slower = fail
  p95_latency_increase_max:     0.50
  cost_per_query_increase_max:  0.30
```

### `ci-check` decision logic

```
fail_reasons = []

# 1. Absolute floors (candidate alone)
for metric, floor in thresholds.absolute.items():
    if candidate[metric] < floor:
        fail_reasons.append(("absolute", metric, candidate[metric], floor))

# 2. Relative regressions (candidate vs baseline)
for metric, max_drop in thresholds.relative.items():
    delta = baseline[metric] - candidate[metric]
    if delta > max_drop:
        fail_reasons.append(("relative", metric, delta, max_drop))

# 3. Latency/cost
for budget in ["p50_latency", "p95_latency", "cost_per_query"]:
    increase = (candidate[budget] - baseline[budget]) / baseline[budget]
    if increase > thresholds.relative[f"{budget}_increase_max"]:
        fail_reasons.append(("budget", budget, increase, ...))

exit_code = 1 if fail_reasons else 0
print(json.dumps({"status": "fail" if fail_reasons else "pass",
                  "reasons": fail_reasons}))
sys.exit(exit_code)
```

The CLI also writes a markdown summary to `$GITHUB_STEP_SUMMARY` when in GitHub Actions (just `print(...)` to that file). This shows up directly on the PR.

---

## 13. Reporting / Dashboard Design

**Use static HTML for the MVP. Skip Streamlit until stretch.** A self-contained HTML file is:

- A CI artifact (uploadable, downloadable, archivable)
- Zero-infra to demo
- Faster to iterate on (no server reload)
- More impressive in a README screenshot (no Streamlit chrome)

### Implementation

- Jinja2 templates rendered to a single HTML file.
- Chart.js loaded from CDN (or vendored inline for offline use).
- Pico.css or a minimal hand-rolled CSS for a clean look — **do not** spend a day on design.
- All data inlined as JSON in a `<script>` tag at the top, then the templates render the chrome and Chart.js handles charts. Total page ~200KB.

### MVP report sections

1. **Header**: run name, tag, config hash, git SHA, timestamp, evalset name
2. **Run summary card**: total items, % passed, % failed, % unknown, total cost, total time
3. **Metric table**: each metric, mean, median, p10, p90 + sparkline
4. **Pass/fail rates** per metric (stacked bar)
5. **Root-cause distribution** (donut chart)
6. **Latency histogram** + p50/p95/p99
7. **Cost breakdown**: tokens in/out per query, total
8. **Worst failing examples** (top 10 by primary metric): question, answer, chunks, unsupported claims highlighted
9. **Per-query trace viewer**: expandable rows with the full trace
10. **Comparison mode** (when called with `--compare a b`): config diff at top, side-by-side metric deltas, biggest regressions section, biggest improvements section

### Fastest implementation path

1. Hardcode the template against a sample DuckDB. Get one ugly HTML rendering.
2. Add Chart.js charts (two: pass/fail and root-cause).
3. Add comparison mode.
4. Spend one afternoon making it not ugly.

That's it. Streamlit is a stretch goal that re-renders the same DuckDB queries in `st.dataframe` calls — easy to add later, but skip until everything else works.

---

## 14. GitHub Action Design

### Workflow

The intended developer experience: a PR changes a chunking strategy, prompt template, or generation model. CI runs the candidate config, compares against the last successful main-branch run, and posts a pass/fail with a downloadable report.

### Sample `.github/workflows/rageval.yml`

```yaml
name: RAG Eval

on:
  pull_request:
    paths:
      - "configs/**"
      - "prompts/**"
      - "src/**"
      - "rageval.yaml"

jobs:
  eval:
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install rageval
        run: pip install -e ".[ci]"

      # Restore the baseline run produced from main
      - name: Restore baseline
        uses: actions/cache@v4
        with:
          path: .rageval/baseline.db
          key: rageval-baseline-${{ github.base_ref }}

      - name: Ingest corpus
        run: rageval ingest ./docs --config configs/candidate.yaml

      - name: Run candidate
        run: rageval run --config configs/candidate.yaml --tag candidate

      - name: Compare against baseline
        run: rageval compare baseline candidate

      - name: Generate report
        run: rageval report --compare baseline candidate --output report.html

      - name: Upload report artifact
        uses: actions/upload-artifact@v4
        with:
          name: rageval-report
          path: report.html

      - name: CI threshold check
        run: rageval ci-check --baseline baseline --candidate candidate --thresholds rageval.yaml
```

The candidate fails the PR via the non-zero exit from `ci-check`. The report is always uploaded (good UX: failed PRs still have a report to inspect). The baseline DB is cached so re-runs of unchanged main don't repay LLM costs.

For demo purposes ship a second simpler workflow that runs on a tiny fixture corpus and uses a mocked LLM via VCR cassettes — that one runs in CI on every commit without spending real money.

---

## 15. Folder Structure

```
rageval-kit/
├── pyproject.toml
├── README.md
├── LICENSE
├── Dockerfile
├── docker-compose.yml
├── rageval.yaml                       # default thresholds
├── .github/
│   └── workflows/
│       ├── rageval.yml                # demo eval workflow
│       └── ci.yml                     # tests/lint
├── configs/
│   ├── baseline.yaml
│   └── experiment.yaml
├── prompts/
│   ├── system.txt
│   ├── rag.j2
│   └── judges/
│       ├── question_gen.j2
│       ├── retrieval_relevance.j2
│       ├── claim_extraction.j2
│       ├── groundedness.j2
│       ├── answer_relevance.j2
│       ├── completeness.j2
│       └── root_cause.j2
├── src/
│   └── rageval/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py                # Typer app
│       │   ├── init_cmd.py
│       │   ├── ingest.py
│       │   ├── generate_evalset.py
│       │   ├── run.py
│       │   ├── compare.py
│       │   ├── report.py
│       │   ├── ci_check.py
│       │   └── inspect.py
│       ├── core/
│       │   ├── config.py              # Pydantic models
│       │   ├── ids.py                 # ULID, hashes
│       │   ├── llm.py                 # LLM client protocol + adapters
│       │   ├── embeddings.py
│       │   ├── chunking.py
│       │   └── pipeline.py            # the RAG pipeline orchestrator
│       ├── storage/
│       │   ├── duckdb_dao.py
│       │   ├── chroma_dao.py
│       │   └── schema.sql
│       ├── evalset/
│       │   ├── synthesize.py
│       │   └── loader.py
│       ├── evaluators/
│       │   ├── base.py
│       │   ├── retrieval_relevance.py
│       │   ├── recall_at_k.py
│       │   ├── claim_extractor.py
│       │   ├── groundedness.py
│       │   ├── answer_relevance.py
│       │   ├── completeness.py
│       │   ├── safety.py
│       │   └── aggregator.py
│       ├── root_cause/
│       │   ├── rules.py
│       │   └── diagnose.py
│       ├── compare/
│       │   └── engine.py
│       ├── ci/
│       │   └── check.py
│       └── report/
│           ├── render.py
│           ├── templates/
│           │   ├── base.html.j2
│           │   ├── run.html.j2
│           │   ├── compare.html.j2
│           │   └── partials/...
│           └── static/
│               ├── pico.min.css
│               └── chart.umd.js
├── examples/
│   ├── tiny-corpus/                   # 3 markdown files for demos & tests
│   └── walkthrough.md
├── tests/
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_retrieval_relevance.py
│   │   ├── test_groundedness.py
│   │   ├── test_root_cause_rules.py
│   │   └── test_ci_check.py
│   ├── integration/
│   │   ├── test_end_to_end_tiny.py
│   │   └── cassettes/                 # VCR cassettes
│   └── fixtures/
│       ├── runs/
│       └── evalsets/
└── docs/
    ├── architecture.md
    ├── prompts.md
    ├── cli.md
    └── images/
        └── architecture.svg
```

---

## 16. Implementation Roadmap

Each milestone has tasks, files, commands, expected outputs, and a smoke test. Treat the smoke test as the milestone's exit criterion — don't move on until it passes.

### Milestone 1 — Skeleton CLI + Config + Storage (Days 1–2)

- **Tasks:** scaffold repo; `pyproject.toml`; Typer app with all commands stubbed (each prints "not implemented"); Pydantic config models; DuckDB schema + DAO with `init_db()`; `rageval init` actually works.
- **Files:** `pyproject.toml`, `src/rageval/cli/main.py`, `src/rageval/cli/init_cmd.py`, `src/rageval/core/config.py`, `src/rageval/storage/duckdb_dao.py`, `src/rageval/storage/schema.sql`.
- **Commands:** `pip install -e .` → `rageval init` → `rageval --help`.
- **Expected output:** `.rageval/runs.db` exists with all tables; help text lists all 8 commands.
- **Smoke test:** `pytest tests/unit/test_storage_init.py` — open the DB, list tables, assert all expected names.

### Milestone 2 — Ingest, Chunk, Embed, Retrieve (Days 3–5)

- **Tasks:** document loaders (md/txt/pdf); chunker; sentence-transformers embedder; Chroma DAO; `ingest` CLI; a thin `retrieve(question, k)` function.
- **Files:** `core/chunking.py`, `core/embeddings.py`, `storage/chroma_dao.py`, `cli/ingest.py`.
- **Commands:** `rageval ingest examples/tiny-corpus`.
- **Expected output:** counts printed; Chroma collection populated; `chunks` and `documents` rows in DuckDB.
- **Smoke test:** integration test: ingest the tiny corpus and assert `retrieve("what is ...", k=3)` returns 3 chunks with non-zero scores.

### Milestone 3 — Generate Evalset + Run RAG and Store Traces (Days 6–8)

- **Tasks:** `LLMClient` adapter (OpenAI first), synthetic question generator, `generate-evalset` CLI, prompt template loader, the full `pipeline.run(question)` that returns a `Trace`, the `rageval run` command that writes runs/run_items/retrieved_contexts.
- **Files:** `core/llm.py`, `evalset/synthesize.py`, `core/pipeline.py`, `cli/generate_evalset.py`, `cli/run.py`, `prompts/judges/question_gen.j2`, `prompts/rag.j2`.
- **Commands:** `rageval generate-evalset examples/tiny-corpus --num-questions 20` → `rageval run --config configs/baseline.yaml`.
- **Expected output:** 20 questions in `eval_questions`; 20 items in `run_items` with retrieved chunks and generated answers; cost/latency populated.
- **Smoke test:** integration test under VCR cassettes producing 5 items end-to-end.

### Milestone 4 — Core Evaluators (Days 9–12)

- **Tasks:** retrieval relevance, claim extractor, groundedness, answer relevance, recall@k. Evaluator runner that fans out and writes `metric_scores` + `claim_evaluations`. Concurrency via `asyncio.gather` with a semaphore.
- **Files:** everything in `evaluators/`, the corresponding judge prompts in `prompts/judges/`.
- **Commands:** `rageval run --config configs/baseline.yaml` now also writes metric_scores.
- **Expected output:** every run_item has rows for `retrieval_relevance`, `faithfulness`, `answer_relevance`; questions with `source_chunk_ids` also have `recall_at_k`.
- **Smoke test:** unit test per evaluator using a cassette; integration test asserting metric_scores cardinality.

### Milestone 5 — Comparison + CI Thresholds (Days 13–14)

- **Tasks:** `compare/engine.py` (SQL-driven diff), `ci/check.py` (threshold logic + exit code + GH summary writer), `cli/compare.py`, `cli/ci_check.py`, root-cause rules engine.
- **Files:** `compare/engine.py`, `ci/check.py`, `root_cause/rules.py`, `cli/compare.py`, `cli/ci_check.py`.
- **Commands:** `rageval run --config configs/experiment.yaml --tag candidate` → `rageval compare baseline candidate` → `rageval ci-check --baseline baseline --candidate candidate --thresholds rageval.yaml`.
- **Expected output:** Rich table with deltas; `ci-check` exits 0 or 1.
- **Smoke test:** unit tests for `ci/check.py` against hand-built metric dicts, asserting expected exit codes.

### Milestone 6 — Report (Days 15–17)

- **Tasks:** Jinja2 templates, Chart.js wiring, root-cause donut, worst-failures section, comparison view, `cli/report.py`.
- **Files:** `report/render.py`, `report/templates/...`.
- **Commands:** `rageval report --compare baseline candidate --output report.html --open`.
- **Expected output:** a single HTML file you'd be proud to screenshot.
- **Smoke test:** render against fixture data and assert the file contains expected section headers.

### Milestone 7 — Polish, Demo, README (Days 18–21+)

- **Tasks:** README with architecture diagram (Mermaid in repo, exported SVG for the README header), demo video (Loom/QuickTime, 60–90s), Dockerfile + docker-compose, GitHub Actions workflow, sample report committed at `examples/sample-report.html`, stretch evaluators if time allows.
- **Files:** `README.md`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/rageval.yml`, `examples/`, `docs/`.
- **Smoke test:** `docker compose run rageval bash -lc "rageval init && rageval ingest examples/tiny-corpus && rageval run --config configs/baseline.yaml && rageval report --run latest"` runs clean.

---

## 17. MVP Vertical Slice

The absolute smallest end-to-end demo, runnable in <2 minutes on a laptop with one API key:

1. `rageval init`
2. Drop 3 markdown files into `docs/` (e.g., three short articles on the same topic so retrieval has real choices to make).
3. `rageval ingest docs/`
4. `rageval generate-evalset docs/ --num-questions 20 --output evalsets/v1.jsonl`
5. `rageval run --config configs/baseline.yaml --tag baseline` (top_k=5, gpt-4o-mini, bge-small)
6. Edit `configs/experiment.yaml` to use `top_k=3` *and* a worse chunk size, so we manufacture a regression.
7. `rageval run --config configs/experiment.yaml --tag candidate`
8. `rageval compare baseline candidate` — terminal shows faithfulness and retrieval_relevance dropped.
9. `rageval report --compare baseline candidate --output report.html --open`
10. `rageval ci-check --baseline baseline --candidate candidate --thresholds rageval.yaml` — exits 1, prints which thresholds were breached.

This is the **demo video script**. The recruiter sees: real RAG runs, real metrics, a real regression caught, a real report. That's the whole pitch.

---

## 18. Testing Strategy

The hardest testing problem here is **don't burn your API budget on every test run**. Solve it with VCR.

### Unit tests

- Pure functions only: chunking, hashing, threshold logic, root-cause rules, comparison math, prompt rendering. These need zero LLM calls.
- Aim: ~70% of test count, ~95% of test runs.

### Integration tests

- One end-to-end test on the tiny corpus (3 docs, 5 questions). Uses **VCR.py** cassettes — first run hits real API and records; subsequent runs replay.
- Cassettes are committed to git under `tests/integration/cassettes/`.
- A separate `tests/integration/test_live.py` is *not* run in CI but available for manual re-recording when prompts change. Mark with `@pytest.mark.live`.

### Golden test fixtures

- Hand-curated `tests/fixtures/runs/golden_run.db` (a DuckDB committed to git). Comparison and report rendering assert against expected outputs from this DB.
- For report tests: render to HTML, assert presence of expected substrings (don't snapshot the whole HTML — too brittle).

### Mock LLM judge responses

- Provide a `MockLLMClient` in test utilities that returns canned JSON. Useful for unit-testing evaluator output parsing without VCR overhead.

### Smoke tests in CI

- Run the full vertical slice on the tiny corpus with VCR cassettes — should take <30 seconds and spend zero dollars.
- A nightly scheduled workflow (manual trigger only) re-records cassettes against real APIs.

### Cost discipline

- Set a `RAGEVAL_MAX_LIVE_CALLS` env var; the LLM client refuses past the limit. Default to something safe like 50 for local dev.
- Cache judge responses keyed by `(prompt_hash, model)` on disk under `.rageval/cache/judges/`. Rerunning the same eval costs nothing.
- During development, hardcode `RAGEVAL_FORCE_CACHE=1` to forbid net calls.

---

## 19. Risks and Scope Control

| Risk | Mitigation |
|---|---|
| **Spending too long on the dashboard** | Hard-cap: 2 days for the HTML report. If you want pretty, do it after the README is shipped. Skip Streamlit until everything else is done. |
| **LLM judge inconsistency** | Run each judge call at `temperature=0`. Cache. For groundedness, add an NLI cross-encoder as a second opinion in stretch and flag disagreements as `judge_uncertain`. Report Cohen's kappa from a small labeled subset as a credibility signal. |
| **Vector DB complexity** | Chroma persistent client only. No multi-tenant, no remote server. If Chroma misbehaves, fallback plan: in-memory FAISS-flat — 30 lines of code. |
| **Synthetic eval quality** | Generate from chunks (so labels are mechanically grounded), enforce diversity (factoid/multi-hop/reasoning), filter questions whose reference_answer is empty or duplicates another's. Manually spot-check the first 20 you ever generate. |
| **API costs** | Aggressive caching, VCR cassettes in CI, `gpt-4o-mini` everywhere by default, semaphore-limited concurrency, `--max-diagnoses` cap on root-cause LLM calls. Set a hard monthly OpenAI budget alert. |
| **Spending too long benchmarking** | Don't benchmark against existing eval libraries during the build. Add a "comparison to alternatives" paragraph in the README at the very end if you have time. |
| **Not finishing a demo** | Build the vertical slice (Section 17) by end of week 2 — even if it's ugly. Polish from there. A done ugly demo beats a half-built clean one. |
| **Scope creep into "agentic RAG", reranking, multi-modal** | Write the README skeleton on day 3. If a feature isn't in the README, don't build it. |
| **Prompt brittleness** | All judge prompts force JSON output, validated by Pydantic. One retry on parse failure with an explicit "your JSON was invalid" follow-up. After that, log and mark `unknown`. |

---

## 20. Recruiter-Facing Deliverables

### README structure

```
# RAGEvalKit
[badges: PyPI, CI, license, demo]

> One-sentence pitch.

## The problem (3–4 sentences)
## What this does (the GIF of the comparison report goes here)
## Quickstart (5 commands, copy-pasteable)
## How it works (architecture diagram + 2 paragraphs)
## Evaluators (table of what's measured and how)
## CI usage (the YAML snippet from §14)
## Configuration (link to docs/cli.md)
## Sample report (link to examples/sample-report.html on GitHub Pages)
## Roadmap (checkboxes — what's done, what's next)
## License
```

The GIF at the top of the README is the most important asset in the entire project. Record the vertical slice happening end-to-end. ~15 seconds.

### Architecture diagram

Use Mermaid for the in-README version (renders natively on GitHub). Export an SVG of the ASCII diagram in §4 for the docs site. Caption: "RAGEvalKit is CLI-first: no servers required for the MVP. The optional dashboard is a static HTML artifact."

### Demo video script (60–90s)

1. (0:00–0:10) Title card: "RAGEvalKit — catch RAG regressions in CI."
2. (0:10–0:25) `rageval ingest docs/` → 3 docs ingested.
3. (0:25–0:40) `rageval generate-evalset` → 20 questions appear.
4. (0:40–0:55) `rageval run --tag baseline`; `rageval run --tag candidate`.
5. (0:55–1:10) `rageval compare` → highlight the red regression line.
6. (1:10–1:25) Open the HTML report — scroll past the root-cause donut and one expanded failing trace.
7. (1:25–1:30) `rageval ci-check` exits non-zero. End card: "github.com/<you>/rageval-kit".

### Screenshots to produce

1. Terminal showing colorized comparison table with red regression
2. HTML report header card (run summary)
3. HTML report root-cause donut + worst-failures section
4. HTML report comparison view with config diff at top
5. GitHub PR with the failed `ci-check` and the attached report artifact

### Metrics table to include in README

| Metric | What it measures | Reference required? |
|---|---|---|
| `retrieval_relevance` | Fraction of retrieved chunks judged relevant | No |
| `recall_at_k` | Did retrieval find the source chunk? | Yes |
| `mrr` | Reciprocal rank of first relevant chunk | Yes |
| `faithfulness` | Fraction of answer claims supported by retrieved context | No |
| `answer_relevance` | Does the answer address the question? | No |
| `completeness` (stretch) | Coverage of question sub-aspects | No |
| `safety_pass` (stretch) | No PII, no toxicity, no jailbreak compliance | No |
| `expected_output_similarity` (stretch) | Match to reference answer | Yes |

### Resume bullets

- Built RAGEvalKit, an open-source CLI for property-based RAG evaluation and CI regression testing (Python, Typer, DuckDB, ChromaDB, LlamaIndex); used by ⟨N⟩ developers / ⟨N⟩ GitHub stars / featured in ⟨X⟩.
- Designed an LLM-as-judge evaluation engine with claim-level groundedness verification, retrieval relevance scoring, and rule-based root-cause analysis across 6 failure categories.
- Engineered a `ci-check` command that fails GitHub Actions when faithfulness drops >5%, retrieval recall regresses, or latency/cost budgets are exceeded; integrates as a 12-line YAML step.
- Implemented deterministic, cassette-mocked test suite that exercises full evaluation pipelines without API spend; calibrated LLM-judge reliability against 50 hand-labeled examples (Cohen's κ = ⟨X⟩).
- Wrote analytical DuckDB schema for cross-run metric comparison and per-query trace storage; renders self-contained HTML reports with embedded Chart.js visualizations.

### LinkedIn post draft

> Shipping a RAG pipeline is the easy part. Knowing whether your latest change made it better or quietly broke it — that's the hard part.
>
> I built **RAGEvalKit**, an open-source CLI that scores retrieval and generation quality on every config change, diagnoses *why* answers fail (retrieval miss vs hallucination vs incompleteness), and fails your PR if metrics regress past configurable thresholds.
>
> One command runs a full eval. A GitHub Action turns it into a quality gate. The report is a single HTML file you can attach to any PR.
>
> Built with Python, Typer, DuckDB, ChromaDB, and an LLM-as-judge evaluation engine with claim-level groundedness verification.
>
> ⭐ ⟨github link⟩ — feedback and contributions very welcome.

---

## 21. Final Recommendation

### The exact build path

**Stack:** Python 3.11 · Typer + Rich · Pydantic v2 · DuckDB · ChromaDB · LlamaIndex (core) · sentence-transformers (`bge-small-en-v1.5`) · OpenAI `gpt-4o-mini` · Jinja2 + Chart.js · pytest + VCR · GitHub Actions · Docker.

**MVP features that ship (in this order):**

1. CLI skeleton with all 8 commands routable
2. `init` + DuckDB schema
3. `ingest` + Chroma
4. `generate-evalset`
5. `run` with full trace logging
6. Three evaluators: retrieval relevance, groundedness (with claim extraction), answer relevance
7. `compare` + `ci-check`
8. Static HTML report (run + comparison views)
9. GitHub Action + Dockerfile
10. README + demo GIF + architecture diagram

**Build first:** the vertical slice from §17 against a tiny corpus, with everything stubbed/ugly but end-to-end working by end of week 2. Polish afterward.

**Defer to week 4 or beyond:** completeness, safety/PII, expected-output similarity, NLI second-opinion judging, Cohen's kappa calibration, Streamlit dashboard.

**Cut entirely:** auth, multi-tenant, hosted deployment, custom model training, plugin systems, second RAG framework adapter.

### What would make this project truly standout

Three multipliers, in priority order:

1. **The judge-calibration story.** Hand-label 50 examples. Compute Cohen's κ between your LLM judge and yourself. Report it in the README and explain how you'd improve it (NLI second opinion, prompt ensembling, etc.). This single section is the difference between "another RAG eval tool" and "this person has actually thought about whether their measurements are real."

2. **The CI demo.** A linked PR in your own repo where a config change triggers `ci-check` to fail, with the report attached as an artifact. Recruiters can click and see your tool *working* on a real PR. Almost no student project has this.

3. **The root-cause UX.** Most eval tools dump metrics; very few tell you *why* something failed in plain English. The combination of (a) rule-based primary cause, (b) LLM-generated suggested fix, and (c) inline highlighting of unsupported claims in the report is the single most screenshot-worthy thing in the project. Invest a day on this beyond the bare minimum.

Build those three things on top of a working vertical slice and this project will outperform 95% of undergraduate ML side projects in an internship inbox. Good luck — now go ship it.
