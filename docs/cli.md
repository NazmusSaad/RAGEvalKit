# CLI Reference

All commands share the same entry point: `rageval <command> [options]`. Run `rageval --help` or `rageval <command> --help` for inline help.

---

## `rageval init`

Initializes the `.rageval/` workspace in the current directory.

Creates:
- `.rageval/runs.db` — DuckDB database with the full schema
- `.rageval/chroma/` — directory for ChromaDB persistent data

Safe to re-run: existing data is not overwritten.

```bash
rageval init
```

---

## `rageval ingest <corpus_path>`

Loads documents from `<corpus_path>`, chunks them, embeds them, and stores them in both DuckDB and ChromaDB.

```bash
rageval ingest examples/demo-corpus --config examples/configs/demo_openai.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` / `-c` | `rageval.yaml` | Path to pipeline config YAML |

The config controls chunking strategy (`recursive`), chunk size, overlap, embedding provider, and Chroma collection name. Ingestion is idempotent: re-ingesting the same documents updates chunks without creating duplicates.

---

## `rageval retrieve`

Interactive retrieval from the vector store. Useful for debugging retrieval quality before running a full evaluation.

```bash
rageval retrieve --query "What is recall@k?" --config examples/configs/demo_openai.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `--query` / `-q` | required | Query string |
| `--config` / `-c` | `rageval.yaml` | Pipeline config |
| `--top-k` / `-k` | from config | Number of results to return |

---

## `rageval generate-evalset`

Generates a synthetic evalset by prompting an LLM to write questions grounded in the ingested corpus chunks.

```bash
rageval generate-evalset examples/demo-corpus \
  --num-questions 5 \
  --output evalsets/demo.jsonl \
  --config examples/configs/demo_openai.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `--num-questions` / `-n` | `10` | Number of questions to generate |
| `--output` / `-o` | `evalsets/evalset.jsonl` | Output JSONL path |
| `--config` / `-c` | `rageval.yaml` | Pipeline config (controls judge LLM) |

Each question in the output JSONL includes:
- `question_id` — stable hash
- `question` — generated question text
- `reference_answer` — reference answer (optional)
- `source_chunk_ids` — the chunk IDs used to ground the question (used as retrieval ground truth by `evaluate-retrieval`)

---

## `rageval run`

Runs the full RAG pipeline on an evalset: retrieves top-k chunks, generates an answer with the LLM, and stores the trace.

```bash
rageval run \
  --config examples/configs/demo_openai.yaml \
  --evalset evalsets/demo.jsonl \
  --tag "v1.2-candidate" \
  --limit 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` / `-c` | `rageval.yaml` | Pipeline config |
| `--evalset` / `-e` | from config | Evalset JSONL path |
| `--tag` | `""` | Human-readable label for the run |
| `--limit` / `-l` | no limit | Only process the first N questions |

Outputs the run ID on completion. All subsequent evaluation commands take `--run <RUN_ID>`.

---

## `rageval evaluate-retrieval`

Computes recall@k and MRR for a run, using `source_chunk_ids` from the evalset as ground truth.

```bash
rageval evaluate-retrieval --run <RUN_ID>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run` / `-r` | required | Run ID |

Items where `source_chunk_ids` is empty are labelled `unknown` (recall@k cannot be computed without ground truth).

---

## `rageval evaluate-answer-relevance`

Scores answer relevance for each run item using an LLM judge. Scores are on a 1–5 scale; scores ≥ 3 are labelled `pass`.

```bash
rageval evaluate-answer-relevance --run <RUN_ID> --config examples/configs/demo_openai.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run` / `-r` | required | Run ID |
| `--config` / `-c` | `rageval.yaml` | Config (controls judge LLM) |

---

## `rageval extract-claims`

Decomposes each generated answer into atomic claims. Claims are stored with `verdict="unjudged"` and later scored by `evaluate-groundedness`.

```bash
rageval extract-claims --run <RUN_ID> --config examples/configs/demo_openai.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run` / `-r` | required | Run ID |
| `--config` / `-c` | `rageval.yaml` | Config (controls judge LLM) |

---

## `rageval evaluate-groundedness`

Judges each extracted claim as `supported`, `contradicted`, or `not_enough_info` by checking it against the retrieved context. Computes a per-item faithfulness score (fraction of claims supported).

```bash
rageval evaluate-groundedness --run <RUN_ID> --config examples/configs/demo_openai.yaml
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run` / `-r` | required | Run ID |
| `--config` / `-c` | `rageval.yaml` | Config (controls judge LLM) |

---

## `rageval summarize-run`

Aggregates all metric scores for a run, assigns overall `pass`/`fail`/`unknown` labels to each item, and classifies primary root causes.

```bash
rageval summarize-run --run <RUN_ID>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run` / `-r` | required | Run ID |

Root-cause labels:
- `retrieval_failure` — recall@k = 0 or below threshold
- `grounding_failure` — faithfulness below threshold
- `answer_relevance_failure` — answer relevance below threshold
- `missing_metric` — a required metric was not computed
- `judge_uncertain` — judge returned ambiguous result
- `none` — all checks passed

---

## `rageval compare`

Compares metric means, label counts, and root-cause distributions between two runs.

```bash
rageval compare --baseline <BASELINE_RUN_ID> --candidate <CANDIDATE_RUN_ID>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--baseline` / `-b` | required | Baseline run ID |
| `--candidate` / `-c` | required | Candidate run ID |

Metric deltas are shown as `+N.NNN` (improvement) or `-N.NNN` (regression). Missing metrics are shown as `N/A`.

---

## `rageval ci-check`

Checks a candidate run against configured thresholds and exits with code 0 (pass) or 1 (fail).

```bash
rageval ci-check \
  --baseline <BASELINE_RUN_ID> \
  --candidate <CANDIDATE_RUN_ID> \
  --thresholds rageval.yaml \
  --json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--baseline` / `-b` | required | Baseline run ID |
| `--candidate` / `-c` | required | Candidate run ID |
| `--thresholds` / `-t` | required | YAML file containing `thresholds:` block |
| `--json` | off | Emit machine-readable JSON result |

**Threshold YAML format:**

```yaml
thresholds:
  absolute:
    recall_at_k_min: 0.70        # candidate score must be >= this
    answer_relevance_min: 0.70
    faithfulness_min: 0.80
  relative:
    recall_at_k_drop_max: 0.05   # candidate may drop at most this much from baseline
    faithfulness_drop_max: 0.05
    answer_relevance_drop_max: 0.05
    mrr_drop_max: 0.05
```

All threshold fields are optional (default `null` = skip that check). Backward-compatible aliases: `retrieval_relevance_min` maps to `recall_at_k_min`; `retrieval_relevance_drop_max` maps to `recall_at_k_drop_max`.

---

## `rageval report`

Generates a self-contained HTML report for a run.

```bash
rageval report --run <RUN_ID> --output reports/run.html --open
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run` / `-r` | required | Run ID |
| `--output` / `-o` | `report.html` | Output path |
| `--open` | off | Open the report in the default browser after writing |

The report includes run metadata, metric summary, label counts, root-cause distribution, and per-item expandable sections with Q&A, retrieved chunks, claims, and groundedness verdicts. Missing data sections show friendly notes directing you to the relevant command.

---

## `rageval inspect`

Inspect raw stored data for a run (items, metrics, contexts).

```bash
rageval inspect --run <RUN_ID>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--run` / `-r` | required | Run ID |
