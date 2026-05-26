# RAGEvalKit Live Demo (OpenAI)

This guide walks through the end-to-end RAGEvalKit pipeline using the bundled demo corpus and real OpenAI calls.

## Prerequisites

- Python 3.11+
- RAGEvalKit installed (`pip install -e ".[dev]"`)
- An OpenAI API key

## Set Your API Key

**bash / zsh:**
```bash
export OPENAI_API_KEY=sk-...
```

**PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

The key is read from the environment at runtime. It is never written to any config file or stored on disk.

## Run the Demo

From the project root:

**bash / zsh (macOS / Linux / WSL):**
```bash
bash examples/demo_live_openai.sh
```

**PowerShell (Windows):**
```powershell
.\examples\demo_live_openai.ps1
```

## What the Demo Does

The script runs the full 10-step evaluation pipeline:

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `rageval init` | Creates the `.rageval/` workspace |
| 2 | `rageval ingest` | Loads the 3-document demo corpus, chunks it, embeds with `BAAI/bge-small-en-v1.5`, stores in DuckDB + Chroma |
| 3 | `rageval generate-evalset` | Uses `gpt-4o-mini` to generate 5 evaluation questions from the corpus |
| 4 | `rageval run` | Runs the RAG pipeline on 3 questions (retrieval + generation via `gpt-4o-mini`) |
| 5 | `rageval evaluate-retrieval` | Computes recall@k and MRR against generated reference chunk IDs |
| 6 | `rageval evaluate-answer-relevance` | Judge LLM scores answer relevance 1–5 |
| 7 | `rageval extract-claims` | Breaks each answer into atomic claims |
| 8 | `rageval evaluate-groundedness` | Judge LLM verifies each claim against retrieved context |
| 9 | `rageval summarize-run` | Classifies primary root cause per item |
| 10 | `rageval report` | Writes a self-contained HTML report to `reports/live_demo_report.html` |

## Demo Corpus

The corpus is under `examples/demo-corpus/` and contains three Markdown documents:

- `01_rag_evaluation.md` — RAG evaluation metrics (recall@k, MRR, faithfulness, answer relevance)
- `02_rag_failure_modes.md` — Retrieval failure, grounding failure, answer relevance failure
- `03_ci_regression_testing.md` — CI regression testing, absolute/relative thresholds, exit codes

## Configuration

The demo uses `examples/configs/demo_openai.yaml`, which configures:

- **Embedding**: `sentence-transformers` with `BAAI/bge-small-en-v1.5` (runs locally, no API key required)
- **Generation / Judge**: `openai` with `gpt-4o-mini`
- **Vector store**: Chroma collection `demo_openai` under `.rageval/chroma/`

The collection name `demo_openai` is intentionally distinct from any development collection (e.g. `docs_dev`, `docs_v1`) to prevent cross-contamination.

## Expected Cost

Running the default 3-item demo against `gpt-4o-mini` costs a fraction of a cent. Actual cost depends on current OpenAI pricing. See [OpenAI pricing](https://platform.openai.com/pricing) for current rates.

## Output

After the demo completes, open the HTML report:

```bash
open reports/live_demo_report.html          # macOS
xdg-open reports/live_demo_report.html      # Linux
Invoke-Item reports/live_demo_report.html   # PowerShell
```

The report is self-contained (no CDN, no JS) and shows:
- Per-run metric means (recall@k, MRR, faithfulness, answer relevance)
- Pass / fail / unknown label counts
- Root-cause distribution
- Per-item expandable sections with Q&A, retrieved context, claims, and groundedness verdicts

## Troubleshooting

**`OPENAI_API_KEY is not set`**: Export the key before running the script (see above).

**`ModuleNotFoundError: sentence_transformers`**: Install the extra: `pip install sentence-transformers`.

**`chromadb` errors on first run**: ChromaDB downloads tokeniser data on first use. This is normal; subsequent runs are faster.

**Empty evalset**: If `generate-evalset` produces no questions, verify the corpus path is correct and the OpenAI key has quota.
