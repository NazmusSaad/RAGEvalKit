# Demo Scripts

RAGEvalKit includes two demo scripts that run the full end-to-end evaluation pipeline:

| Script | Platform | Mode |
|--------|----------|------|
| `examples/demo_live_openai.sh` | bash / zsh / WSL | Live OpenAI calls |
| `examples/demo_live_openai.ps1` | PowerShell (Windows) | Live OpenAI calls |
| `examples/demo_commands.ps1` | PowerShell | Mock/dev (no API key) |

---

## Live Demo (OpenAI)

Runs the complete 10-step pipeline using real OpenAI calls. Produces an HTML report with real generated answers, relevance scores, faithfulness scores, and root-cause classification.

### Prerequisites

- RAGEvalKit installed: `pip install -e ".[dev]"`
- An OpenAI API key

### Setup

```bash
# bash/zsh
export OPENAI_API_KEY=sk-...

# PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

### Run

```bash
# bash / zsh / WSL
bash examples/demo_live_openai.sh

# PowerShell
.\examples\demo_live_openai.ps1
```

### What happens

The script runs these steps automatically:

1. `rageval init` — creates `.rageval/` workspace
2. `rageval ingest examples/demo-corpus` — chunks and embeds 3 Markdown documents
3. `rageval generate-evalset` — generates 5 questions via `gpt-4o-mini`
4. `rageval run --limit 3` — runs the RAG pipeline on 3 questions
5. `rageval evaluate-retrieval` — computes recall@k and MRR
6. `rageval evaluate-answer-relevance` — scores answer relevance with `gpt-4o-mini`
7. `rageval extract-claims` — extracts atomic claims from each answer
8. `rageval evaluate-groundedness` — judges claims against retrieved context
9. `rageval summarize-run` — assigns labels and classifies root causes
10. `rageval report` — writes `reports/live_demo_report.html`

### Expected output

```
══════════════════════════════════════════
  RAGEvalKit Live Demo  (OpenAI)
══════════════════════════════════════════

▶ [1/10] Initialising .rageval/ workspace...
▶ [2/10] Ingesting demo corpus ...
▶ [3/10] Generating evaluation questions (5 questions via OpenAI)...
▶ [4/10] Running RAG pipeline (limit 3 items)...

  Run ID: run_abc123...

▶ [5/10] Evaluating retrieval metrics (recall@k, MRR)...
▶ [6/10] Evaluating answer relevance (via OpenAI judge)...
▶ [7/10] Extracting atomic claims from generated answers...
▶ [8/10] Evaluating groundedness of claims (via OpenAI judge)...
▶ [9/10] Summarising run and classifying root causes...
▶ [10/10] Generating HTML report...

══════════════════════════════════════════
  Demo complete!
  Report: reports/live_demo_report.html
══════════════════════════════════════════
```

Open the report:
```bash
open reports/live_demo_report.html          # macOS
xdg-open reports/live_demo_report.html      # Linux
Invoke-Item reports/live_demo_report.html   # PowerShell
```

### Cost

Running the default 3-item demo against `gpt-4o-mini` costs a fraction of a cent. Exact cost depends on current OpenAI pricing.

---

## Mock/Dev Demo (no API key)

`examples/demo_commands.ps1` runs the same pipeline using the built-in mock providers. No API key, no model downloads, instant results.

```powershell
.\examples\demo_commands.ps1
```

This is useful for:
- Verifying the installation works end-to-end before setting up an API key
- Running the pipeline in CI without spending API credits
- Understanding the output format before seeing real LLM responses

The mock mode uses:
- `DummyEmbedder`: deterministic SHA-256-based 16-dim vectors
- `MockLLMClient`: returns canned JSON responses for each judge task

---

## Demo Corpus

The demo corpus (`examples/demo-corpus/`) contains three Markdown documents:

| File | Content |
|------|---------|
| `01_rag_evaluation.md` | recall@k, MRR, faithfulness, answer relevance, overall labels, root causes |
| `02_rag_failure_modes.md` | Retrieval failure, grounding failure, answer relevance failure, cascading failures |
| `03_ci_regression_testing.md` | CI regression testing, absolute/relative thresholds, exit codes, recommended workflow |

These documents are self-referential: the corpus describes the framework you're evaluating with. This makes it easy to verify that generated answers are reasonable without domain expertise.

---

## GitHub Actions CI Workflows

RAGEvalKit ships two workflow files under `.github/workflows/`:

### `tests.yml` — Default CI (mock-only, runs on every push/PR)

```yaml
on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]
```

- Uses `ubuntu-latest` and Python 3.11
- Installs the package with `pip install -e ".[dev]"` (pip cache enabled)
- Runs `rageval --help` and the full pytest suite
- **Does not set `OPENAI_API_KEY`**; no real API calls are made
- **Does not download any embedding models**; all tests use `DummyEmbedder` and `MockLLMClient`
- Deterministic and free to run on every commit

### `live-demo.yml` — Optional live validation (manual trigger only)

```yaml
on:
  workflow_dispatch:
    inputs:
      num_questions: ...
      limit: ...
```

- Triggered manually from the GitHub Actions tab
- Requires `OPENAI_API_KEY` configured as a GitHub repository secret
- Runs the full 10-step pipeline on the demo corpus
- Uploads `reports/live_demo_report.html` as a downloadable Actions artifact (retained 30 days)
- **Never runs on push or pull_request**

### Why live LLM evaluation should not run on every push

Running real LLM calls in default CI has several problems:

1. **Cost**: every commit (including docs and formatting changes) burns API quota
2. **Flakiness**: LLM outputs are non-deterministic; the same input can produce different scores on consecutive runs, making threshold checks unreliable as a gate
3. **Secret exposure surface**: more frequent secret use means more exposure opportunity
4. **Slow feedback**: real API calls add latency to PR review cycles

The `MockLLMClient` in the default workflow covers all code paths that matter for correctness. The live workflow exists to validate the real provider path before releases or after provider-side changes.

### Adding `OPENAI_API_KEY` as a GitHub repository secret

1. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `OPENAI_API_KEY`
4. Value: your OpenAI API key (starts with `sk-`)
5. Click **Add secret**

The secret is automatically masked in GitHub Actions log output and is never stored in the repository.
