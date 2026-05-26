# demo_commands.ps1 — Mock/dev demo for RAGEvalKit.
#
# Runs the full evaluation pipeline using built-in mock providers.
# No API key required. No model downloads.
#
# Run from the project root:
#   .\examples\demo_commands.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "══════════════════════════════════════════"
Write-Host "  RAGEvalKit Mock/Dev Demo"
Write-Host "  (no API key required)"
Write-Host "══════════════════════════════════════════"
Write-Host ""

# ── Step 1: init ──────────────────────────────────────────────────────────────
Write-Host "▶ [1/9] Initialising .rageval/ workspace..."
rageval init

# ── Step 2: ingest ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "▶ [2/9] Ingesting tiny corpus (dummy embeddings)..."
rageval ingest examples/tiny-corpus

# ── Step 3: generate evalset ──────────────────────────────────────────────────
Write-Host ""
Write-Host "▶ [3/9] Generating eval questions (mock LLM)..."
rageval generate-evalset examples/tiny-corpus `
  --num-questions 3 `
  --output evalsets/dev_mock.jsonl

# ── Step 4: run RAG pipeline ──────────────────────────────────────────────────
Write-Host ""
Write-Host "▶ [4/9] Running RAG pipeline (mock generation)..."
rageval run `
  --evalset evalsets/dev_mock.jsonl `
  --tag "mock-demo"

# ── Capture latest run ID ─────────────────────────────────────────────────────
$RUN_ID = python -c @"
import duckdb, sys
try:
    con = duckdb.connect('.rageval/runs.db')
    row = con.execute(
        'SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1'
    ).fetchone()
    con.close()
    print(row[0] if row else '')
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
"@

if (-not $RUN_ID) {
    Write-Host "ERROR: Could not determine run ID from .rageval/runs.db"
    exit 1
}

Write-Host ""
Write-Host "  Run ID: $RUN_ID"

# ── Step 5: evaluate retrieval ────────────────────────────────────────────────
Write-Host ""
Write-Host "▶ [5/9] Evaluating retrieval metrics (recall@k, MRR)..."
rageval evaluate-retrieval --run $RUN_ID

# ── Step 6: evaluate answer relevance ─────────────────────────────────────────
Write-Host ""
Write-Host "▶ [6/9] Evaluating answer relevance (mock judge)..."
rageval evaluate-answer-relevance --run $RUN_ID

# ── Step 7: extract claims ────────────────────────────────────────────────────
Write-Host ""
Write-Host "▶ [7/9] Extracting atomic claims (mock judge)..."
rageval extract-claims --run $RUN_ID

# ── Step 8: evaluate groundedness ─────────────────────────────────────────────
Write-Host ""
Write-Host "▶ [8/9] Evaluating groundedness (mock judge)..."
rageval evaluate-groundedness --run $RUN_ID

# ── Step 9: summarise + report ────────────────────────────────────────────────
Write-Host ""
Write-Host "▶ [9/9] Summarising run and generating HTML report..."
rageval summarize-run --run $RUN_ID

$null = New-Item -ItemType Directory -Force -Path "reports"
rageval report --run $RUN_ID --output reports/mock_demo_report.html

Write-Host ""
Write-Host "══════════════════════════════════════════"
Write-Host "  Mock demo complete!"
Write-Host "  Report: reports/mock_demo_report.html"
Write-Host "  Run ID: $RUN_ID"
Write-Host "══════════════════════════════════════════"
Write-Host ""
Write-Host "Open the report:"
Write-Host "  Invoke-Item reports/mock_demo_report.html"
