#!/usr/bin/env bash
# demo_live_openai.sh — Full RAGEvalKit live demo using OpenAI.
#
# Run from the project root:
#   export OPENAI_API_KEY=sk-...
#   bash examples/demo_live_openai.sh
#
# Expected cost: small for this 3-item demo; depends on provider pricing.
# Uses gpt-4o-mini by default (configured in examples/configs/demo_openai.yaml).
#
# What this script does:
#   1. Initialise the .rageval/ workspace
#   2. Ingest the demo corpus into DuckDB + Chroma (collection: demo_openai)
#   3. Generate 5 evaluation questions using the judge LLM
#   4. Run the RAG pipeline on 3 questions (--limit 3)
#   5. Evaluate retrieval metrics (recall@k, MRR)
#   6. Evaluate answer relevance with the judge LLM
#   7. Extract atomic claims from generated answers
#   8. Evaluate groundedness of each claim
#   9. Summarise the run and classify root causes
#  10. Generate a self-contained HTML report

set -euo pipefail

# ── Preflight: require OPENAI_API_KEY ────────────────────────────────────────
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo ""
  echo "ERROR: OPENAI_API_KEY is not set."
  echo ""
  echo "  export OPENAI_API_KEY=sk-..."
  echo ""
  echo "Then re-run:  bash examples/demo_live_openai.sh"
  exit 1
fi

CONFIG="examples/configs/demo_openai.yaml"
CORPUS="examples/demo-corpus"
EVALSET="evalsets/demo_openai.jsonl"
REPORT="reports/live_demo_report.html"

echo ""
echo "══════════════════════════════════════════"
echo "  RAGEvalKit Live Demo  (OpenAI)"
echo "══════════════════════════════════════════"
echo ""

# ── Step 1: init ─────────────────────────────────────────────────────────────
echo "▶ [1/10] Initialising .rageval/ workspace..."
rageval init

# ── Step 2: ingest ───────────────────────────────────────────────────────────
echo ""
echo "▶ [2/10] Ingesting demo corpus (embedding with sentence-transformers)..."
rageval ingest "$CORPUS" --config "$CONFIG"

# ── Step 3: generate evalset ─────────────────────────────────────────────────
echo ""
echo "▶ [3/10] Generating evaluation questions (5 questions via OpenAI)..."
rageval generate-evalset "$CORPUS" \
  --num-questions 5 \
  --output "$EVALSET" \
  --config "$CONFIG"

# ── Step 4: run RAG pipeline ──────────────────────────────────────────────────
echo ""
echo "▶ [4/10] Running RAG pipeline (limit 3 items)..."
rageval run \
  --config "$CONFIG" \
  --evalset "$EVALSET" \
  --tag "live-demo" \
  --limit 3

# ── Capture latest run ID ─────────────────────────────────────────────────────
RUN_ID=$(python - <<'EOF'
import duckdb, sys
try:
    con = duckdb.connect(".rageval/runs.db")
    row = con.execute(
        "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    con.close()
    print(row[0] if row else "")
except Exception as e:
    print("", file=sys.stderr)
    sys.exit(1)
EOF
)

if [ -z "$RUN_ID" ]; then
  echo "ERROR: Could not determine run ID from .rageval/runs.db"
  exit 1
fi

echo ""
echo "  Run ID: $RUN_ID"

# ── Step 5: evaluate retrieval ───────────────────────────────────────────────
echo ""
echo "▶ [5/10] Evaluating retrieval metrics (recall@k, MRR)..."
rageval evaluate-retrieval --run "$RUN_ID"

# ── Step 6: evaluate answer relevance ────────────────────────────────────────
echo ""
echo "▶ [6/10] Evaluating answer relevance (via OpenAI judge)..."
rageval evaluate-answer-relevance --run "$RUN_ID" --config "$CONFIG"

# ── Step 7: extract claims ───────────────────────────────────────────────────
echo ""
echo "▶ [7/10] Extracting atomic claims from generated answers..."
rageval extract-claims --run "$RUN_ID" --config "$CONFIG"

# ── Step 8: evaluate groundedness ────────────────────────────────────────────
echo ""
echo "▶ [8/10] Evaluating groundedness of claims (via OpenAI judge)..."
rageval evaluate-groundedness --run "$RUN_ID" --config "$CONFIG"

# ── Step 9: summarise run ────────────────────────────────────────────────────
echo ""
echo "▶ [9/10] Summarising run and classifying root causes..."
rageval summarize-run --run "$RUN_ID"

# ── Step 10: generate report ─────────────────────────────────────────────────
echo ""
echo "▶ [10/10] Generating HTML report..."
mkdir -p reports
rageval report --run "$RUN_ID" --output "$REPORT"

echo ""
echo "══════════════════════════════════════════"
echo "  Demo complete!"
echo "  Report: $REPORT"
echo "  Run ID: $RUN_ID"
echo "══════════════════════════════════════════"
echo ""
echo "Open the report in your browser:"
echo "  open $REPORT          # macOS"
echo "  xdg-open $REPORT      # Linux"
