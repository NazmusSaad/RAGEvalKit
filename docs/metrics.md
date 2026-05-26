# Metrics Reference

RAGEvalKit evaluates both the retrieval stage and the generation stage independently. This is intentional: a model can produce a fluent answer from bad context, and an LLM can fail to answer even when the retriever surfaced the right chunks. Measuring both separately is the only reliable way to attribute failures.

---

## Retrieval Metrics

### recall@k

Measures whether the retriever surfaced all ground-truth relevant chunks in the top-k results.

```
recall@k = |retrieved_top_k ∩ relevant_chunks| / |relevant_chunks|
```

- **1.0** — every relevant chunk was retrieved
- **0.0** — no relevant chunks were retrieved
- **unknown** — `source_chunk_ids` is empty in the evalset (ground truth unavailable)

`source_chunk_ids` in the evalset come from `generate-evalset`, which records which chunks it used when writing each question. This is the retrieval ground truth.

**Diagnosis when recall@k = 0.0:**
- Vocabulary mismatch between the question and the document
- Chunk granularity too coarse (relevant information split across chunks)
- Embedding model does not capture the semantic relationship
- Corpus gap: the answer genuinely does not exist in the corpus

### MRR (Mean Reciprocal Rank)

Measures how highly the first relevant chunk is ranked. If the first relevant chunk appears at rank position `r`, the reciprocal rank is `1/r`. MRR is the mean across all questions.

```
MRR = mean(1/r_i)  where r_i = rank of first relevant chunk for question i
```

- **1.0** — the first relevant chunk is always ranked first
- **0.5** — the first relevant chunk is on average ranked second
- **0.0** — no relevant chunks retrieved

MRR is more sensitive to ranking order than recall@k. A retriever that always puts the relevant chunk at rank 5 might have recall@5 = 1.0 but MRR = 0.2.

---

## Generation Metrics

### answer_relevance

Measures whether the generated answer actually addresses the question asked. Scored by an LLM judge on a 1–5 scale.

| Score | Meaning |
|-------|---------|
| 5 | Answer directly and completely addresses the question |
| 4 | Answer addresses the question with minor gaps |
| 3 | Answer partially addresses the question |
| 2 | Answer is related but does not address the question |
| 1 | Answer is off-topic or irrelevant |

Scores ≥ 3 are labelled `pass`; scores < 3 are labelled `fail`.

The raw score stored in DuckDB is in the range [1, 5]. The mean reported by `summarize-run` and `report` is the mean raw score divided by 5, normalizing to [0, 1] for comparability with other metrics.

**Diagnosis when answer_relevance fails:**
- Model deflects ("I don't know") despite the context containing the answer → strengthen the system prompt or RAG template
- Answer addresses a related but different question → review retrieval quality and prompt template
- Answer is too vague or general → add instructions to be specific in the system prompt

### faithfulness (groundedness)

Measures whether the generated answer is supported by the retrieved context. Computed at the claim level.

1. `extract-claims` decomposes the answer into atomic claims (one fact per claim)
2. `evaluate-groundedness` checks each claim against the retrieved chunks using an LLM judge

Each claim receives one of:
- `supported` — the claim is directly supported by the retrieved chunks
- `contradicted` — the claim is explicitly contradicted by the retrieved chunks
- `not_enough_info` — the retrieved chunks do not provide enough information to verify the claim

```
faithfulness = |supported claims| / |total claims|
```

- **1.0** — every claim in the answer is grounded in the retrieved context
- **0.0** — no claims are supported (the answer is entirely ungrounded)

**Diagnosis when faithfulness fails:**
- Model extrapolates beyond context: add an explicit instruction in the system prompt ("Only use the provided context. Do not use outside knowledge.")
- Model confuses chunks: check whether retrieved chunks actually contain the right information (use `report` to inspect)
- Retrieval failed: if recall@k = 0, the model had no grounded context to begin with

---

## Overall Item Labels

After `summarize-run`, each run item receives one of:

| Label | Condition |
|-------|-----------|
| `pass` | All required metrics are present and labelled pass |
| `fail` | At least one required metric is present and labelled fail |
| `unknown` | At least one required metric is missing or returned an uncertain judge result |

---

## Root-Cause Classification

When an item fails, `summarize-run` assigns a primary root cause:

| Root cause | Meaning |
|------------|---------|
| `retrieval_failure` | recall@k = 0 or below threshold |
| `grounding_failure` | faithfulness below threshold |
| `answer_relevance_failure` | answer relevance below threshold |
| `missing_metric` | A required metric was not computed |
| `judge_uncertain` | The judge model returned an ambiguous result |
| `none` | No failure; item passed all checks |

Secondary causes are also recorded. Retrieval failure frequently co-occurs with grounding failure: when the retriever returns irrelevant chunks, the model has no grounded context, so it may hallucinate (grounding_failure). When both appear together, fixing retrieval is the higher-leverage intervention.

---

## The Case for Evaluating Both Stages

Consider this scenario: a recall@k = 0.0 item (retrieval failure) where the model still produces a relevant, faithful-sounding answer. This happens because the model draws on parametric knowledge when the retrieved chunks are unhelpful.

- An answer-quality-only evaluation labels this item **pass**.
- RAGEvalKit labels it **fail** (primary cause: `retrieval_failure`).

The answer-only label is misleading: if you deploy a different question where the model's parametric knowledge is wrong, you'll get a hallucinated answer with no way to detect it. The retrieval regression is a latent defect.

This is why RAGEvalKit measures both stages and prioritizes retrieval health in root-cause classification.
