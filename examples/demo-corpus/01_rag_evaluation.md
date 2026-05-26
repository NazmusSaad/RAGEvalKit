# RAG Evaluation Metrics

Retrieval-Augmented Generation (RAG) pipelines combine a retrieval step with a language model to produce grounded answers. Evaluating these pipelines requires measuring both the quality of the retrieved context and the quality of the generated answer separately, since failures can originate at either stage.

## Retrieval Metrics

**Recall at k (recall@k)** measures whether the retriever surfaced all the chunks that are relevant to a question. Given a set of ground-truth relevant chunk IDs and the top-k retrieved chunks, recall@k is computed as:

```
recall@k = |retrieved_top_k ∩ relevant_chunks| / |relevant_chunks|
```

A recall@k score of 1.0 means every relevant chunk was retrieved. A score of 0.0 means the retriever returned no relevant chunks at all. When the ground-truth relevant chunks are unknown (empty source_chunk_ids), recall@k cannot be computed and the item is labelled unknown.

**Mean Reciprocal Rank (MRR)** measures how highly the first relevant chunk is ranked. If the first relevant chunk appears at rank position r, then the reciprocal rank is 1/r. MRR is the mean of reciprocal ranks across all questions. An MRR of 1.0 means the first relevant chunk is always ranked first. MRR is used as a diagnostic metric alongside recall@k.

## Answer Quality Metrics

**Faithfulness** (also called groundedness) measures whether the generated answer is supported by the retrieved context. To compute faithfulness, the answer is broken into atomic claims. Each claim is then verified against the retrieved chunks by a judge model. The faithfulness score is the fraction of claims that are supported by retrieved context. A score of 1.0 means every claim in the answer is grounded. A score of 0.0 means the answer is entirely unsupported.

**Answer relevance** measures whether the generated answer actually addresses the question asked. A judge model scores relevance on a scale from 1 to 5, where 5 means the answer directly and completely answers the question, and 1 means the answer is off-topic or irrelevant. Scores above a threshold (for example, 3/5) are labelled pass; scores below are labelled fail.

## Overall Item Labels

Each evaluated question-answer pair receives an overall label based on all metric results:

- **pass**: all required metrics (recall@k, answer relevance, faithfulness) are present and labelled pass.
- **fail**: at least one required metric is present and labelled fail.
- **unknown**: at least one required metric is missing or its judge returned an uncertain result.

The overall label summarises whether a given item represents a successful RAG response.

## Root Cause Classification

When an item fails, RAGEvalKit classifies the most likely root cause:

- **retrieval_failure**: recall@k is 0 or below threshold, meaning the retriever did not surface relevant context.
- **grounding_failure**: faithfulness is below threshold, meaning the answer contains claims not supported by retrieved context.
- **answer_relevance_failure**: answer relevance is below threshold, meaning the answer does not address the question.
- **missing_metric**: a required metric was not computed, preventing root cause determination.
- **judge_uncertain**: the judge model returned an ambiguous result.
- **none**: no failure detected; the item passed all checks.

Secondary causes can also be recorded. For example, when retrieval fails, the model may also hallucinate (grounding_failure), making both the primary and secondary causes relevant.
