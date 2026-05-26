# RAG Failure Modes

RAG pipelines fail in distinct ways that require different remediation strategies. Understanding which component failed — the retriever, the generator, or both — is essential for improving system quality. RAGEvalKit classifies failures into three primary modes: retrieval failure, grounding failure, and answer relevance failure.

## Retrieval Failure

Retrieval failure occurs when the retriever does not return the chunks needed to answer a question. This is detected by a recall@k score of 0.0 or below a configured minimum threshold. The retriever returned chunks, but none of them were relevant to the question.

Common causes of retrieval failure include:
- **Vocabulary mismatch**: the question uses different terminology than the documents (e.g., asking about "model hallucination" when documents use "factual error").
- **Chunk granularity**: the relevant information is split across chunks in a way that makes individual chunks score low in similarity.
- **Embedding model limitations**: the embedding model does not capture the semantic relationship between the question and the relevant passage.
- **Corpus gaps**: the answer genuinely does not exist in the corpus.

When retrieval fails, the generator has no grounded context to work from. This often triggers cascading failure: because the retrieved chunks are irrelevant, the generator either admits it does not know or — more dangerously — hallucinates a plausible-sounding but incorrect answer. This is why a retrieval_failure primary cause frequently co-occurs with a grounding_failure secondary cause.

**Diagnosis**: If recall@k is consistently 0.0 for a category of questions, examine whether those questions use vocabulary that appears in the documents. Try lower chunk sizes, overlapping chunks, or a different embedding model.

## Grounding Failure

Grounding failure occurs when the generated answer makes claims that are not supported by the retrieved context. This is measured by the faithfulness metric. A faithfulness score below the threshold (for example, below 0.8) indicates that a significant fraction of the answer's claims are unsupported.

Grounding failure can occur even when retrieval succeeds. The model may:
- **Extrapolate beyond the context**: the model uses its parametric knowledge to fill gaps, adding facts not present in the retrieved chunks.
- **Confuse chunks**: the model combines information from different chunks in an incorrect way.
- **Ignore the context**: the model produces a fluent but context-free answer based entirely on training data.

**Diagnosis**: Inspect the claim-level evaluation to see which specific claims were marked as contradicted or not_enough_info. Compare the claims against the retrieved chunk text. If the model is systematically hallucinating, consider adding an explicit instruction in the system prompt to only use the provided context.

## Answer Relevance Failure

Answer relevance failure occurs when the generated answer does not address the question asked, even if the answer is factually correct and grounded. A judge model scores relevance on a 1–5 scale. An answer scoring below 3 is labelled fail.

Examples of answer relevance failure:
- The answer addresses a related but different question.
- The answer is too vague or general to be useful.
- The answer says "I don't know" when the context contains the answer.

**Diagnosis**: Review the question and answer pair. If the model is consistently deflecting (saying it cannot answer) when relevant chunks exist, the system prompt or RAG prompt template may need to more explicitly encourage the model to extract and state information from the context.

## Cascading and Multiple Failures

In practice, failures often cascade. Retrieval failure is the most dangerous starting point because a bad retrieval set makes both grounding and relevance harder. When both retrieval_failure and grounding_failure are observed for many items, fixing retrieval should be the first priority — improving grounding when retrieval is broken will have limited impact.

RAGEvalKit records both a primary cause and secondary causes for each item, enabling analysis of which failure mode dominates and which failure modes co-occur.
