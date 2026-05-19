# Common RAG Failure Modes

Retrieval failures occur when the relevant chunk is not surfaced in the top-k results. Grounding failures occur when the model generates claims not supported by the retrieved context. Both failure types can be detected automatically with an LLM-as-judge evaluator that scores each chunk and each claim.
