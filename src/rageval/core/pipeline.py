from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Template

from rageval.core.config import PipelineConfig
from rageval.core.embedder import Embedder
from rageval.core.ids import sha256_text
from rageval.core.llm import LLMClient
from rageval.core.retrieval import retrieve_top_k
from rageval.storage.chroma_dao import RetrievedChunk


@dataclass
class QueryTrace:
    """Full trace of one RAG question-answer cycle.

    ``item_id`` is deterministic: sha256(run_id:question_id).
    ``retrieved_contexts`` is a snapshot of the chunks retrieved at run time.
    ``error`` is non-None when retrieval or generation raised an exception;
    the run continues to the next question in that case.
    """

    item_id: str
    run_id: str
    question_id: str
    question: str
    generated_answer: str
    retrieved_contexts: list[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float | None
    latency_ms: int
    model: str
    error: str | None = None


def _load_text(path: Path) -> str:
    """Read *path*, returning an empty string if the file is missing."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


class RAGPipeline:
    """Executes the full RAG cycle (retrieve → generate) for a single question.

    Per-question errors are caught and stored in :attr:`QueryTrace.error` so
    the run never aborts mid-way due to a single bad question.
    """

    def __init__(
        self,
        config: PipelineConfig,
        embedder: Embedder,
        llm_client: LLMClient,
        project_dir: Path,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._llm_client = llm_client
        self._chroma_path = Path(project_dir) / config.vector_store.path
        self._system_prompt = _load_text(
            Path(project_dir) / config.generation.system_prompt_path
        )
        self._user_template = Template(
            _load_text(Path(project_dir) / config.generation.prompt_template_path)
        )

    def run_question(
        self,
        run_id: str,
        question_id: str,
        question: str,
    ) -> QueryTrace:
        """Run retrieve → generate for one question, capturing a full trace.

        Exceptions are stored in :attr:`QueryTrace.error`; the caller never
        sees them propagate.
        """
        item_id = sha256_text(f"{run_id}:{question_id}")
        t0 = time.perf_counter()

        generated_answer = ""
        retrieved_contexts: list[RetrievedChunk] = []
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd: float | None = None
        error: str | None = None

        try:
            retrieved_contexts = retrieve_top_k(
                query_text=question,
                chroma_path=self._chroma_path,
                collection_name=self._config.vector_store.collection,
                embedder=self._embedder,
                top_k=self._config.retrieval.top_k,
            )

            user_prompt = self._user_template.render(
                question=question,
                chunks=retrieved_contexts,
            )

            result = self._llm_client.complete(
                system=self._system_prompt,
                user=user_prompt,
                temperature=self._config.generation.temperature,
                max_tokens=self._config.generation.max_tokens,
            )

            generated_answer = result.text
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            cost_usd = result.cost_usd

        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        latency_ms = max(0, int((time.perf_counter() - t0) * 1000))

        return QueryTrace(
            item_id=item_id,
            run_id=run_id,
            question_id=question_id,
            question=question,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=cost_usd,
            latency_ms=latency_ms,
            model=self._config.generation.model,
            error=error,
        )
