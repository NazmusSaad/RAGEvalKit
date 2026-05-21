"""Unit tests for RAGPipeline — uses DummyEmbedder and MockLLMClient."""
import textwrap

import pytest

from rageval.core.chunker import Chunk
from rageval.core.config import (
    CorpusConfig,
    EmbeddingConfig,
    EvalSetConfig,
    GenerationConfig,
    PipelineConfig,
    VectorStoreConfig,
)
from rageval.core.embedder import DummyEmbedder
from rageval.core.llm import MockLLMClient
from rageval.core.pipeline import QueryTrace, RAGPipeline
from rageval.storage.chroma_dao import get_or_create_collection, upsert_chunks

_DIM = 16
_EMB = DummyEmbedder(dim=_DIM)
_MOCK_ANSWER = "This is a mock generated answer."


def _make_config(collection: str = "test_col") -> PipelineConfig:
    return PipelineConfig(
        name="test",
        corpus=CorpusConfig(path="./docs"),
        embedding=EmbeddingConfig(provider="dummy", model="dummy"),
        vector_store=VectorStoreConfig(path=".rageval/chroma", collection=collection),
        generation=GenerationConfig(
            provider="mock", model="mock-model",
            temperature=0.0, max_tokens=100,
        ),
        evalset=EvalSetConfig(path="evalsets/test.jsonl"),
    )


@pytest.fixture
def pipeline_env(tmp_path):
    """Creates prompt files and a populated Chroma collection, returns (pipeline, mock_llm)."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.txt").write_text("You are a helpful assistant.", encoding="utf-8")
    (prompts / "rag.j2").write_text(
        textwrap.dedent("""\
            {% for chunk in chunks %}[{{ loop.index }}] {{ chunk.text }}
            {% endfor %}
            Question: {{ question }}
            Answer:"""),
        encoding="utf-8",
    )

    # Populate Chroma with 3 chunks
    chroma_path = tmp_path / ".rageval" / "chroma"
    col = get_or_create_collection(chroma_path, "test_col")
    chunks = [
        Chunk(chunk_id=f"c{i}", doc_id="doc1", ordinal=i, text=f"Chunk {i} about RAG.", num_chars=18)
        for i in range(3)
    ]
    upsert_chunks(col, chunks, _EMB.embed([c.text for c in chunks]))

    mock_llm = MockLLMClient(response_text=_MOCK_ANSWER)
    pipeline = RAGPipeline(_make_config(), _EMB, mock_llm, tmp_path)
    return pipeline, mock_llm


class TestQueryTrace:
    def test_returns_query_trace(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "What is RAG?")
        assert isinstance(trace, QueryTrace)

    def test_item_id_is_deterministic(self, pipeline_env):
        pipeline, _ = pipeline_env
        t1 = pipeline.run_question("run1", "q1", "Q?")
        t2 = pipeline.run_question("run1", "q1", "Q?")
        assert t1.item_id == t2.item_id

    def test_item_id_changes_with_run_id(self, pipeline_env):
        pipeline, _ = pipeline_env
        t1 = pipeline.run_question("runA", "q1", "Q?")
        t2 = pipeline.run_question("runB", "q1", "Q?")
        assert t1.item_id != t2.item_id

    def test_item_id_changes_with_question_id(self, pipeline_env):
        pipeline, _ = pipeline_env
        t1 = pipeline.run_question("run1", "q1", "Q?")
        t2 = pipeline.run_question("run1", "q2", "Q?")
        assert t1.item_id != t2.item_id

    def test_generated_answer_is_mock_response(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Q?")
        assert trace.generated_answer == _MOCK_ANSWER

    def test_run_id_propagated(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("my-run", "q1", "Q?")
        assert trace.run_id == "my-run"

    def test_question_id_propagated(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "my-q", "Q?")
        assert trace.question_id == "my-q"

    def test_question_text_propagated(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Is RAG useful?")
        assert trace.question == "Is RAG useful?"

    def test_retrieved_contexts_populated(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Chunk 0 about RAG.")
        assert len(trace.retrieved_contexts) > 0

    def test_chunk_text_snapshotted(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Chunk 0 about RAG.")
        assert all(ctx.text for ctx in trace.retrieved_contexts)

    def test_chunk_ids_present(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Q?")
        assert all(ctx.chunk_id for ctx in trace.retrieved_contexts)

    def test_latency_ms_nonnegative(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Q?")
        assert trace.latency_ms >= 0

    def test_model_field(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Q?")
        assert trace.model == "mock-model"

    def test_no_error_on_success(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Q?")
        assert trace.error is None

    def test_prompt_tokens_set(self, pipeline_env):
        pipeline, _ = pipeline_env
        trace = pipeline.run_question("run1", "q1", "Q?")
        assert trace.prompt_tokens > 0

    def test_llm_was_called(self, pipeline_env):
        pipeline, mock_llm = pipeline_env
        pipeline.run_question("run1", "q1", "Q?")
        assert len(mock_llm.calls) == 1

    def test_empty_collection_no_error(self, tmp_path):
        """Retrieval from an empty collection returns [] without raising."""
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "system.txt").write_text("System.", encoding="utf-8")
        (prompts / "rag.j2").write_text("{{ question }}", encoding="utf-8")

        config = _make_config(collection="empty_xyz")
        mock_llm = MockLLMClient(response_text="Answer.")
        pipeline = RAGPipeline(config, _EMB, mock_llm, tmp_path)

        trace = pipeline.run_question("run1", "q1", "Q?")
        assert trace.retrieved_contexts == []
        assert trace.error is None

    def test_missing_prompt_files_no_crash(self, tmp_path):
        """Missing prompt files produce empty prompts; pipeline still runs."""
        config = _make_config()
        chroma_path = tmp_path / ".rageval" / "chroma"
        col = get_or_create_collection(chroma_path, "test_col")
        chunks = [Chunk(chunk_id="c0", doc_id="d0", ordinal=0, text="text", num_chars=4)]
        upsert_chunks(col, chunks, _EMB.embed(["text"]))

        mock_llm = MockLLMClient(response_text="Answer.")
        pipeline = RAGPipeline(config, _EMB, mock_llm, tmp_path)  # no prompts/ dir

        trace = pipeline.run_question("run1", "q1", "Q?")
        assert trace.error is None
        assert trace.generated_answer == "Answer."
