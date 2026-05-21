from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    glob: str = "**/*.{md,pdf,txt}"


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["recursive", "sentence", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["sentence_transformers", "openai", "dummy"] = "sentence_transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64


class VectorStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["chroma"] = "chroma"
    path: str = ".rageval/chroma"
    collection: str = "docs_v1"
    distance: Literal["cosine", "l2", "ip"] = "cosine"


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = 5
    rerank: str | None = None
    filter: dict[str, Any] | None = None


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    system_prompt_path: str = "prompts/system.txt"
    prompt_template_path: str = "prompts/rag.j2"


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_concurrent: int = 4


class EvalSetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class CostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_1k: float = 0.015
    output_per_1k: float = 0.060


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str
    seed: int = 42
    corpus: CorpusConfig
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    judge: JudgeConfig = JudgeConfig()
    evaluators: list[str] = ["retrieval_relevance", "groundedness", "answer_relevance"]
    evalset: EvalSetConfig
    cost: CostConfig = CostConfig()

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)


class AbsoluteThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness_min: float = 0.80
    retrieval_relevance_min: float = 0.70


class RelativeThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness_drop_max: float = 0.05
    retrieval_relevance_drop_max: float = 0.05
    recall_at_k_drop_max: float = 0.05
    answer_relevance_drop_max: float = 0.05
    p50_latency_increase_max: float = 0.30
    cost_per_query_increase_max: float = 0.30


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_all_absolute: bool = True
    require_all_relative: bool = True
    allow_unknown_as_pass: bool = False


class ThresholdsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    absolute: AbsoluteThresholds = AbsoluteThresholds()
    relative: RelativeThresholds = RelativeThresholds()
    policy: PolicyConfig = PolicyConfig()

    @classmethod
    def from_yaml(cls, path: Path | str) -> ThresholdsConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)
