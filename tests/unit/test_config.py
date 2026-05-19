import pytest
from pydantic import ValidationError

from rageval.core.config import (
    PipelineConfig,
    ThresholdsConfig,
)

_MINIMAL = {
    "name": "test",
    "corpus": {"path": "./docs"},
    "evalset": {"path": "evalsets/v1.jsonl"},
}


class TestPipelineConfig:
    def test_minimal_config_uses_defaults(self):
        cfg = PipelineConfig(**_MINIMAL)
        assert cfg.version == 1
        assert cfg.seed == 42
        assert cfg.chunking.chunk_size == 512
        assert cfg.chunking.chunk_overlap == 64
        assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"
        assert cfg.retrieval.top_k == 5
        assert cfg.generation.model == "gpt-4o-mini"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            PipelineConfig(**_MINIMAL, unknown_field="bad")

    def test_invalid_chunking_strategy_raises(self):
        with pytest.raises(ValidationError):
            PipelineConfig(**_MINIMAL, chunking={"strategy": "invalid"})

    def test_invalid_embedding_provider_raises(self):
        with pytest.raises(ValidationError):
            PipelineConfig(**_MINIMAL, embedding={"provider": "cohere"})

    def test_custom_retrieval_top_k(self):
        cfg = PipelineConfig(**_MINIMAL, retrieval={"top_k": 10})
        assert cfg.retrieval.top_k == 10

    def test_nullable_retrieval_fields_default_to_none(self):
        cfg = PipelineConfig(**_MINIMAL)
        assert cfg.retrieval.rerank is None
        assert cfg.retrieval.filter is None

    def test_from_yaml_roundtrip(self, tmp_path):
        import yaml

        data = {
            "version": 1,
            "name": "roundtrip",
            "seed": 99,
            "corpus": {"path": "./data"},
            "evalset": {"path": "evalsets/test.jsonl"},
        }
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(data))
        cfg = PipelineConfig.from_yaml(yaml_file)
        assert cfg.name == "roundtrip"
        assert cfg.seed == 99


class TestThresholdsConfig:
    def test_defaults(self):
        cfg = ThresholdsConfig()
        assert cfg.absolute.faithfulness_min == 0.80
        assert cfg.absolute.retrieval_relevance_min == 0.70
        assert cfg.relative.faithfulness_drop_max == 0.05
        assert cfg.policy.require_all_absolute is True
        assert cfg.policy.allow_unknown_as_pass is False

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            ThresholdsConfig(bad_field=1)

    def test_custom_absolute_threshold(self):
        cfg = ThresholdsConfig(absolute={"faithfulness_min": 0.75, "retrieval_relevance_min": 0.65})
        assert cfg.absolute.faithfulness_min == 0.75

    def test_from_yaml_roundtrip(self, tmp_path):
        import yaml

        data = {
            "version": 1,
            "absolute": {"faithfulness_min": 0.75, "retrieval_relevance_min": 0.65},
            "relative": {
                "faithfulness_drop_max": 0.10,
                "retrieval_relevance_drop_max": 0.10,
                "recall_at_k_drop_max": 0.05,
                "answer_relevance_drop_max": 0.05,
                "p50_latency_increase_max": 0.30,
                "cost_per_query_increase_max": 0.30,
            },
            "policy": {
                "require_all_absolute": True,
                "require_all_relative": False,
                "allow_unknown_as_pass": False,
            },
        }
        yaml_file = tmp_path / "rageval.yaml"
        yaml_file.write_text(yaml.dump(data))
        cfg = ThresholdsConfig.from_yaml(yaml_file)
        assert cfg.absolute.faithfulness_min == 0.75
        assert cfg.policy.require_all_relative is False
