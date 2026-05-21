"""Unit tests for the LLM client abstraction."""
import subprocess
import sys

from rageval.core.llm import CompletionResult, LLMClient, MockLLMClient


class TestMockLLMClientConstruction:
    def test_default_response_is_valid_json(self):
        import json
        mock = MockLLMClient()
        result = mock.complete("s", "u")
        data = json.loads(result.text)
        assert "questions" in data

    def test_custom_response_text(self):
        mock = MockLLMClient(response_text="custom")
        assert mock.complete("s", "u").text == "custom"

    def test_custom_model(self):
        mock = MockLLMClient(model="my-model")
        assert mock.complete("s", "u").model == "my-model"

    def test_empty_response(self):
        mock = MockLLMClient(response_text="")
        result = mock.complete("s", "u")
        assert result.text == ""

    def test_fenced_json_passthrough(self):
        fenced = '```json\n{"key": "val"}\n```'
        mock = MockLLMClient(response_text=fenced)
        result = mock.complete("s", "u")
        assert "key" in result.text

    def test_invalid_json_passthrough(self):
        mock = MockLLMClient(response_text="not json at all")
        result = mock.complete("s", "u")
        assert result.text == "not json at all"


class TestMockLLMClientComplete:
    def test_returns_completion_result(self):
        result = MockLLMClient().complete("s", "u")
        assert isinstance(result, CompletionResult)

    def test_prompt_tokens_positive(self):
        result = MockLLMClient().complete("s", "u")
        assert result.prompt_tokens > 0

    def test_total_tokens_ge_completion_tokens(self):
        result = MockLLMClient().complete("s", "u")
        assert result.total_tokens >= result.completion_tokens

    def test_cost_usd_is_none(self):
        result = MockLLMClient().complete("s", "u")
        assert result.cost_usd is None

    def test_records_each_call(self):
        mock = MockLLMClient()
        mock.complete("sys1", "usr1")
        mock.complete("sys2", "usr2")
        assert len(mock.calls) == 2
        assert mock.calls[0] == {"system": "sys1", "user": "usr1"}
        assert mock.calls[1]["system"] == "sys2"

    def test_starts_with_empty_call_log(self):
        mock = MockLLMClient()
        assert mock.calls == []

    def test_satisfies_llm_client_protocol(self):
        assert isinstance(MockLLMClient(), LLMClient)


def test_importing_llm_does_not_import_openai():
    """openai must not be imported at module load time."""
    code = (
        "import sys; "
        "from rageval.core.llm import MockLLMClient; "
        "assert 'openai' not in sys.modules, 'openai was imported eagerly'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"openai was loaded eagerly:\n{result.stderr}"
