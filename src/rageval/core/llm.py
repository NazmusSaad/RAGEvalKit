from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rageval.core.config import JudgeConfig


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    model: str


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResult: ...


# Canned response used by MockLLMClient when constructed via create_llm_client.
# Tests that need specific scenarios (fenced JSON, bad JSON, etc.) construct
# MockLLMClient(response_text=...) directly.
_DEFAULT_MOCK_RESPONSE: str = _json.dumps({
    "questions": [
        {
            "question": "What is the main concept described in this passage?",
            "reference_answer": "The passage describes the primary concept at hand.",
            "qtype": "factoid",
            "difficulty": "easy",
        },
        {
            "question": "How do the components described relate to the overall process?",
            "reference_answer": "The components interact through the described mechanism.",
            "qtype": "reasoning",
            "difficulty": "medium",
        },
    ]
})


class MockLLMClient:
    """Configurable test double for :class:`LLMClient`.

    Pass any string as *response_text* to control what ``complete()`` returns:
    valid JSON, fenced JSON, malformed JSON, or empty strings.
    All calls are recorded in ``self.calls`` for assertion.
    """

    def __init__(self, response_text: str = _DEFAULT_MOCK_RESPONSE, model: str = "mock") -> None:
        self.response_text = response_text
        self.model = model
        self.calls: list[dict[str, str]] = []

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        self.calls.append({"system": system, "user": user})
        words = len(self.response_text.split())
        return CompletionResult(
            text=self.response_text,
            prompt_tokens=50,
            completion_tokens=max(1, words),
            total_tokens=50 + max(1, words),
            cost_usd=None,
            model=self.model,
        )


class OpenAIClient:
    """LLMClient backed by the OpenAI API.

    The ``openai`` SDK is imported lazily inside ``complete()`` so importing
    this module never requires the package to be installed or causes a
    FutureWarning at load time.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        import openai  # lazy import

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = response.usage
        return CompletionResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=None,
            model=self._model,
        )


def create_llm_client(config: JudgeConfig) -> LLMClient:
    """Build an :class:`LLMClient` from a :class:`JudgeConfig`."""
    if config.provider == "openai":
        return OpenAIClient(model=config.model)
    if config.provider == "mock":
        return MockLLMClient(response_text=_DEFAULT_MOCK_RESPONSE, model=config.model)
    raise ValueError(f"Unknown LLM provider: {config.provider!r}")
