"""Provider-neutral paper classification over an OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
)
from pydantic import ValidationError

from synth_extract.agents.classification.schemas import (
    ClassificationFailure,
    ClassificationOutcome,
    ClassificationResult,
)


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPT_DIR / "system_prompt.md"
_USER_TEMPLATE_PATH = _PROMPT_DIR / "user_template.md"


class PaperClassifier:
    """Classify a paper title and abstract with a binary label.

    The client works with any service implementing OpenAI-compatible Chat
    Completions and JSON-schema ``response_format``, including vLLM and
    compatible OpenRouter models/providers.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout: float = 60.0,
        system_prompt_path: str | Path | None = None,
        user_template_path: str | Path | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("VLLM_MODEL")
        if not self.model:
            raise ValueError(
                "A model name is required. Pass model=... or set LLM_MODEL/VLLM_MODEL."
            )

        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("VLLM_API_KEY")
            or "not-required"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt_path = Path(system_prompt_path or _SYSTEM_PROMPT_PATH)
        self.user_template_path = Path(user_template_path or _USER_TEMPLATE_PATH)

        # Disable the OpenAI SDK's built-in retries: every failed call should
        # produce one explicit ClassificationFailure.
        if "max_retries" in client_kwargs:
            raise ValueError("PaperClassifier fixes max_retries=0 by design.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
            **client_kwargs,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
            **client_kwargs,
        )
        self.reload_prompts()

    @staticmethod
    def _load_prompt(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    def reload_prompts(self) -> None:
        """Reload both prompt files from disk."""
        self._system_prompt = self._load_prompt(self.system_prompt_path)
        self._user_template = self._load_prompt(self.user_template_path)

    def system_prompt(self) -> str:
        """Return the active system prompt."""
        return self._system_prompt

    def user_prompt_template(self) -> str:
        """Return the active user prompt template."""
        return self._user_template

    def build_messages(self, title: str, abstract: str) -> list[dict[str, str]]:
        """Build the Chat Completions messages without calling the model."""
        title = title.strip()
        abstract = abstract.strip()
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": self._user_template.format(
                    title=title or "[Not provided]",
                    abstract=abstract or "[Not provided]",
                ),
            },
        ]

    def render_prompt(self, title: str, abstract: str) -> str:
        """Return a readable rendering of the exact messages sent."""
        messages = self.build_messages(title, abstract)
        return "\n\n".join(
            f"=== {message['role'].upper()} ===\n{message['content']}"
            for message in messages
        )

    @staticmethod
    def response_format() -> dict[str, Any]:
        """Return the strict JSON-schema response format for the API call."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "paper_classification",
                "strict": True,
                "schema": ClassificationResult.model_json_schema(),
            },
        }

    def prompt_hash(self) -> str:
        """Return a stable hash for tracking the active prompt pair."""
        prompt_text = f"{self._system_prompt}\n\n{self._user_template}"
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    def llm_config(self) -> dict[str, Any]:
        """Return non-secret configuration useful for debugging."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_provided": bool(self.api_key),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": 0,
            "system_prompt_path": str(self.system_prompt_path),
            "user_template_path": str(self.user_template_path),
            "prompt_hash": self.prompt_hash(),
        }

    @staticmethod
    def _api_failure(exc: Exception) -> ClassificationFailure:
        if isinstance(exc, APITimeoutError):
            return ClassificationFailure(error_type="timeout", message=str(exc))
        if isinstance(exc, APIConnectionError):
            return ClassificationFailure(error_type="transport", message=str(exc))
        if isinstance(exc, APIStatusError):
            error_type = "server" if exc.status_code >= 500 else "provider"
            return ClassificationFailure(
                error_type=error_type,
                message=f"HTTP {exc.status_code}: {exc}",
            )
        return ClassificationFailure(error_type="unknown", message=str(exc))

    def _create_completion(self, title: str, abstract: str) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.build_messages(title, abstract),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=self.response_format(),
        )

    async def _create_completion_async(self, title: str, abstract: str) -> Any:
        return await self.async_client.chat.completions.create(
            model=self.model,
            messages=self.build_messages(title, abstract),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=self.response_format(),
        )

    def classify_raw(
        self,
        title: str,
        abstract: str,
    ) -> Any | ClassificationFailure:
        """Return the raw completion, or an explicit provider failure."""
        if not title.strip() and not abstract.strip():
            return ClassificationFailure(
                error_type="input",
                message="At least one of title or abstract must be non-empty.",
            )
        try:
            return self._create_completion(title, abstract)
        except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
            return self._api_failure(exc)
        except Exception as exc:  # Preserve an inspectable failure at the API boundary.
            return self._api_failure(exc)

    def classify(self, title: str, abstract: str) -> ClassificationOutcome:
        """Classify a paper and return a validated label or failure object."""
        completion = self.classify_raw(title, abstract)
        if isinstance(completion, ClassificationFailure):
            return completion

        return self._parse_completion(completion)

    async def aclassify_raw(
        self,
        title: str,
        abstract: str,
    ) -> Any | ClassificationFailure:
        """Asynchronously return a raw completion or explicit provider failure."""
        if not title.strip() and not abstract.strip():
            return ClassificationFailure(
                error_type="input",
                message="At least one of title or abstract must be non-empty.",
            )
        try:
            return await self._create_completion_async(title, abstract)
        except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
            return self._api_failure(exc)
        except Exception as exc:  # Preserve an inspectable failure at the API boundary.
            return self._api_failure(exc)

    async def aclassify(
        self,
        title: str,
        abstract: str,
    ) -> ClassificationOutcome:
        """Asynchronously classify one paper."""
        completion = await self.aclassify_raw(title, abstract)
        if isinstance(completion, ClassificationFailure):
            return completion

        return self._parse_completion(completion)

    async def aclassify_many(
        self,
        papers: Sequence[tuple[str, str]],
        max_parallel_requests: int = 8,
    ) -> list[ClassificationOutcome]:
        """Classify papers concurrently while preserving input order.

        Only ``max_parallel_requests`` worker coroutines are created, so a large
        input sequence does not create one asyncio task per paper.
        """
        if max_parallel_requests <= 0:
            raise ValueError("max_parallel_requests must be greater than zero")
        if not papers:
            return []

        outcomes: list[ClassificationOutcome | None] = [None] * len(papers)
        indexed_papers = iter(enumerate(papers))

        async def worker() -> None:
            for index, (title, abstract) in indexed_papers:
                outcomes[index] = await self.aclassify(title, abstract)

        worker_count = min(max_parallel_requests, len(papers))
        await asyncio.gather(*(worker() for _ in range(worker_count)))

        # Every index is assigned by exactly one worker before gather returns.
        return [outcome for outcome in outcomes if outcome is not None]

    @staticmethod
    def _parse_completion(completion: Any) -> ClassificationOutcome:
        """Validate one sync or async Chat Completions response."""

        if not completion.choices:
            return ClassificationFailure(
                error_type="empty_response",
                message="The provider returned no completion choices.",
            )

        choice = completion.choices[0]
        if choice.finish_reason == "length":
            return ClassificationFailure(
                error_type="truncated",
                message="The classification response reached the token limit.",
            )

        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            return ClassificationFailure(error_type="refusal", message=refusal)

        content = message.content
        if not content:
            return ClassificationFailure(
                error_type="empty_response",
                message="The provider returned an empty response.",
            )

        try:
            return ClassificationResult.model_validate_json(content)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return ClassificationFailure(
                error_type="invalid_response",
                message=f"The response did not match ClassificationResult: {exc}",
            )

    def close(self) -> None:
        """Close the synchronous HTTP client."""
        self.client.close()

    async def aclose(self) -> None:
        """Close the asynchronous HTTP client."""
        await self.async_client.close()

    def list_models(self) -> list[str] | ClassificationFailure:
        """List model IDs exposed by the configured endpoint."""
        try:
            return [model.id for model in self.client.models.list().data]
        except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
            return self._api_failure(exc)
        except Exception as exc:
            return self._api_failure(exc)

    def health_check(self) -> bool | ClassificationFailure:
        """Check whether the configured endpoint responds to a model-list call."""
        models = self.list_models()
        if isinstance(models, ClassificationFailure):
            return models
        return True


__all__ = ["PaperClassifier"]
