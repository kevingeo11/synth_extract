"""Paper classifiers built on a reusable OpenAI-compatible backend."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from synth_extract.agents.llm import LLMBackend, LLMBackendError

from .schemas import (
    ClassificationFailure,
    ClassificationOutcome,
    ClassificationResult,
)


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_TITLE_ABSTRACT_PROMPT_DIR = _PROMPT_DIR / "title_abstract"
_TITLE_ABSTRACT_SYSTEM_PROMPT_PATH = (
    _TITLE_ABSTRACT_PROMPT_DIR / "system_prompt.md"
)
_TITLE_ABSTRACT_USER_TEMPLATE_PATH = (
    _TITLE_ABSTRACT_PROMPT_DIR / "user_template.md"
)
_FULL_TEXT_PROMPT_DIR = _PROMPT_DIR / "full_text"
_FULL_TEXT_SYSTEM_PROMPT_PATH = _FULL_TEXT_PROMPT_DIR / "system_prompt.md"
_FULL_TEXT_USER_TEMPLATE_PATH = _FULL_TEXT_PROMPT_DIR / "user_template.md"


class PaperClassifier(ABC):
    """Shared machinery for structured binary paper classification.

    The base class owns task-level behavior: prompt files, the strict Pydantic
    response schema, response validation, failure objects, prompt diagnostics,
    and health-check delegation. It does not know which paper fields form an
    input. Concrete subclasses validate their inputs and build messages before
    calling the shared request helpers.

    API communication and HTTP-client ownership belong exclusively to the
    injected :class:`~synth_extract.agents.llm.LLMBackend`. Direct construction
    is intentionally unsupported because :meth:`build_messages` has no generic
    paper representation.
    """

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt_path: str | Path,
        user_template_path: str | Path,
    ) -> None:
        """Bind a backend and load the classifier's prompt pair from disk."""
        self.backend = backend
        self.system_prompt_path = Path(system_prompt_path)
        self.user_template_path = Path(user_template_path)
        self.reload_prompts()

    @property
    def model(self) -> str:
        """Return the model identifier configured on the backend."""
        return self.backend.model

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible base URL used by the backend."""
        return self.backend.base_url

    @property
    def api_key(self) -> str:
        """Return the API key configured on the backend."""
        return self.backend.api_key

    @property
    def temperature(self) -> float:
        """Return the backend's sampling temperature."""
        return self.backend.temperature

    @property
    def max_tokens(self) -> int:
        """Return the backend's maximum completion-token count."""
        return self.backend.max_tokens

    @property
    def timeout(self) -> float:
        """Return the backend's request timeout in seconds."""
        return self.backend.timeout

    @staticmethod
    def _load_prompt(path: Path) -> str:
        """Read a UTF-8 prompt file and remove surrounding whitespace."""
        return path.read_text(encoding="utf-8").strip()

    def reload_prompts(self) -> None:
        """Reload the system prompt and user template from their files."""
        self._system_prompt = self._load_prompt(self.system_prompt_path)
        self._user_template = self._load_prompt(self.user_template_path)

    def system_prompt(self) -> str:
        """Return the active system prompt exactly as sent to the model."""
        return self._system_prompt

    def user_prompt_template(self) -> str:
        """Return the active, unformatted user prompt template."""
        return self._user_template

    @abstractmethod
    def build_messages(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        """Validate/format subclass input into Chat Completions messages."""
        raise NotImplementedError

    @staticmethod
    def _render_messages(messages: list[dict[str, str]]) -> str:
        """Render already-built messages in a readable debugging format."""
        return "\n\n".join(
            f"=== {message['role'].upper()} ===\n{message['content']}"
            for message in messages
        )

    def render_full_message(self, *args: Any, **kwargs: Any) -> str:
        """Render the complete request body without calling the model.

        The concrete classifier first builds its input-specific messages. The
        backend then renders the same model, messages, generation settings, and
        structured response format used by real sync and async requests.
        """
        messages = self.build_messages(*args, **kwargs)
        return self.backend.render_request(
            messages=messages,
            response_format=self.response_format(),
        )

    @staticmethod
    def response_format() -> dict[str, Any]:
        """Return the strict JSON schema required from the model endpoint."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "paper_classification",
                "strict": True,
                "schema": ClassificationResult.model_json_schema(),
            },
        }

    def prompt_hash(self) -> str:
        """Return a stable SHA-256 hash of the active prompt pair."""
        prompt_text = f"{self._system_prompt}\n\n{self._user_template}"
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    def llm_config(self) -> dict[str, Any]:
        """Return non-secret backend and prompt configuration for debugging."""
        return {
            **self.backend.config(),
            "system_prompt_path": str(self.system_prompt_path),
            "user_template_path": str(self.user_template_path),
            "prompt_hash": self.prompt_hash(),
        }

    @staticmethod
    def _request_failure(exc: Exception) -> ClassificationFailure:
        """Convert backend or unexpected request errors into the public schema."""
        if isinstance(exc, LLMBackendError):
            return ClassificationFailure(
                error_type=exc.error_type,
                message=exc.message,
            )
        return ClassificationFailure(error_type="unknown", message=str(exc))

    def _classify_messages_raw(
        self,
        messages: list[dict[str, str]],
    ) -> Any | ClassificationFailure:
        """Synchronously request a raw completion for prepared messages."""
        try:
            return self.backend.create_completion(
                messages=messages,
                response_format=self.response_format(),
            )
        except Exception as exc:
            return self._request_failure(exc)

    async def _aclassify_messages_raw(
        self,
        messages: list[dict[str, str]],
    ) -> Any | ClassificationFailure:
        """Asynchronously request a raw completion for prepared messages."""
        try:
            return await self.backend.acreate_completion(
                messages=messages,
                response_format=self.response_format(),
            )
        except Exception as exc:
            return self._request_failure(exc)

    @staticmethod
    def _parse_completion(completion: Any) -> ClassificationOutcome:
        """Validate one raw Chat Completions response.

        Provider refusals, truncation, missing choices/content, and invalid JSON
        are returned as explicit failures. Invalid content is never coerced into
        a label.
        """
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

    def list_models(self) -> list[str] | ClassificationFailure:
        """List endpoint model IDs or return an explicit provider failure."""
        try:
            return self.backend.list_models()
        except Exception as exc:
            return self._request_failure(exc)

    def health_check(self) -> bool | ClassificationFailure:
        """Return ``True`` when the endpoint responds to a model-list request."""
        models = self.list_models()
        if isinstance(models, ClassificationFailure):
            return models
        return True

    def close(self) -> None:
        """Close the backend's synchronous client.

        Prefer calling ``backend.close()`` in new code. This forwarding method
        retains the former classifier lifecycle API without transferring client
        ownership back to the classifier.
        """
        self.backend.close()

    async def aclose(self) -> None:
        """Close the backend's asynchronous client.

        Prefer calling ``await backend.aclose()`` in new code.
        """
        await self.backend.aclose()


class TitleAbstractClassifier(PaperClassifier):
    """Classify a paper from its title and abstract.

    At least one input must contain non-whitespace text. Missing title or
    abstract values are rendered as ``[Not provided]``. Sync and async methods
    return either a strictly validated :class:`ClassificationResult` or a
    :class:`ClassificationFailure`; they never silently invent a label.
    """

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt_path: str | Path | None = None,
        user_template_path: str | Path | None = None,
    ) -> None:
        """Load title/abstract prompts and bind the supplied backend."""
        super().__init__(
            backend=backend,
            system_prompt_path=(
                system_prompt_path or _TITLE_ABSTRACT_SYSTEM_PROMPT_PATH
            ),
            user_template_path=(
                user_template_path or _TITLE_ABSTRACT_USER_TEMPLATE_PATH
            ),
        )

    def build_messages(self, title: str, abstract: str) -> list[dict[str, str]]:
        """Build the exact system and user messages without calling the model."""
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
        """Return a readable rendering of the exact messages to be sent."""
        return self._render_messages(self.build_messages(title, abstract))

    @staticmethod
    def _validate_input(title: str, abstract: str) -> ClassificationFailure | None:
        """Return the established failure when both paper fields are empty."""
        if not title.strip() and not abstract.strip():
            return ClassificationFailure(
                error_type="input",
                message="At least one of title or abstract must be non-empty.",
            )
        return None

    def classify_raw(
        self,
        title: str,
        abstract: str,
    ) -> Any | ClassificationFailure:
        """Return the raw synchronous completion or an explicit failure."""
        input_failure = self._validate_input(title, abstract)
        if input_failure is not None:
            return input_failure
        try:
            messages = self.build_messages(title, abstract)
        except Exception as exc:
            return self._request_failure(exc)
        return self._classify_messages_raw(messages)

    def classify(self, title: str, abstract: str) -> ClassificationOutcome:
        """Synchronously classify one title/abstract pair."""
        completion = self.classify_raw(title, abstract)
        if isinstance(completion, ClassificationFailure):
            return completion
        return self._parse_completion(completion)

    async def aclassify_raw(
        self,
        title: str,
        abstract: str,
    ) -> Any | ClassificationFailure:
        """Return the raw asynchronous completion or an explicit failure."""
        input_failure = self._validate_input(title, abstract)
        if input_failure is not None:
            return input_failure
        try:
            messages = self.build_messages(title, abstract)
        except Exception as exc:
            return self._request_failure(exc)
        return await self._aclassify_messages_raw(messages)

    async def aclassify(
        self,
        title: str,
        abstract: str,
    ) -> ClassificationOutcome:
        """Asynchronously classify one title/abstract pair."""
        completion = await self.aclassify_raw(title, abstract)
        if isinstance(completion, ClassificationFailure):
            return completion
        return self._parse_completion(completion)


class FullTextClassifier(PaperClassifier):
    """Classify a paper from its full text.

    The input must contain non-whitespace text. Sync and async methods return
    either a strictly validated :class:`ClassificationResult` or a
    :class:`ClassificationFailure`; they never silently invent a label.
    """

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt_path: str | Path | None = None,
        user_template_path: str | Path | None = None,
    ) -> None:
        """Load full-text prompts and bind the supplied backend."""
        super().__init__(
            backend=backend,
            system_prompt_path=(
                system_prompt_path or _FULL_TEXT_SYSTEM_PROMPT_PATH
            ),
            user_template_path=(
                user_template_path or _FULL_TEXT_USER_TEMPLATE_PATH
            ),
        )

    def build_messages(self, fulltext: str) -> list[dict[str, str]]:
        """Build the exact system and user messages without calling the model."""
        fulltext = fulltext.strip()
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": self._user_template.format(fulltext=fulltext),
            },
        ]

    def render_prompt(self, fulltext: str) -> str:
        """Return a readable rendering of the exact messages to be sent."""
        return self._render_messages(self.build_messages(fulltext))

    @staticmethod
    def _validate_input(fulltext: str) -> ClassificationFailure | None:
        """Return an input failure when the full text is empty."""
        if not fulltext.strip():
            return ClassificationFailure(
                error_type="input",
                message="Full text must be non-empty.",
            )
        return None

    def classify_raw(self, fulltext: str) -> Any | ClassificationFailure:
        """Return the raw synchronous completion or an explicit failure."""
        input_failure = self._validate_input(fulltext)
        if input_failure is not None:
            return input_failure
        try:
            messages = self.build_messages(fulltext)
        except Exception as exc:
            return self._request_failure(exc)
        return self._classify_messages_raw(messages)

    def classify(self, fulltext: str) -> ClassificationOutcome:
        """Synchronously classify one paper's full text."""
        completion = self.classify_raw(fulltext)
        if isinstance(completion, ClassificationFailure):
            return completion
        return self._parse_completion(completion)

    async def aclassify_raw(
        self,
        fulltext: str,
    ) -> Any | ClassificationFailure:
        """Return the raw asynchronous completion or an explicit failure."""
        input_failure = self._validate_input(fulltext)
        if input_failure is not None:
            return input_failure
        try:
            messages = self.build_messages(fulltext)
        except Exception as exc:
            return self._request_failure(exc)
        return await self._aclassify_messages_raw(messages)

    async def aclassify(self, fulltext: str) -> ClassificationOutcome:
        """Asynchronously classify one paper's full text."""
        completion = await self.aclassify_raw(fulltext)
        if isinstance(completion, ClassificationFailure):
            return completion
        return self._parse_completion(completion)


__all__ = [
    "FullTextClassifier",
    "PaperClassifier",
    "TitleAbstractClassifier",
]
