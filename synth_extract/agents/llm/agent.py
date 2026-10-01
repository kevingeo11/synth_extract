"""General-purpose text and schema-validated interfaces to an LLM backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar, overload

from pydantic import BaseModel, ConfigDict, ValidationError

from .backend import LLMBackend, LLMBackendError


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT", bound=BaseModel)


class GeneralAgent:
    """Send arbitrary prompts through an injected :class:`LLMBackend`.

    The agent keeps no conversation state and imposes no output schema. A call
    may contain only a user prompt, or a system prompt plus a user prompt.
    Provider-specific reasoning can be enabled through ``LLMBackend.extra_body``;
    pass ``include_reasoning=True`` to receive it alongside the answer.
    """

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def set_reasoning(self, enabled: bool = True) -> None:
        """Enable or disable model reasoning in the backend request body.

        Other ``extra_body`` values and existing chat-template settings are
        preserved. Calling ``set_reasoning()`` without an argument enables
        reasoning.
        """
        extra_body = dict(self.backend.extra_body or {})
        existing_kwargs = extra_body.get("chat_template_kwargs")
        chat_template_kwargs = (
            dict(existing_kwargs) if isinstance(existing_kwargs, Mapping) else {}
        )
        chat_template_kwargs["enable_thinking"] = enabled
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        self.backend.extra_body = extra_body

    @staticmethod
    def build_messages(
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages for a user-only or system-plus-user prompt."""
        if not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty when provided")

        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def _read_completion(completion: Any) -> tuple[str, str | None]:
        """Extract answer and optional reasoning from a raw chat completion."""
        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError("The provider returned no completion choices.")

        message = choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"The provider refused the request: {refusal}")

        content = getattr(message, "content", None)
        reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning_content", None)

        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        if reasoning is not None and not isinstance(reasoning, str):
            reasoning = str(reasoning)

        if not content and not reasoning:
            raise RuntimeError("The provider returned an empty response.")
        return content, reasoning

    @overload
    def chat(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        include_reasoning: Literal[False] = False,
    ) -> str: ...

    @overload
    def chat(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        include_reasoning: Literal[True],
    ) -> tuple[str, str | None]: ...

    def chat(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        include_reasoning: bool = False,
    ) -> str | tuple[str, str | None]:
        """Return a plain-text completion and, optionally, provider reasoning."""
        completion = self.backend.create_completion(
            messages=self.build_messages(user_prompt, system_prompt)
        )
        content, reasoning = self._read_completion(completion)
        if include_reasoning:
            return content, reasoning
        return content

    @overload
    async def achat(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        include_reasoning: Literal[False] = False,
    ) -> str: ...

    @overload
    async def achat(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        include_reasoning: Literal[True],
    ) -> tuple[str, str | None]: ...

    async def achat(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        include_reasoning: bool = False,
    ) -> str | tuple[str, str | None]:
        """Asynchronously return text and, optionally, provider reasoning."""
        completion = await self.backend.acreate_completion(
            messages=self.build_messages(user_prompt, system_prompt)
        )
        content, reasoning = self._read_completion(completion)
        if include_reasoning:
            return content, reasoning
        return content

    def render_request(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Render the schema-free request without calling the model."""
        return self.backend.render_request(
            messages=self.build_messages(user_prompt, system_prompt)
        )


StructuredErrorType = Literal[
    "input",
    "timeout",
    "transport",
    "server",
    "provider",
    "refusal",
    "truncated",
    "empty_response",
    "invalid_response",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class StructuredTask(Generic[InputT, OutputT]):
    """Describe how one input becomes one schema-validated LLM result.

    The provider-facing response format remains explicit because compatible
    endpoints may not implement every keyword produced by Pydantic's JSON
    Schema generator. ``validate_result`` can enforce relationships between
    the input and output, such as preserving input item IDs and order.
    """

    name: str
    prompt_builder: Callable[[InputT], str]
    output_model: type[OutputT]
    response_format: Mapping[str, Any]
    system_prompt: str | None = None
    validate_result: Callable[[InputT, OutputT], None] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("StructuredTask.name must be non-empty")
        if not issubclass(self.output_model, BaseModel):
            raise TypeError("StructuredTask.output_model must be a Pydantic model")
        if not self.response_format:
            raise ValueError("StructuredTask.response_format must be non-empty")
        if self.system_prompt is not None and not self.system_prompt.strip():
            raise ValueError(
                "StructuredTask.system_prompt must be non-empty when provided"
            )


class StructuredTokenUsage(BaseModel):
    """Token consumption reported by the completion provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class StructuredCompletionMetadata(BaseModel):
    """Provider metadata attached locally to a structured result."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    created: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    reasoning: str | None = None
    usage: StructuredTokenUsage | None = None


class StructuredSuccess(BaseModel, Generic[OutputT]):
    """A successfully parsed and validated structured completion."""

    model_config = ConfigDict(extra="forbid")

    result: OutputT
    metadata: StructuredCompletionMetadata
    raw_response: str


class StructuredFailure(BaseModel):
    """Describe why a structured request did not produce a valid result."""

    model_config = ConfigDict(extra="forbid")

    error_type: StructuredErrorType
    message: str
    raw_response: str | None = None


StructuredOutcome: TypeAlias = StructuredSuccess[OutputT] | StructuredFailure


class StructuredAgent:
    """Execute one schema-constrained task synchronously or asynchronously.

    The agent owns no task definition and imposes no concurrency policy.
    Callers remain responsible for semaphores, batching, persistence, retries,
    and progress reporting.
    """

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    @staticmethod
    def build_messages(
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build a user-only or system-plus-user chat request."""
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("The task prompt must be a non-empty string")
        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty when provided")

        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def _request_failure(exc: Exception) -> StructuredFailure:
        if isinstance(exc, LLMBackendError):
            return StructuredFailure(
                error_type=exc.error_type,
                message=exc.message,
            )
        return StructuredFailure(
            error_type="unknown",
            message=f"{type(exc).__name__}: {exc}",
        )

    @staticmethod
    def _completion_metadata(completion: Any) -> StructuredCompletionMetadata:
        choice = completion.choices[0]
        message = choice.message
        usage = getattr(completion, "usage", None)

        token_usage = None
        if usage is not None:
            token_usage = StructuredTokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )

        reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning_content", None)
        if reasoning is not None and not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return StructuredCompletionMetadata(
            model=getattr(completion, "model", None),
            created=getattr(completion, "created", None),
            finish_reason=getattr(choice, "finish_reason", None),
            stop_reason=getattr(choice, "stop_reason", None),
            reasoning=reasoning,
            usage=token_usage,
        )

    @staticmethod
    def _build_task_messages(
        task: StructuredTask[InputT, OutputT],
        task_input: InputT,
    ) -> list[dict[str, str]] | StructuredFailure:
        try:
            user_prompt = task.prompt_builder(task_input)
            return StructuredAgent.build_messages(
                user_prompt=user_prompt,
                system_prompt=task.system_prompt,
            )
        except Exception as exc:
            return StructuredFailure(
                error_type="input",
                message=f"{type(exc).__name__}: {exc}",
            )

    @classmethod
    def _parse_completion(
        cls,
        task: StructuredTask[InputT, OutputT],
        task_input: InputT,
        completion: Any,
    ) -> StructuredOutcome[OutputT]:
        choices = getattr(completion, "choices", None)
        if not choices:
            return StructuredFailure(
                error_type="empty_response",
                message="The provider returned no completion choices.",
            )

        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            return StructuredFailure(
                error_type="truncated",
                message="The structured response reached the token limit.",
            )

        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            return StructuredFailure(
                error_type="refusal",
                message=str(refusal),
            )

        content = getattr(message, "content", None)
        if not content:
            return StructuredFailure(
                error_type="empty_response",
                message="The provider returned an empty response.",
            )
        if not isinstance(content, str):
            return StructuredFailure(
                error_type="invalid_response",
                message="The provider response content was not a string.",
                raw_response=str(content),
            )

        try:
            result = task.output_model.model_validate_json(content)
            if task.validate_result is not None:
                task.validate_result(task_input, result)
        except (ValidationError, ValueError, TypeError) as exc:
            return StructuredFailure(
                error_type="invalid_response",
                message=(
                    f"The response did not match task {task.name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ),
                raw_response=content,
            )

        return StructuredSuccess[OutputT](
            result=result,
            metadata=cls._completion_metadata(completion),
            raw_response=content,
        )

    def render_request(
        self,
        task: StructuredTask[InputT, OutputT],
        task_input: InputT,
    ) -> str:
        """Render the exact structured request without calling the provider."""
        messages = self._build_task_messages(task, task_input)
        if isinstance(messages, StructuredFailure):
            raise ValueError(messages.message)
        return self.backend.render_request(
            messages=messages,
            response_format=task.response_format,
        )

    def run(
        self,
        task: StructuredTask[InputT, OutputT],
        task_input: InputT,
    ) -> StructuredOutcome[OutputT]:
        """Execute and validate one synchronous structured request."""
        messages = self._build_task_messages(task, task_input)
        if isinstance(messages, StructuredFailure):
            return messages

        try:
            completion = self.backend.create_completion(
                messages=messages,
                response_format=task.response_format,
            )
        except Exception as exc:
            return self._request_failure(exc)
        return self._parse_completion(task, task_input, completion)

    async def arun(
        self,
        task: StructuredTask[InputT, OutputT],
        task_input: InputT,
    ) -> StructuredOutcome[OutputT]:
        """Execute and validate one asynchronous structured request."""
        messages = self._build_task_messages(task, task_input)
        if isinstance(messages, StructuredFailure):
            return messages

        try:
            completion = await self.backend.acreate_completion(
                messages=messages,
                response_format=task.response_format,
            )
        except Exception as exc:
            return self._request_failure(exc)
        return self._parse_completion(task, task_input, completion)


__all__ = [
    "GeneralAgent",
    "StructuredAgent",
    "StructuredCompletionMetadata",
    "StructuredFailure",
    "StructuredOutcome",
    "StructuredSuccess",
    "StructuredTask",
    "StructuredTokenUsage",
]
