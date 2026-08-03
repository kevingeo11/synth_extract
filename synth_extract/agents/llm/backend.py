"""Provider-neutral communication with an OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, NoReturn

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
)


BackendErrorType = Literal[
    "timeout",
    "transport",
    "server",
    "provider",
    "unknown",
]


class LLMBackendError(Exception):
    """Describe a failed OpenAI-compatible API operation.

    The backend normalizes SDK-specific exceptions into stable categories so
    agents do not need to import or understand the OpenAI Python SDK's error
    hierarchy. The original exception remains available through exception
    chaining.
    """

    def __init__(self, error_type: BackendErrorType, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class LLMBackend:
    """Own sync and async clients for one OpenAI-compatible model endpoint.

    This class is deliberately unaware of agents, prompts, input schemas, and
    output schemas. Callers provide already-built messages and an optional
    response format. The backend contributes only endpoint configuration and
    performs the HTTP request with retries disabled.

    Both clients are created once and reused. This allows many concurrent
    calls to :meth:`acreate_completion` without constructing a client for each
    request. The owner must close both clients with :meth:`close` and
    :meth:`aclose` when they are no longer needed.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout: float = 60.0,
        **client_kwargs: Any,
    ) -> None:
        """Configure reusable clients and generation defaults.

        All endpoint values are explicit; the backend does not inspect
        environment variables or supply provider-specific defaults. SDK
        retries are fixed at zero so each backend call represents exactly one
        provider request.
        """
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        if not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not api_key.strip():
            raise ValueError("api_key must be a non-empty string")

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if "max_retries" in client_kwargs:
            raise ValueError("LLMBackend fixes max_retries=0 by design.")

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
            **client_kwargs,
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
            **client_kwargs,
        )

    @staticmethod
    def _raise_backend_error(exc: Exception) -> NoReturn:
        """Raise a provider-neutral error corresponding to an SDK exception."""
        if isinstance(exc, APITimeoutError):
            error = LLMBackendError("timeout", str(exc))
        elif isinstance(exc, APIConnectionError):
            error = LLMBackendError("transport", str(exc))
        elif isinstance(exc, APIStatusError):
            error_type: BackendErrorType = (
                "server" if exc.status_code >= 500 else "provider"
            )
            error = LLMBackendError(
                error_type,
                f"HTTP {exc.status_code}: {exc}",
            )
        else:
            error = LLMBackendError("unknown", str(exc))
        raise error from exc

    def create_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        response_format: Mapping[str, Any] | None = None,
    ) -> Any:
        """Create one synchronous chat completion.

        The raw SDK response is returned unchanged. Any request failure is
        raised as :class:`LLMBackendError`; interpreting response content is
        the responsibility of the calling agent.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            request["response_format"] = response_format

        try:
            return self._client.chat.completions.create(**request)
        except Exception as exc:
            self._raise_backend_error(exc)

    async def acreate_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        response_format: Mapping[str, Any] | None = None,
    ) -> Any:
        """Create one asynchronous chat completion.

        Concurrent calls share the backend's asynchronous HTTP client. The
        backend deliberately imposes no concurrency limit; the calling agent
        or application controls request scheduling.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            request["response_format"] = response_format

        try:
            return await self._async_client.chat.completions.create(**request)
        except Exception as exc:
            self._raise_backend_error(exc)

    def list_models(self) -> list[str]:
        """Return model identifiers exposed by the configured endpoint."""
        try:
            return [model.id for model in self._client.models.list().data]
        except Exception as exc:
            self._raise_backend_error(exc)

    def config(self) -> dict[str, Any]:
        """Return non-secret configuration suitable for logs and debugging."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_provided": bool(self.api_key),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": 0,
        }

    def close(self) -> None:
        """Close the synchronous HTTP client owned by this backend."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the asynchronous HTTP client owned by this backend."""
        await self._async_client.close()


__all__ = ["LLMBackend", "LLMBackendError"]
