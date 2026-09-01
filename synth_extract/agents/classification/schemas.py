"""Pydantic schemas returned by the paper classifier."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """Token consumption reported by the provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class CompletionMetadata(BaseModel):
    """Useful metadata taken from the raw ChatCompletion."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    created: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    reasoning: str | None = None
    usage: TokenUsage | None = None


class ClassificationResult(BaseModel):
    """A binary paper-classification result."""

    model_config = ConfigDict(extra="forbid", strict=True)

    label: bool = Field(
        description="Whether the paper matches the classification criteria."
    )
    metadata: CompletionMetadata


class ClassificationFailure(BaseModel):
    """A local description of why classification did not produce a label."""

    model_config = ConfigDict(extra="forbid")

    error_type: Literal[
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
    message: str


ClassificationOutcome = ClassificationResult | ClassificationFailure


__all__ = [
    "ClassificationFailure",
    "ClassificationOutcome",
    "ClassificationResult",
    "CompletionMetadata",
    "TokenUsage",
]
