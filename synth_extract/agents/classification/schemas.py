"""Pydantic schemas returned by the paper classifier."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ClassificationResult(BaseModel):
    """A successfully validated binary paper classification."""

    model_config = ConfigDict(extra="forbid", strict=True)

    label: bool


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
]
