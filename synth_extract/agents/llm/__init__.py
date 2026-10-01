"""Reusable OpenAI-compatible language-model interfaces."""

from .backend import LLMBackend, LLMBackendError
from .agent import (
    GeneralAgent,
    StructuredAgent,
    StructuredCompletionMetadata,
    StructuredFailure,
    StructuredOutcome,
    StructuredSuccess,
    StructuredTask,
    StructuredTokenUsage,
)

__all__ = [
    "GeneralAgent",
    "LLMBackend",
    "LLMBackendError",
    "StructuredAgent",
    "StructuredCompletionMetadata",
    "StructuredFailure",
    "StructuredOutcome",
    "StructuredSuccess",
    "StructuredTask",
    "StructuredTokenUsage",
]
