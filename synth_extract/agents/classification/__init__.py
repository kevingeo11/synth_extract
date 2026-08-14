"""Binary paper classification over OpenAI-compatible APIs."""

from .classifier import (
    FullTextClassifier,
    PaperClassifier,
    TitleAbstractClassifier,
)
from .schemas import (
    ClassificationFailure,
    ClassificationOutcome,
    ClassificationResult,
    CompletionMetadata,
    TokenUsage,
)

__all__ = [
    "ClassificationFailure",
    "ClassificationOutcome",
    "ClassificationResult",
    "CompletionMetadata",
    "FullTextClassifier",
    "PaperClassifier",
    "TitleAbstractClassifier",
    "TokenUsage",
]
