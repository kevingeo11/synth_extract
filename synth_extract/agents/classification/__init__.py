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
)

__all__ = [
    "ClassificationFailure",
    "ClassificationOutcome",
    "ClassificationResult",
    "FullTextClassifier",
    "PaperClassifier",
    "TitleAbstractClassifier",
]
