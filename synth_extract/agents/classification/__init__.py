"""Binary paper classification over OpenAI-compatible APIs."""

from .core import PaperClassifier
from .schemas import (
    ClassificationFailure,
    ClassificationOutcome,
    ClassificationResult,
)

__all__ = [
    "ClassificationFailure",
    "ClassificationOutcome",
    "ClassificationResult",
    "PaperClassifier",
]
