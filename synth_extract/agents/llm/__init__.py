"""Reusable OpenAI-compatible language-model backends."""

from .backend import LLMBackend, LLMBackendError

__all__ = ["LLMBackend", "LLMBackendError"]
