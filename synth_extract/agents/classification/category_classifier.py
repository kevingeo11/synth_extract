"""Full-text classification of polymer material-creation categories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from synth_extract.agents.llm import LLMBackend

from .classifier import PaperClassifier
from .schemas import (
    CategoryClassificationOutcome,
    CategoryClassificationResult,
    ClassificationFailure,
)


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "category"
_SYSTEM_PROMPT_PATH = _PROMPT_DIR / "system_prompt.md"
_USER_TEMPLATE_PATH = _PROMPT_DIR / "user_template.md"


class CategoryClassifier(PaperClassifier):
    """Classify a paper's full text into a polymer-creation category."""

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt_path: str | Path | None = None,
        user_template_path: str | Path | None = None,
    ) -> None:
        """Load category prompts and bind the supplied backend."""
        super().__init__(
            backend=backend,
            system_prompt_path=system_prompt_path or _SYSTEM_PROMPT_PATH,
            user_template_path=user_template_path or _USER_TEMPLATE_PATH,
        )

    @staticmethod
    def response_format() -> dict[str, Any]:
        """Return the category-only JSON schema required from the model."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "paper_category_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "polymer_synthesis",
                                "polymer_modification_combination",
                                "polymer_composite_formulation",
                            ],
                            "description": "Material-creation category.",
                        }
                    },
                    "required": ["category"],
                    "additionalProperties": False,
                },
            },
        }

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

    @classmethod
    def _parse_completion(
        cls,
        completion: Any,
    ) -> CategoryClassificationOutcome:
        """Validate category content and attach provider metadata."""
        if not completion.choices:
            return ClassificationFailure(
                error_type="empty_response",
                message="The provider returned no completion choices.",
            )

        choice = completion.choices[0]
        if choice.finish_reason == "length":
            return ClassificationFailure(
                error_type="truncated",
                message="The category response reached the token limit.",
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
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("The response must be a JSON object.")
            return CategoryClassificationResult.model_validate(
                {
                    **payload,
                    "metadata": cls._completion_metadata(completion),
                }
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return ClassificationFailure(
                error_type="invalid_response",
                message=(
                    "The response did not match "
                    f"CategoryClassificationResult: {exc}"
                ),
            )

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

    def classify(self, fulltext: str) -> CategoryClassificationOutcome:
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

    async def aclassify(
        self,
        fulltext: str,
    ) -> CategoryClassificationOutcome:
        """Asynchronously classify one paper's full text."""
        completion = await self.aclassify_raw(fulltext)
        if isinstance(completion, ClassificationFailure):
            return completion
        return self._parse_completion(completion)


__all__ = ["CategoryClassifier"]
