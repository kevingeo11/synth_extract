"""Full-text classification for a fixed polymer-property vocabulary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from synth_extract.agents.llm import (
    LLMBackend,
    StructuredAgent,
    StructuredFailure,
    StructuredOutcome,
    StructuredTask,
)

from .schemas import (
    CANONICAL_PROPERTIES,
    PropertyPresenceRequest,
    PropertyPresenceResult,
)


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "property_presence"
DEFAULT_PROPERTY_PRESENCE_SYSTEM_PROMPT = _PROMPT_DIR / "system_prompt.md"
DEFAULT_PROPERTY_PRESENCE_USER_TEMPLATE = _PROMPT_DIR / "user_template.md"


class PropertyPresenceClassifier:
    """Identify which target properties occur for eligible prepared samples."""

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt_path: str | Path = DEFAULT_PROPERTY_PRESENCE_SYSTEM_PROMPT,
        user_template_path: str | Path = DEFAULT_PROPERTY_PRESENCE_USER_TEMPLATE,
    ) -> None:
        self.backend = backend
        self.system_prompt_path = Path(system_prompt_path).expanduser().resolve()
        self.user_template_path = Path(user_template_path).expanduser().resolve()
        self._agent = StructuredAgent(backend)
        self.reload_prompts()

    @staticmethod
    def _load_prompt(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {path}")
        return text

    def reload_prompts(self) -> None:
        """Reload prompts and rebuild the reusable structured task."""
        self._system_prompt = self._load_prompt(self.system_prompt_path)
        self._user_template = self._load_prompt(self.user_template_path)
        self._task = StructuredTask[PropertyPresenceRequest, PropertyPresenceResult](
            name="property_presence",
            prompt_builder=self._build_user_prompt,
            output_model=PropertyPresenceResult,
            response_format=self.response_format(),
            system_prompt=self._system_prompt,
        )

    def _build_user_prompt(self, request: PropertyPresenceRequest) -> str:
        return self._user_template.format(
            markdown_full_text=request.markdown_full_text
        )

    @staticmethod
    def response_format() -> dict[str, Any]:
        """Return the strict provider-facing classification schema."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "property_presence",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "properties": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(CANONICAL_PROPERTIES),
                            },
                        },
                        "has_property": {"type": "boolean"},
                    },
                    "required": ["properties", "has_property"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _request(markdown_full_text: str) -> PropertyPresenceRequest:
        return PropertyPresenceRequest(markdown_full_text=markdown_full_text)

    def classify(
        self,
        markdown_full_text: str,
    ) -> StructuredOutcome[PropertyPresenceResult]:
        """Synchronously classify one complete Markdown paper."""
        try:
            request = self._request(markdown_full_text)
        except Exception as exc:
            return StructuredFailure(
                error_type="input",
                message=f"{type(exc).__name__}: {exc}",
            )
        return self._agent.run(self._task, request)

    async def aclassify(
        self,
        markdown_full_text: str,
    ) -> StructuredOutcome[PropertyPresenceResult]:
        """Asynchronously classify one complete Markdown paper."""
        try:
            request = self._request(markdown_full_text)
        except Exception as exc:
            return StructuredFailure(
                error_type="input",
                message=f"{type(exc).__name__}: {exc}",
            )
        return await self._agent.arun(self._task, request)

    def render_request(self, markdown_full_text: str) -> str:
        """Render the exact request without calling the provider."""
        return self._agent.render_request(
            self._task,
            self._request(markdown_full_text),
        )

    def prompt_hash(self) -> str:
        text = f"{self._system_prompt}\n\n{self._user_template}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def llm_config(self) -> dict[str, Any]:
        return {
            **self.backend.config(),
            "system_prompt_path": str(self.system_prompt_path),
            "user_template_path": str(self.user_template_path),
            "prompt_hash": self.prompt_hash(),
        }

    def health_check(self) -> bool:
        self.backend.list_models()
        return True


__all__ = [
    "DEFAULT_PROPERTY_PRESENCE_SYSTEM_PROMPT",
    "DEFAULT_PROPERTY_PRESENCE_USER_TEMPLATE",
    "PropertyPresenceClassifier",
]
