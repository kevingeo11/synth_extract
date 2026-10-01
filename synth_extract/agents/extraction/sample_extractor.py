"""Sample-centric full-text extraction agent."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from synth_extract.agents.llm import (
    LLMBackend,
    StructuredAgent,
    StructuredOutcome,
    StructuredTask,
)

from .schemas import SampleExtractionRequest, SampleList


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "sample"
DEFAULT_SYSTEM_PROMPT_PATH = _PROMPT_DIR / "system_prompt.md"
DEFAULT_USER_TEMPLATE_PATH = _PROMPT_DIR / "user_template.md"


class SampleExtractor:
    """Extract the distinct polymer samples prepared in one full-text paper.

    The class defines the prompt and structured-output contract. Concurrency,
    batching, retries, and persistence are intentionally owned by the caller.
    """

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
        user_template_path: str | Path = DEFAULT_USER_TEMPLATE_PATH,
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
        """Reload the system prompt and user template from disk."""
        self._system_prompt = self._load_prompt(self.system_prompt_path)
        self._user_template = self._load_prompt(self.user_template_path)
        self._task = StructuredTask[SampleExtractionRequest, SampleList](
            name="sample_extraction",
            prompt_builder=self._build_user_prompt,
            output_model=SampleList,
            response_format=self.response_format(),
            system_prompt=self._system_prompt,
        )

    def _build_user_prompt(self, request: SampleExtractionRequest) -> str:
        return self._user_template.format(
            markdown_full_text=request.markdown_full_text
        )

    @staticmethod
    def response_format() -> dict[str, Any]:
        """Return the provider-facing strict JSON schema."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "sample_list",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "samples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "identifier": {"type": "string"},
                                    "aliases": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                    },
                                    "description": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                },
                                "required": [
                                    "name",
                                    "identifier",
                                    "aliases",
                                    "description",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["samples"],
                    "additionalProperties": False,
                },
            },
        }

    def extract(
        self,
        request: SampleExtractionRequest,
    ) -> StructuredOutcome[SampleList]:
        """Synchronously extract and validate samples from one paper."""
        return self._agent.run(self._task, request)

    async def aextract(
        self,
        request: SampleExtractionRequest,
    ) -> StructuredOutcome[SampleList]:
        """Asynchronously extract and validate samples from one paper."""
        return await self._agent.arun(self._task, request)

    def render_request(self, request: SampleExtractionRequest) -> str:
        """Render the exact backend request without calling the model."""
        return self._agent.render_request(self._task, request)

    def prompt_hash(self) -> str:
        text = f"{self._system_prompt}\n\n{self._user_template}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def llm_config(self) -> dict[str, Any]:
        """Return non-secret backend and prompt configuration."""
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
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "DEFAULT_USER_TEMPLATE_PATH",
    "SampleExtractor",
]
