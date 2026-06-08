from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

from synth_extract.config import OPENROUTER_API_BASE, OPENROUTER_API_KEY
from synth_extract.agents.schemas import (
    build_extraction_parser,
    ExtractionResult,
)

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPT_DIR / "extraction_system_prompt.md"
_USER_TEMPLATE_PATH = _PROMPT_DIR / "extraction_user_template.md"


class ExtractorAgent:
    """LangChain-compatible extractor agent for markdown content."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        **llm_kwargs: Any,
    ) -> None:
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_API_BASE,
            **llm_kwargs,
        )
        self.parser = build_extraction_parser()
        self.prompt = self.build_prompt_template()

    @staticmethod
    def _load_prompt_file(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    def build_prompt_template(self) -> ChatPromptTemplate:
        """Build the LangChain prompt template from markdown files."""
        system_prompt = self._load_prompt_file(_SYSTEM_PROMPT_PATH)
        user_template = self._load_prompt_file(_USER_TEMPLATE_PATH)

        return ChatPromptTemplate(
                    messages=[
                        ("system", system_prompt),
                        ("user", user_template),
                    ],
                    input_variables=["text", "format_instructions"],
                )
        
    
    def extract(self, markdown_text: str) -> ExtractionResult:
        """Extract structured data from markdown text."""
        prompt_inputs = {
            "text": markdown_text,
            "format_instructions": self.parser.get_format_instructions(),
        }

        chain = self.prompt | self.llm | self.parser

        response = chain.invoke(prompt_inputs)

        return response
    
    @staticmethod
    def _clean_prompt_text(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.strip().splitlines())

    def system_prompt(self) -> str:
        """Return the cleaned system prompt text."""
        return self._clean_prompt_text(self._load_prompt_file(_SYSTEM_PROMPT_PATH))

    def user_prompt_template(self) -> str:
        """Return the cleaned user prompt template text."""
        return self._clean_prompt_text(self._load_prompt_file(_USER_TEMPLATE_PATH))

    def build_extraction_prompt(self) -> str:
        """Build the filled prompt text using markdown + format instructions."""
        system_prompt = self.system_prompt()
        user_prompt = self.user_prompt_template()
        filled_user_prompt = user_prompt.format(
            text="<<paper content in markdown format>>",
            format_instructions=self.parser.get_format_instructions(),
        )

        return "\n\n".join(
            [
                "=== System Prompt ===",
                system_prompt,
                "=== User Prompt ===",
                filled_user_prompt,
            ]
        )

    def llm_config(self) -> dict[str, Any]:
        """Return the current LLM configuration."""
        return {
            "model_name": getattr(self.llm, "model_name", None),
            "temperature": getattr(self.llm, "temperature", None),
            "openai_api_base": getattr(self.llm, "openai_api_base", None),
            "api_key_provided": bool(getattr(self.llm, "openai_api_key", None)),
        }

    def extract_raw(self, markdown_text: str) -> str:
        """Invoke the LLM and return its raw generated output."""
        prompt_inputs = {
            "text": markdown_text,
            "format_instructions": self.parser.get_format_instructions(),
        }

        chain = self.prompt | self.llm
        response = chain.invoke(prompt_inputs)

        return response
