"""Schemas for extracting sample records from one polymer paper."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ExtractionModel(BaseModel):
    """Strict base model for sample-extraction inputs and outputs."""

    model_config = ConfigDict(extra="forbid")


class Sample(ExtractionModel):
    """One distinct polymer material sample prepared in a paper."""

    name: str = Field(
        description="Polymer or material name as written in the paper, not a code."
    )
    identifier: str = Field(
        description="Label used for this specific sample; empty if none."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names, codes, or identifiers for the same sample.",
    )
    description: str = Field(
        description=(
            "Self-contained description of the sample and the synthesis details "
            "that distinguish it, using only information from the paper."
        )
    )

    @field_validator("name", "description")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("identifier")
    @classmethod
    def strip_identifier(cls, value: str) -> str:
        return value.strip()

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: list[str]) -> list[str]:
        """Normalize aliases and silently discard empty model output."""
        return [alias for value in values if (alias := value.strip())]


class SampleList(ExtractionModel):
    """All qualifying samples extracted from one paper."""

    samples: list[Sample] = Field(
        default_factory=list,
        description="Distinct polymer material samples prepared in the paper.",
    )

    @field_validator("samples", mode="before")
    @classmethod
    def retain_valid_samples(cls, values: Any) -> Any:
        """Drop malformed items without discarding valid sibling samples."""
        if not isinstance(values, list):
            return values

        valid_samples: list[Sample] = []
        for value in values:
            try:
                valid_samples.append(Sample.model_validate(value))
            except (ValidationError, ValueError, TypeError):
                continue
        return valid_samples


class SampleExtractionRequest(ExtractionModel):
    """Input required to extract samples from one Markdown paper."""

    paper_uid: str = Field(description="Local paper identifier used by the caller.")
    markdown_full_text: str = Field(description="Complete paper text in Markdown.")

    @field_validator("paper_uid", "markdown_full_text")
    @classmethod
    def require_nonempty_input(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


__all__ = ["ExtractionModel", "Sample", "SampleExtractionRequest", "SampleList"]
