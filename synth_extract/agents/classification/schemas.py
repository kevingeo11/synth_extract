"""Pydantic schemas returned by the paper classifier."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ClassificationCategory = Literal[
    "polymer_synthesis",
    "polymer_modification_combination",
    "polymer_composite_formulation",
]

CanonicalProperty = Literal[
    "number-average molecular weight",
    "weight-average molecular weight",
    "dispersity",
    "degree of polymerization",
    "glass transition temperature",
    "melting temperature",
    "thermal decomposition temperature",
    "Young's modulus",
    "tensile strength",
    "elongation at break",
    "storage modulus",
    "melt viscosity",
    "degree of crystallinity",
    "d-spacing",
    "particle size",
    "density",
    "solubility",
]
CANONICAL_PROPERTIES: tuple[str, ...] = get_args(CanonicalProperty)


class PropertyPresenceRequest(BaseModel):
    """Full-text input for one property-presence classification."""

    model_config = ConfigDict(extra="forbid")

    markdown_full_text: str = Field(description="Complete paper text in Markdown.")

    @field_validator("markdown_full_text")
    @classmethod
    def require_full_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("markdown_full_text must be non-empty")
        return value


class PropertyPresenceResult(BaseModel):
    """Canonical target properties reported for an eligible paper sample."""

    model_config = ConfigDict(extra="forbid", strict=True)

    properties: list[CanonicalProperty] = Field(
        default_factory=list,
        description="Distinct canonical target properties reported in the paper.",
    )
    has_property: bool = Field(
        description="True exactly when the properties list is non-empty."
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "PropertyPresenceResult":
        present = set(self.properties)
        self.properties = [
            property_name
            for property_name in CANONICAL_PROPERTIES
            if property_name in present
        ]
        if self.has_property != bool(self.properties):
            raise ValueError("has_property must equal bool(properties)")
        return self


class TokenUsage(BaseModel):
    """Token consumption reported by the provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class CompletionMetadata(BaseModel):
    """Useful metadata taken from the raw ChatCompletion."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    created: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    reasoning: str | None = None
    usage: TokenUsage | None = None


class ClassificationResult(BaseModel):
    """A binary paper-classification result."""

    model_config = ConfigDict(extra="forbid", strict=True)

    label: bool = Field(
        description="Whether the paper matches the classification criteria."
    )
    metadata: CompletionMetadata


class CategoryClassificationResult(BaseModel):
    """A polymer material-creation category with completion metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    category: ClassificationCategory = Field(
        description="Material-creation category."
    )
    metadata: CompletionMetadata


class ClassificationFailure(BaseModel):
    """A local description of why classification did not produce a label."""

    model_config = ConfigDict(extra="forbid")

    error_type: Literal[
        "input",
        "timeout",
        "transport",
        "server",
        "provider",
        "refusal",
        "truncated",
        "empty_response",
        "invalid_response",
        "unknown",
    ]
    message: str


ClassificationOutcome = ClassificationResult | ClassificationFailure
CategoryClassificationOutcome = (
    CategoryClassificationResult | ClassificationFailure
)


__all__ = [
    "CANONICAL_PROPERTIES",
    "CanonicalProperty",
    "CategoryClassificationOutcome",
    "CategoryClassificationResult",
    "ClassificationCategory",
    "ClassificationFailure",
    "ClassificationOutcome",
    "ClassificationResult",
    "CompletionMetadata",
    "PropertyPresenceRequest",
    "PropertyPresenceResult",
    "TokenUsage",
]
