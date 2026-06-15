from __future__ import annotations

from typing import List, Optional, Literal

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# class PolymerEntry(BaseModel):
#     polymer_name: Optional[str] = None
#     glass_transition_temperature: Optional[str] = None
#     synthesis_procedure: Optional[str] = None


# class ExtractionResult(BaseModel):
#     polymers: List[PolymerEntry]


class PaperMetadata(BaseModel):
    """Basic paper-level metadata."""

    title: Optional[str] = Field(
        default=None,
        description="Title of the paper."
    )
    doi: Optional[str] = Field(
        default=None,
        description="DOI of the paper, if available."
    )
    year: Optional[str] = Field(
        default=None,
        description="Publication year, if available."
    )
    journal: Optional[str] = Field(
        default=None,
        description="Journal name, if available."
    )


class TgMeasurement(BaseModel):
    """Glass transition temperature or closely related Tg/softening transition."""

    property_value_raw: Optional[str] = Field(
        default=None,
        description="Reported 'Tg', 'T_g', 'glass transition temperature' value exactly as written, can include  including units, ranges, uncertainty.",
    )
    property_unit: Optional[str] = Field(
        default=None,
        description="Unit of the Tg value, e.g. °C, K, °F.",
    )
    method: Optional[str] = Field(
        default=None,
        description=(
            "Measurement method if explicitly stated or clear from context, e.g. DSC, DMA, TMA. "
            "Leave null if unknown."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any additional notes or context about the Tg measurement"
    )
    

class SampleRecord(BaseModel):
    """One experimentally prepared material/sample.

    Do not merge records merely because two samples have the same polymer name or monomers.
    Different table rows/runs/samples should usually be separate records.
    """

    sample_label: Optional[str] = Field(
        default=None,
        description="Sample label used in the paper",
    )
    sample_aliases: list[str] = Field(default_factory=list)


    polymer_name_raw: Optional[str] = Field(
        default=None,
        description="Polymer/material name exactly or nearly as written.",
    )
    polymer_abbreviation: Optional[str] = Field(
        default=None,
        description="Polymer abbreviation if stated",
    )
    architecture_raw: Optional[str] = Field(
        default=None,
        description="Raw architecture description, e.g. homopolymer, block copolymer etc.",
    )


    polymerization_reaction_raw: Optional[str] = Field(
        default=None,
        description="Polymerization reaction/type as written",
    )

    synthesis_procedure_text: Optional[str] = Field(
        default=None,
        description="Free-text synthesis procedure relevant to this sample or sample series."
    )


    glass_transition_temperatures: Optional[TgMeasurement] = Field(
        default_factory=list,
        description="Tg or Tg-like softening transitions linked to this sample.",
    )

    needs_review: Optional[bool] = Field(
        default=None,
        description="True if identity, synthesis, or Tg link is ambiguous.",
    )

    notes: Optional[str] = Field(
        default=None,
        description=(
            "Short note explaining ambiguity, missing Tg, shared series-level Tg, "
            "or assumptions made during extraction."
        ),
    )


class ExtractionResult(BaseModel):
    """Top-level extraction result for one markdown paper."""

    paper: PaperMetadata = Field(default_factory=PaperMetadata)

    samples: list[SampleRecord] = Field(
        default_factory=list,
        description=(
            "One record per distinct experimentally prepared and characterized sample/material. "
            "Do not merge samples with the same monomers if table rows, conditions, or properties differ."
        ),
    )


def build_extraction_parser() -> PydanticOutputParser[ExtractionResult]:
    """Build an output parser for structured extraction results."""
    return PydanticOutputParser(pydantic_object=ExtractionResult)