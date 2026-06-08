from __future__ import annotations

from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel


class PolymerEntry(BaseModel):
    polymer_name: Optional[str] = None
    glass_transition_temperature: Optional[str] = None
    synthesis_procedure: Optional[str] = None


class ExtractionResult(BaseModel):
    polymers: List[PolymerEntry]


def build_extraction_parser() -> PydanticOutputParser[ExtractionResult]:
    """Build an output parser for structured extraction results."""
    return PydanticOutputParser(pydantic_object=ExtractionResult)


