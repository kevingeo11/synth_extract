"""synth_extract package entrypoint."""
from .agents.extractor_agent import ExtractorAgent
from .agents.schemas import ExtractionResult

__all__ = ["ExtractorAgent", "ExtractionResult"]
