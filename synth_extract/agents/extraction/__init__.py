"""Full-text material extraction agents."""

from .property_config import PROPERTY_CONFIG, STRONG_MATCHES, PropertyKey
from .property_extractor import PaperPropertyExtractor, SamplePropertyExtractor
from .property_prompts import DEFAULT_PROPERTY_PROMPT_DIR
from .property_schemas import (
    FLAGS,
    Condition,
    ExtractedRecord,
    PaperContext,
    PaperPropertyList,
    PaperPropertyRecord,
    PaperPropertyRequest,
    PaperPropertyResult,
    Property,
    PropertyRecord,
    RequestFailure,
    SamplePropertyList,
    SamplePropertyRequest,
    SampleRef,
)
from .sample_extractor import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_USER_TEMPLATE_PATH,
    SampleExtractor,
)
from .schemas import ExtractionModel, Sample, SampleExtractionRequest, SampleList

__all__ = [
    "Condition",
    "DEFAULT_PROPERTY_PROMPT_DIR",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "DEFAULT_USER_TEMPLATE_PATH",
    "ExtractedRecord",
    "ExtractionModel",
    "FLAGS",
    "PROPERTY_CONFIG",
    "PaperContext",
    "PaperPropertyExtractor",
    "PaperPropertyList",
    "PaperPropertyRecord",
    "PaperPropertyRequest",
    "PaperPropertyResult",
    "Property",
    "PropertyKey",
    "PropertyRecord",
    "RequestFailure",
    "STRONG_MATCHES",
    "Sample",
    "SampleExtractionRequest",
    "SampleExtractor",
    "SampleList",
    "SamplePropertyExtractor",
    "SamplePropertyList",
    "SamplePropertyRequest",
    "SampleRef",
]
