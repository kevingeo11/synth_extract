"""Schemas for the property extraction agents.

Two layers:

* Strict provider-facing JSON schemas (:func:`make_response_format` for the
  per-sample prompt, :func:`make_paper_response_format` for the per-paper
  prompt). Their enums come from ``PROPERTY_CONFIG``, so they differ per
  property. Ported from the notebook (cell pt-08).
* Pydantic models that parse the responses and carry the results. They hold
  the fields shared by all properties; property-specific rules the strict
  schema cannot express are checked by :func:`check_record`, which flags a
  record instead of dropping it.

:class:`Property` is the stored record: one property value of one sample, the
same for both prompts. :meth:`PaperPropertyResult.to_properties` maps an
agent result to these records.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, computed_field, field_validator, model_validator

from .property_config import (
    CONDITION_TYPES,
    MATCH_BASES,
    PROPERTY_CONFIG,
    VALUE_TYPES,
    ConditionType,
    MatchBasis,
    Origin,
    ValueType,
)
from .schemas import ExtractionModel


# ---- strict response formats (ported from the notebook) -----------------------------------

RECORD_FIELDS = ["property", "method", "subtype", "value_raw", "value", "value_type",
                 "unit", "conditions", "origin", "description"]
LINK_FIELDS = ["paper_label", "value_raw", "identifying_data", "match_basis", "sample_id"]


def _record_properties(prop: str) -> dict[str, Any]:
    cfg = PROPERTY_CONFIG[prop]
    subtypes = sorted({s for subs in cfg["methods"].values() if subs for s in subs})
    has_null = any(subs is None for subs in cfg["methods"].values())
    return {
        "property": {"type": "string", "enum": [prop]},
        "method": {"type": "string", "enum": list(cfg["methods"])},
        "subtype": ({"type": ["string", "null"], "enum": subtypes + [None]} if has_null
                    else {"type": "string", "enum": subtypes}),
        "value_raw": {"type": "string"},
        "value": {"type": ["number", "null"]},
        "value_type": {"type": "string", "enum": VALUE_TYPES},
        "unit": {"type": "string", "enum": cfg["units"]},
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "type", "value", "unit", "description"],
                "properties": {
                    "name": {"type": "string", "enum": [n for n, _ in cfg["conditions"]]},
                    "type": {"type": "string", "enum": CONDITION_TYPES},
                    "value": {"type": ["number", "string"]},
                    "unit": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                },
            },
        },
        "origin": {"type": "string", "enum": ["measured", "calculated"]},
        "description": {"type": "string"},
    }


def _wrap(name: str, item_properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["properties"],
                "properties": {"properties": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": required, "properties": item_properties,
                }}},
            },
        },
    }


def make_response_format(prop: str) -> dict[str, Any]:
    """Strict schema of the per-sample prompts (prompts 1 and 3)."""
    return _wrap(f"sample_{prop}", _record_properties(prop), RECORD_FIELDS)


def make_paper_response_format(prop: str, sample_ids: list[int]) -> dict[str, Any]:
    """Strict schema of the per-paper prompt (prompt 2).

    The linking fields come first, so the model writes the evidence before it
    picks the sample. ``sample_id`` is limited to the paper's sample IDs.
    """
    linking = {
        "paper_label": {"type": "string"},
        "value_raw": {"type": "string"},
        "identifying_data": {"type": "string"},
        "match_basis": {"type": "string", "enum": list(MATCH_BASES)},
        "sample_id": {"type": ["integer", "null"], "enum": [int(i) for i in sample_ids] + [None]},
    }
    record = _record_properties(prop)
    properties = {**linking, **{k: v for k, v in record.items() if k != "value_raw"}}
    required = LINK_FIELDS + [f for f in RECORD_FIELDS if f != "value_raw"]
    return _wrap(f"paper_{prop}", properties, required)


# ---- inputs --------------------------------------------------------------------------------

class SampleRef(ExtractionModel):
    """One row of the ``samples`` table, used only to identify samples."""

    sample_id: int
    name: str
    identifier: str = ""
    aliases: list[str] = Field(default_factory=list)
    description: str

    @field_validator("aliases", mode="before")
    @classmethod
    def parse_aliases(cls, value: Any) -> Any:
        """Accept the database's JSON text as well as a list."""
        if isinstance(value, str):
            value = json.loads(value) if value.strip() else []
        return value

    def record_json(self) -> str:
        """The target record as shown to the per-sample prompt (no sample_id)."""
        return json.dumps(
            {"name": self.name, "identifier": self.identifier,
             "aliases": self.aliases, "description": self.description},
            ensure_ascii=False,
        )

    def list_json(self) -> str:
        """One line of a sample list (with sample_id)."""
        return json.dumps(
            {"sample_id": self.sample_id, "name": self.name, "identifier": self.identifier,
             "aliases": self.aliases, "description": self.description},
            ensure_ascii=False,
        )


def _require_text(value: str) -> str:
    if not value.strip():
        raise ValueError("must be a non-empty string")
    return value


def _unique_samples(samples: list[SampleRef]) -> list[SampleRef]:
    if not samples:
        raise ValueError("at least one sample is required")
    ids = [s.sample_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    return samples


class PaperContext(ExtractionModel):
    """Everything the agents need about one paper: its samples and full text."""

    paper_uid: str
    samples: list[SampleRef]
    markdown_full_text: str

    @field_validator("paper_uid", "markdown_full_text")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _require_text(value)

    @field_validator("samples")
    @classmethod
    def require_unique_samples(cls, value: list[SampleRef]) -> list[SampleRef]:
        return _unique_samples(value)


class SamplePropertyRequest(ExtractionModel):
    """One per-sample request (prompt 3): the target and the paper's other samples."""

    paper_uid: str
    target: SampleRef
    other_samples: list[SampleRef] = Field(default_factory=list)
    markdown_full_text: str

    @field_validator("paper_uid", "markdown_full_text")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _require_text(value)

    @model_validator(mode="after")
    def target_not_in_others(self) -> "SamplePropertyRequest":
        if any(s.sample_id == self.target.sample_id for s in self.other_samples):
            raise ValueError("the target must not appear in other_samples")
        return self


class PaperPropertyRequest(ExtractionModel):
    """One per-paper request (prompt 2): every sample, a batch of them as targets."""

    paper_uid: str
    samples: list[SampleRef]
    target_sample_ids: list[int]
    markdown_full_text: str

    @field_validator("paper_uid", "markdown_full_text")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _require_text(value)

    @field_validator("samples")
    @classmethod
    def require_unique_samples(cls, value: list[SampleRef]) -> list[SampleRef]:
        return _unique_samples(value)

    @model_validator(mode="after")
    def targets_in_samples(self) -> "PaperPropertyRequest":
        ids = {s.sample_id for s in self.samples}
        if not self.target_sample_ids:
            raise ValueError("at least one target sample is required")
        if len(set(self.target_sample_ids)) != len(self.target_sample_ids):
            raise ValueError("target_sample_ids must be unique")
        unknown = [i for i in self.target_sample_ids if i not in ids]
        if unknown:
            raise ValueError(f"target_sample_ids not in samples: {unknown}")
        return self

    @property
    def all_targets(self) -> bool:
        return set(self.target_sample_ids) == {s.sample_id for s in self.samples}


# ---- model outputs -------------------------------------------------------------------------

class Condition(ExtractionModel):
    """One measurement condition of a property record."""

    name: str
    type: ConditionType
    value: int | float | str
    unit: str | None = None
    description: str | None = None


class PropertyRecord(ExtractionModel):
    """One reported property value (the unified record; prompt 3 output item)."""

    property: str
    method: str
    subtype: str | None
    value_raw: str
    value: int | float | None
    value_type: ValueType
    unit: str
    conditions: list[Condition] = Field(default_factory=list)
    origin: Origin
    description: str


class PaperPropertyRecord(PropertyRecord):
    """A property record with prompt 2's link to a sample."""

    paper_label: str
    identifying_data: str
    match_basis: MatchBasis
    sample_id: int | None


class SamplePropertyList(ExtractionModel):
    """Response of one per-sample request.

    ``properties`` is required and every item must be a valid record: ``{}`` or
    one malformed item makes the whole response invalid (the request fails), so
    a broken response can never pass as "no values". ``[]`` is a valid answer.
    """

    properties: list[PropertyRecord]


class PaperPropertyList(ExtractionModel):
    """Response of one per-paper request (same strictness as :class:`SamplePropertyList`)."""

    properties: list[PaperPropertyRecord]


# ---- flags and checks the strict schema cannot express ------------------------------------

FLAG_NAME_ONLY = "name_only"                # prompt 2 linked the value only by a shared name
FLAG_SUBTYPE_MISMATCH = "subtype_mismatch"  # subtype not allowed for the method
FLAG_VALUE_MISMATCH = "value_mismatch"      # value null but not not_observed, or the reverse
FLAG_INVALID_FIELD = "invalid_field"        # property, method, unit, or condition name outside the config
FLAGS = (FLAG_NAME_ONLY, FLAG_SUBTYPE_MISMATCH, FLAG_VALUE_MISMATCH, FLAG_INVALID_FIELD)


def check_record(record: PropertyRecord, prop: str) -> list[str]:
    """Return the flag codes of one record; an empty list means no problem.

    The strict schema already limits each field to the property's enums. These
    checks cover cross-field rules, and repeat the enum checks for providers
    that do not enforce the schema.
    """
    cfg = PROPERTY_CONFIG[prop]
    flags: list[str] = []
    names = {n for n, _ in cfg["conditions"]}
    if (record.property != prop or record.method not in cfg["methods"] or record.unit not in cfg["units"]
            or any(c.name not in names for c in record.conditions)):
        flags.append(FLAG_INVALID_FIELD)
    if (record.value is None) != (record.value_type == "not_observed"):
        flags.append(FLAG_VALUE_MISMATCH)
    allowed = cfg["methods"].get(record.method, "unknown")
    if allowed != "unknown":
        if (allowed is None and record.subtype is not None) or (allowed is not None and record.subtype not in allowed):
            flags.append(FLAG_SUBTYPE_MISMATCH)
    return flags


# ---- results returned by the agents ---------------------------------------------------------

class ExtractedRecord(ExtractionModel):
    """One record as returned by an agent, with its sample and flag codes."""

    sample_id: int | None = Field(
        description="Prompt 3: the target; prompt 2: the linked sample, or null when the value matches no sample."
    )
    record: PaperPropertyRecord | PropertyRecord
    flags: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def linked(self) -> bool:
        """True when the record belongs to a sample and becomes a :class:`Property`."""
        return self.sample_id is not None


class RequestFailure(ExtractionModel):
    """A request of a paper that did not return a valid response."""

    sample_ids: list[int] = Field(description="Prompt 3: the target; prompt 2: the batch targets.")
    error_type: str
    message: str
    raw_response: str | None = None


class Property(ExtractionModel):
    """One reported property value of one sample: the stored record.

    The same for both prompts. Problems found during extraction are flag codes
    (see ``FLAGS``); ``clean`` is true when there are none.
    """

    sample_id: int
    paper_uid: str
    property: str
    method: str
    subtype: str | None
    value_raw: str
    value: int | float | None
    value_type: ValueType
    unit: str
    conditions: list[Condition] = Field(default_factory=list)
    origin: Origin
    description: str
    flags: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clean(self) -> bool:
        return not self.flags


class PaperPropertyResult(ExtractionModel):
    """All records of one property in one paper, from one prompt."""

    paper_uid: str
    property: str
    mode: str
    prompt_hash: str
    records: list[ExtractedRecord] = Field(default_factory=list)
    failures: list[RequestFailure] = Field(default_factory=list)
    requests: int = 0
    duplicates_dropped: int = Field(
        default=0, description="Exact duplicates of a record within one sample that were removed."
    )
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def complete(self) -> bool:
        """True when every request of the paper succeeded."""
        return not self.failures

    def to_properties(self) -> list[Property]:
        """Map the records that belong to a sample to stored :class:`Property` records.

        Prompt 2 values that match no listed sample are not included.
        """
        return [
            Property(
                sample_id=item.sample_id,
                paper_uid=self.paper_uid,
                **{field: getattr(item.record, field) for field in RECORD_FIELDS},
                flags=list(item.flags),
            )
            for item in self.records
            if item.linked
        ]


__all__ = [
    "Condition",
    "ExtractedRecord",
    "FLAGS",
    "FLAG_INVALID_FIELD",
    "FLAG_NAME_ONLY",
    "FLAG_SUBTYPE_MISMATCH",
    "FLAG_VALUE_MISMATCH",
    "LINK_FIELDS",
    "PaperContext",
    "PaperPropertyList",
    "PaperPropertyRecord",
    "PaperPropertyRequest",
    "PaperPropertyResult",
    "Property",
    "PropertyRecord",
    "RECORD_FIELDS",
    "RequestFailure",
    "SamplePropertyList",
    "SamplePropertyRequest",
    "SampleRef",
    "check_record",
    "make_paper_response_format",
    "make_response_format",
]
