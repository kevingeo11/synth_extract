"""Small sample-centric schema for initial polymer full-text extraction trials.

The extractor should emit one ``PaperExtraction`` JSON object per paper. Values are
kept close to the language used in the paper; normalization and cross-paper entity
resolution are intentionally out of scope for this first experiment.
"""

from pydantic import BaseModel, ConfigDict, Field


class ExtractionModel(BaseModel):
    """Base model shared by all extraction objects."""

    model_config = ConfigDict(extra="forbid")


class Identity(ExtractionModel):
    """Names and identifying text for a sample or component."""

    name: str | None = Field(default=None, description="Primary name as reported in the paper.")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names used for the same entity in this paper."
    )
    codes: list[str] = Field(
        default_factory=list, description="Reported sample, product, grade, or formulation codes."
    )
    description: str = Field(
        description="Free-text identity description preserving useful material details."
    )


class Component(ExtractionModel):
    """One material or chemical used to make a sample."""

    component_id: str = Field(description="Extractor-generated local ID, e.g. C01.")
    identity: Identity = Field(description="Identity of the component as reported in the paper.")
    role: str | None = Field(
        default=None, description="Reported or plainly stated role, e.g. monomer, matrix, filler, or solvent."
    )
    amount: str | None = Field(
        default=None, description="Amount or proportion largely as reported, including its unit or basis."
    )
    description: str = Field(
        description="Free-text component details not captured by identity, role, or amount."
    )
    evidence: list[str] = Field(
        default_factory=list, description="Short verbatim spans supporting this component record."
    )


class SynthesisStep(ExtractionModel):
    """One ordered step in a sample's synthesis or fabrication path."""

    step_id: str = Field(description="Extractor-generated local ID, e.g. ST01.")
    action: str | None = Field(
        default=None, description="Short action label, e.g. polymerize, mix, cast, cure, wash, or dry."
    )
    inputs: list[str] = Field(
        default_factory=list, description="Local component/sample IDs or reported input names used in this step."
    )
    conditions: str | None = Field(
        default=None, description="Temperature, time, atmosphere, equipment, and other conditions as reported."
    )
    description: str = Field(
        description="Free-text account of the step, including details not captured elsewhere."
    )
    evidence: list[str] = Field(
        default_factory=list, description="Short verbatim spans supporting this synthesis step."
    )


class Synthesis(ExtractionModel):
    """Overall synthesis or fabrication path for a sample."""

    description: str = Field(
        description="Free-text summary of how the sample was synthesized or fabricated."
    )
    steps: list[SynthesisStep] = Field(
        default_factory=list, description="Ordered synthesis or fabrication steps."
    )


class Property(ExtractionModel):
    """One reported property, characterization result, or synthesis outcome."""

    property_id: str = Field(description="Extractor-generated local ID, e.g. PR01.")
    name: str = Field(description="Property name largely as reported in the paper.")
    value: str | None = Field(
        default=None, description="Reported value, range, or qualitative result without forced normalization."
    )
    unit: str | None = Field(default=None, description="Unit as reported, when separable from the value.")
    conditions: str | None = Field(
        default=None, description="Measurement or reporting conditions largely as stated in the paper."
    )
    method: str | None = Field(
        default=None, description="Measurement, characterization, or calculation method as reported."
    )
    description: str = Field(
        description="Free-text context needed to interpret or identify this property."
    )
    evidence: list[str] = Field(
        default_factory=list, description="Short verbatim spans supporting this property record."
    )


class Characterization(ExtractionModel):
    """A reported descriptor or characterization of the realized sample."""

    characterization_id: str = Field(
        description="Extractor-generated local ID, e.g. CH01."
    )
    name: str = Field(
        description="Characteristic name largely as reported, e.g. Mn, dispersity, thickness, or architecture."
    )
    value: str | None = Field(
        default=None,
        description="Reported quantitative or qualitative value without forced normalization."
    )
    unit: str | None = Field(
        default=None,
        description="Unit as reported, when applicable."
    )
    method: str | None = Field(
        default=None,
        description="Method used to determine the characteristic, when reported."
    )
    conditions: str | None = Field(
        default=None,
        description="Relevant characterization conditions as reported."
    )
    description: str = Field(
        description="Free-text description of the sample characteristic and its context."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Short verbatim spans supporting this characterization."
    )


class Sample(ExtractionModel):
    """Central record for one sample described in a paper."""

    sample_id: str = Field(description="Extractor-generated local ID, e.g. S01.")
    identity: Identity = Field(description="Names, codes, and identifying details for the sample.")
    components: list[Component] = Field(
        default_factory=list, description="Components used to define or prepare this sample."
    )
    synthesis: Synthesis | None = Field(
        default=None, description="Synthesis or fabrication path, or null when the paper does not provide one."
    )
    characterization: list[Characterization] = Field(
        default_factory=list
    )
    properties: list[Property] = Field(
        default_factory=list, description="Properties and outcomes reported for this sample."
    )
    description: str = Field(
        description="Free-text sample summary preserving important information."
    )
    evidence: list[str] = Field(
        default_factory=list, description="Short verbatim spans supporting the sample identity and scope."
    )


class PaperExtraction(ExtractionModel):
    """One sample-centric extraction object for one paper."""

    paper_uid: str = Field(description="Local identifier for the source paper.")
    has_samples: bool = Field(
        description="False if the paper describes no prepared material (e.g. theory, simulation, review); "
        "samples is then empty."
    )
    samples: list[Sample] = Field(
        default_factory=list, description="Samples described and characterized in the paper."
    )


def example_extraction() -> PaperExtraction:
    """Return a small valid example suitable for inspecting the JSON shape."""

    polymer = Component(
        component_id="C01",
        identity=Identity(
            name="poly(vinyl alcohol)",
            aliases=["PVA"],
            codes=[],
            description="PVA used as the hydrogel precursor.",
        ),
        role="polymer precursor",
        amount="8.7 wt% in DMSO",
        description="The paper reports a high degree of saponification.",
        evidence=["PVA concentration of 8.7 wt. %"],
    )

    sample = Sample(
        sample_id="S01",
        identity=Identity(
            name="modified PVA hydrogel",
            aliases=["mPVA hydrogel"],
            codes=[],
            description="Gamma-crosslinked PVA modified with glycidyl methacrylate.",
        ),
        components=[polymer],
        synthesis=Synthesis(
            description="PVA was modified and then radiation-crosslinked in water.",
            steps=[
                SynthesisStep(
                    step_id="ST01",
                    action="modify",
                    inputs=["C01", "glycidyl methacrylate"],
                    conditions="80 °C for 2.5 h in DMSO",
                    description="PVA was reacted with glycidyl methacrylate before purification.",
                    evidence=["The synthesis of mPVA was carried out in DMSO"],
                )
            ],
        ),
        properties=[
            Property(
                property_id="PR01",
                name="gel fraction",
                value="60",
                unit="%",
                conditions="after gamma irradiation",
                method="gravimetric after Soxhlet extraction",
                description="Maximum reported gel fraction for the crosslinked hydrogel.",
                evidence=["gel fraction ... reaches 60%"],
            )
        ],
        description="Prepared crosslinked hydrogel sample with reported swelling and mechanical properties.",
        evidence=["PVA hydrogels were prepared in two stages using gamma radiation"],
    )

    return PaperExtraction(paper_uid="ID000601427", has_samples=True, samples=[sample])


if __name__ == "__main__":
    print(example_extraction().model_dump_json(indent=2))
