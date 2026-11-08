"""Property configurations for the property extraction agents.

One entry per property in ``PROPERTY_CONFIG``. The key is the value of the
``property`` field in every extracted record. The entries fill the ``[[...]]``
slots of the prompt templates and set the enums of the strict output schemas.

Ported from ``property_exploration/property_template_prompt.ipynb`` (cell pt-04).
"""

from __future__ import annotations

from typing import Literal, get_args

# ---- shared pieces --------------------------------------------------------------------------
TEMPERATURE = {
    "units": ["°C", "K", "°F", "unspecified"],
    "unit_help": "`°C`, `K`, or `°F`, as reported; `unspecified` if no unit is stated. Never convert.",
    "value_raw_examples": '`"152.3 ± 1.2 °C"`, `"120–125 °C"`, `"> 200 °C"`',
    "value_examples": {  # value_type -> (example as written, value)
        "exact": ("152 °C", "152"), "mean_sd": ("152 ± 2 °C", "152"), "approximate": ("~150 °C", "150"),
        "range": ("120–125 °C", "125 (upper end)"), "lower_bound": ("> 200 °C", "200"),
        "upper_bound": ("< −50 °C", "−50"),
    },
    "value_note": "",
    "condition_examples": {"number": "`10`", "text": '`"N2"`, `"second_heating"`',
                           "units": '`"K/min"`, `"Hz"`, `"mL/min"`, `"°C"`'},
}

MOLECULAR_WEIGHT = {
    "units": ["g/mol", "kg/mol", "Da", "kDa", "unspecified"],
    "unit_help": "`g/mol`, `kg/mol`, `Da`, or `kDa`, as reported; `unspecified` if no unit is stated. Never convert between units.",
    "value_raw_examples": '`"12 400 g/mol"`, `"3.5 × 10^4 g/mol"`, `"25.1 ± 0.8 kg/mol"`',
    "value_examples": {
        "exact": ("12 400 g/mol", "12400"), "mean_sd": ("25.1 ± 0.8 kg/mol", "25.1"),
        "approximate": ("~5000 g/mol", "5000"), "range": ("10–12 kg/mol", "12 (upper end)"),
        "lower_bound": ("> 100 kg/mol", "100"), "upper_bound": ("< 2000 g/mol", "2000"),
    },
    "value_note": ("  A power-of-ten multiplier written with the number or in the column header is applied to `value` "
                   "(e.g., `3.5 × 10^4 g/mol` → 35000 g/mol; `12.4` in a column headed `Mn (×10³ g/mol)` → 12400 g/mol). "
                   "`value_raw` keeps the number as written. Units themselves are never converted."),
    "condition_examples": {"number": "`35`, `1.0`", "text": '`"THF"`, `"refractive index"`',
                           "units": '`"°C"`, `"mL/min"`, `"mg/mL"`'},
}

GPC_SUBTYPES = ["polystyrene_standards", "other_standards", "absolute", "unspecified"]
GPC_GLOSSARY = [
    "`polystyrene_standards`: relative to polystyrene calibration standards.",
    "`other_standards`: relative to any other calibration standards (e.g., PMMA, PEG/PEO, pullulan); name the standard in `description`.",
    "`absolute`: not relative to standards, from a light-scattering detector (e.g., MALS) or universal calibration with a viscometer detector; name which in `description`.",
]
MW_CONDITIONS = [
    ("solvent", "eluent or NMR solvent, e.g., THF, DMF with LiBr, CDCl3"), ("temperature", None),
    ("flow_rate", None), ("concentration", None), ("detector", "e.g., refractive index, MALS, UV"),
    ("column", None), ("other", None),
]
MW_EXCLUDE = [
    "theoretical or target values calculated from the monomer-to-initiator ratio or conversion (e.g., Mn,theo, Mn calcd)",
    "nominal or supplier values of purchased materials",
    "values of precursors or macroinitiators that are not the sample itself",
    "viscosity-average molecular weight (Mv) or peak molecular weight (Mp)",
]
MW_DESCRIPTION = [
    "what the value is: which polymer, block, or fraction it refers to, and the state of the material (e.g., crude product before precipitation, after purification, after degradation);",
    "the method and how the value was obtained (e.g., GPC against polystyrene standards; 1H NMR end-group analysis; MALDI-TOF);",
    "eluent or solvent, column temperature, flow rate, concentration, detector, columns, and instrument;",
    "any other measurement detail not captured in the fields.",
]

# ---- one entry per property -------------------------------------------------------------------
TG_SUBTYPES_DSC = ["onset", "midpoint", "inflection", "unspecified"]
TG_SUBTYPES_DMA = ["tan_delta_peak", "loss_modulus_peak", "storage_modulus_onset", "unspecified"]
TD_SUBTYPES_TGA = ["onset", "mass_loss_5", "mass_loss_10", "mass_loss_20", "mass_loss_50", "mass_loss_other",
                   "max_rate", "endset", "range", "unspecified"]
TD_SUBTYPES_DSC = ["onset", "peak", "endset", "unspecified"]  # DSC measures heat flow, not mass
TM_SUBTYPES = ["peak", "onset", "endset", "unspecified"]

PROPERTY_CONFIG = {
    "glass_transition_temperature": {
        **TEMPERATURE,
        "presence_label": "glass transition temperature",
        "label": "glass transition temperature (Tg)",
        "plain": "glass transition temperature",
        "short": "Tg",
        "methods": {
            "DSC": TG_SUBTYPES_DSC, "DTA": TG_SUBTYPES_DSC, "DMA": TG_SUBTYPES_DMA,
            "TMA": None, "dilatometry": None, "dielectric": None, "other": None, "unspecified": None,
        },
        "method_help": {},
        "subtype_help": "which feature of the curve was taken as Tg",
        "subtype_glossary": [],
        "separate_records": "different methods, subtypes, heating rates, scans, or frequencies",
        "exclude": ["melting, crystallization, or cold-crystallization temperatures (Tm, Tc, Tcc)"],
        "trend_example": "\"Tg increased with filler content\"",
        "not_observed_example": "no Tg detected",
        "conditions": [
            ("heating_rate", None), ("cooling_rate", None),
            ("scan", "e.g., `first_heating`, `second_heating`, `cooling`"),
            ("frequency", None), ("atmosphere", None), ("temperature_range", None),
            ("thermal_history", "treatment of the specimen just before measurement, e.g., quenching"),
            ("specimen_form", "e.g., film, powder"), ("other", None),
        ],
        "calc_examples": "Fox, Gordon–Taylor, simulation",
        "description_bullets": [
            "what the value is: which transition, phase, segment, or block (e.g., soft-segment Tg, second Tg of a blend);",
            "the method and how Tg was taken from the curve (e.g., midpoint of the heat-capacity step, tan δ peak);",
            "the scan and thermal history (e.g., second heating after cooling at 10 °C/min; annealing or quenching before the test);",
            "heating or cooling rate, frequency, strain or amplitude, temperature range, atmosphere, instrument, and specimen form, mass, or preparation for the test;",
            "any other measurement detail not captured in the fields.",
        ],
        "description_none": "Tg measured by DSC; no measurement conditions stated.",
        "description_example": "Tg of the soft polyether segment, taken as the midpoint of the heat-capacity step in the second DSC heating scan, recorded at 10 °C/min under nitrogen from −90 to 200 °C after cooling at 10 °C/min; about 5 mg of film in an aluminium pan.",
    },
    "melting_temperature": {
        **TEMPERATURE,
        "presence_label": "melting temperature",
        "label": "melting temperature (Tm)",
        "plain": "melting temperature",
        "short": "Tm",
        "methods": {
            "DSC": TM_SUBTYPES, "DTA": TM_SUBTYPES, "optical_microscopy": None,
            "other": None, "unspecified": None,
        },
        "method_help": {"optical_microscopy": "e.g., hot-stage polarized optical microscopy"},
        "subtype_help": "which feature of the melting endotherm was taken as Tm",
        "subtype_glossary": [],
        "separate_records": "different methods, subtypes, heating rates, scans, or melting peaks (e.g., double melting)",
        "exclude": ["glass transition, crystallization, cold-crystallization, or decomposition temperatures",
                    "melting enthalpies or crystallinity"],
        "trend_example": "\"Tm decreased with comonomer content\"",
        "not_observed_example": "no melting peak observed",
        "conditions": [
            ("heating_rate", None), ("cooling_rate", None),
            ("scan", "e.g., `first_heating`, `second_heating`"),
            ("atmosphere", None), ("temperature_range", None),
            ("thermal_history", "treatment of the specimen just before measurement, e.g., annealing, isothermal crystallization"),
            ("specimen_form", "e.g., film, powder"), ("other", None),
        ],
        "calc_examples": "Hoffman–Weeks extrapolation of the equilibrium melting temperature, Gibbs–Thomson, simulation",
        "description_bullets": [
            "what the value is: which crystalline phase, block, or melting peak (e.g., low-temperature peak of a double melting endotherm, PEG block);",
            "the method and how Tm was taken from the curve (e.g., peak maximum of the melting endotherm, onset);",
            "the scan and thermal history (e.g., second heating after cooling at 10 °C/min; annealing or isothermal crystallization before the test);",
            "heating rate, temperature range, atmosphere, instrument, and specimen form, mass, or preparation for the test;",
            "any other measurement detail not captured in the fields.",
        ],
        "description_none": "Tm measured by DSC; no measurement conditions stated.",
        "description_example": "Melting temperature of the polyester hard segment, taken as the peak maximum of the melting endotherm in the second DSC heating scan, recorded at 10 °C/min under nitrogen from −50 to 200 °C after cooling at 10 °C/min; about 5 mg in an aluminium pan.",
    },
    "thermal_decomposition_temperature": {
        **TEMPERATURE,
        "presence_label": "thermal decomposition temperature",
        "label": "thermal decomposition temperature",
        "plain": "thermal decomposition temperature",
        "short": "decomposition temperature",
        # DTG is the derivative of the TGA curve, not a separate method: a DTG peak is TGA + max_rate.
        # TGA coupled with MS or FTIR gives the same mass curve, so it is TGA.
        "methods": {"TGA": TD_SUBTYPES_TGA, "DSC": TD_SUBTYPES_DSC,
                    "other": None, "unspecified": TD_SUBTYPES_TGA},
        "method_help": {"TGA": "including TGA coupled with MS or FTIR"},
        "subtype_help": "how the decomposition temperature is defined",
        "subtype_glossary": [
            "`onset`: initial or extrapolated onset of decomposition (e.g., IDT, Ti, Tonset); for DSC, onset of the decomposition peak.",
            "`mass_loss_5`, `mass_loss_10`, `mass_loss_20`, `mass_loss_50`: temperature at that mass loss in % (e.g., T5% or Td5 → `mass_loss_5`).",
            "`mass_loss_other`: temperature at a stated mass loss not in the list (e.g., T1%, T30%; state the percentage in `description`).",
            "`max_rate`: temperature of the maximum mass-loss rate, i.e., the peak of the derivative (DTG) curve (e.g., Tmax, MDT). A reported DTG peak → method `TGA`, subtype `max_rate`.",
            "`peak`: peak temperature of the decomposition exotherm or endotherm in DSC.",
            "`endset`: end of decomposition (e.g., FDT, Tend).",
            "`range`: a stated temperature interval of a decomposition step.",
        ],
        "separate_records": "different methods, subtypes, decomposition steps, heating rates, or atmospheres",
        "exclude": [
            "residue or char yield, or weight loss in %",
            "activation energies or other kinetic parameters",
            "heat deflection or Vicat temperatures",
            "melting or glass transition temperatures (note that \"TG\" often means thermogravimetry)",
        ],
        "trend_example": "\"thermal stability increased with filler content\"",
        "not_observed_example": "no decomposition detected",
        "conditions": [
            ("heating_rate", None), ("atmosphere", "e.g., `N2`, `air`"), ("gas_flow_rate", None),
            ("temperature_range", None), ("sample_mass", None),
            ("specimen_form", "e.g., film, powder"), ("other", None),
        ],
        "calc_examples": "extrapolation to zero heating rate, molecular simulation",
        "description_bullets": [
            "what the value is: which decomposition step it belongs to (e.g., first, second, or main step) and what decomposes, if stated;",
            "the method and how the temperature is defined, with the paper's own label (e.g., temperature at 5 % mass loss, DTG peak, extrapolated onset, IDT);",
            "the atmosphere and gas flow rate (e.g., nitrogen at 50 mL/min, air);",
            "heating rate, temperature range, instrument, crucible, and specimen form, mass, or preparation for the test;",
            "any other measurement detail not captured in the fields.",
        ],
        "description_none": "Decomposition temperature measured by TGA; no measurement conditions stated.",
        "description_example": "Temperature at 5 % mass loss in the main decomposition step, from TGA under nitrogen (50 mL/min) at 10 °C/min from 30 to 600 °C; about 10 mg of powder in an alumina crucible.",
    },
    "number_average_molecular_weight": {
        **MOLECULAR_WEIGHT,
        "presence_label": "number-average molecular weight",
        "label": "number-average molecular weight (Mn)",
        "plain": "number-average molecular weight",
        "short": "Mn",
        "methods": {"GPC": GPC_SUBTYPES, "NMR": None, "other": None, "unspecified": None},
        "method_help": {"GPC": "GPC or SEC", "NMR": "end-group analysis; counts as `measured`"},
        "subtype_help": "for GPC, the calibration or detection used to obtain molar mass",
        "subtype_glossary": GPC_GLOSSARY,
        "separate_records": "different methods, calibrations, eluents, or material states (e.g., crude and purified)",
        "exclude": MW_EXCLUDE + ["weight-average molecular weight (Mw) and dispersity"],
        "trend_example": "\"Mn increased with reaction time\"",
        "not_observed_example": "not measurable (insoluble)",
        "conditions": MW_CONDITIONS,
        "calc_examples": "kinetic modelling, molecular simulation",
        "description_bullets": MW_DESCRIPTION,
        "description_none": "Mn measured by GPC; no measurement conditions stated.",
        "description_example": "Number-average molecular weight of the purified copolymer from GPC in THF at 35 °C and 1.0 mL/min with a refractive-index detector, against polystyrene standards; about 2 mg/mL, filtered through a 0.45 μm PTFE filter.",
    },
    "weight_average_molecular_weight": {
        **MOLECULAR_WEIGHT,
        "presence_label": "weight-average molecular weight",
        "label": "weight-average molecular weight (Mw)",
        "plain": "weight-average molecular weight",
        "short": "Mw",
        "methods": {"GPC": GPC_SUBTYPES, "other": None, "unspecified": None},
        "method_help": {"GPC": "GPC or SEC"},
        "subtype_help": "for GPC, the calibration or detection used to obtain molar mass",
        "subtype_glossary": GPC_GLOSSARY,
        "separate_records": "different methods, calibrations, eluents, or material states (e.g., crude and purified)",
        "exclude": MW_EXCLUDE + ["number-average molecular weight (Mn) and dispersity",
                                 "Mw values you would have to compute from Mn and dispersity"],
        "trend_example": "\"Mw increased with reaction time\"",
        "not_observed_example": "not measurable (insoluble)",
        "conditions": MW_CONDITIONS,
        "calc_examples": "kinetic modelling, molecular simulation",
        "description_bullets": MW_DESCRIPTION,
        "description_none": "Mw measured by GPC; no measurement conditions stated.",
        "description_example": "Weight-average molecular weight of the resin from static light scattering (Zimm plot) in distilled water, filtered through a 0.45 μm nylon filter, at six concentrations from 0.05 to 0.30 mg/mL and scattering angles from 30 to 135°.",
    },
    "dispersity": {
        **MOLECULAR_WEIGHT,
        "units": ["none"],
        "unit_help": "always `none` (dispersity has no unit).",
        "value_raw_examples": '`"1.18"`, `"Đ = 1.25 ± 0.03"`, `"< 1.1"`',
        "value_examples": {
            "exact": ("1.18", "1.18"), "mean_sd": ("1.25 ± 0.03", "1.25"), "approximate": ("~1.2", "1.2"),
            "range": ("1.1–1.3", "1.3 (upper end)"), "lower_bound": ("> 2", "2"), "upper_bound": ("< 1.1", "1.1"),
        },
        "value_note": "",
        "presence_label": "dispersity",
        "label": "dispersity (Đ, also called PDI or Mw/Mn)",
        "plain": "dispersity (PDI)",
        "short": "dispersity",
        "methods": {"GPC": GPC_SUBTYPES, "other": None, "unspecified": None},
        "method_help": {"GPC": "GPC or SEC"},
        "subtype_help": "for GPC, the calibration or detection used",
        "subtype_glossary": GPC_GLOSSARY,
        "separate_records": "different methods, calibrations, eluents, or material states (e.g., crude and purified)",
        "exclude": MW_EXCLUDE[1:] + ["dispersity values you would have to compute from Mn and Mw",
                                     "the dispersity of particle sizes (particle size distribution)"],
        "trend_example": "\"the distribution broadened with conversion\"",
        "not_observed_example": "not measurable (insoluble)",
        "conditions": MW_CONDITIONS,
        "calc_examples": "kinetic modelling, molecular simulation",
        "description_bullets": MW_DESCRIPTION,
        "description_none": "Dispersity measured by GPC; no measurement conditions stated.",
        "description_example": "Dispersity of the crude copolymer from GPC in THF at 40 °C and 1.0 mL/min with a refractive-index detector, against polystyrene standards.",
    },
}
VALUE_TYPES = ["exact", "mean_sd", "approximate", "range", "lower_bound", "upper_bound", "not_observed"]
CONDITION_TYPES = ["exact", "mean_sd", "approximate", "range", "lower_bound", "upper_bound"]


MATCH_BASES = ["same_entry", "same_description", "name_only", "none"]
# Prompt 2 links that count as a sample assignment; name_only is reported but never kept.
STRONG_MATCHES = frozenset({"same_entry", "same_description"})

PropertyKey = Literal[
    "glass_transition_temperature",
    "melting_temperature",
    "thermal_decomposition_temperature",
    "number_average_molecular_weight",
    "weight_average_molecular_weight",
    "dispersity",
]
ValueType = Literal["exact", "mean_sd", "approximate", "range", "lower_bound", "upper_bound", "not_observed"]
ConditionType = Literal["exact", "mean_sd", "approximate", "range", "lower_bound", "upper_bound"]
MatchBasis = Literal["same_entry", "same_description", "name_only", "none"]
Origin = Literal["measured", "calculated"]

assert set(get_args(PropertyKey)) == set(PROPERTY_CONFIG), "PropertyKey must list every configured property"
assert list(get_args(ValueType)) == VALUE_TYPES and list(get_args(ConditionType)) == CONDITION_TYPES
assert list(get_args(MatchBasis)) == MATCH_BASES

__all__ = [
    "CONDITION_TYPES",
    "ConditionType",
    "MATCH_BASES",
    "MatchBasis",
    "Origin",
    "PROPERTY_CONFIG",
    "PropertyKey",
    "STRONG_MATCHES",
    "VALUE_TYPES",
    "ValueType",
]
