"""Render the property extraction prompts from templates and ``PROPERTY_CONFIG``.

Templates are Markdown files with ``[[SLOT]]`` placeholders, filled from the
property's configuration. The user templates also contain request placeholders
such as ``{paper_uid}``; these are filled per request by :func:`fill_user_prompt`,
which replaces only the known names, so literal JSON braces in the templates
are left untouched.

Ported from ``property_exploration/property_template_prompt.ipynb`` (cell pt-08).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .property_config import PROPERTY_CONFIG, VALUE_TYPES


PromptMode = Literal["per_paper", "with_context"]
PROMPT_MODES: tuple[str, ...] = ("per_paper", "with_context")

DEFAULT_PROPERTY_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "property"

# Request placeholders allowed in each mode's user template.
USER_FIELDS: dict[str, tuple[str, ...]] = {
    "per_paper": ("paper_uid", "target_sample_ids", "sample_list", "markdown_full_text"),
    "with_context": ("paper_uid", "sample_id", "sample_record", "other_samples", "markdown_full_text"),
}

_SLOT = re.compile(r"\[\[[A-Z_]+\]\]")


@dataclass(frozen=True, slots=True)
class PromptTemplates:
    """The raw templates of one mode plus the shared blocks."""

    system: str
    user: str
    conditions_field: str
    description_field: str


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def load_templates(mode: str, prompt_dir: str | Path = DEFAULT_PROPERTY_PROMPT_DIR) -> PromptTemplates:
    """Load the system and user templates of ``mode`` and the shared blocks."""
    if mode not in PROMPT_MODES:
        raise ValueError(f"mode must be one of {PROMPT_MODES}, got {mode!r}")
    root = Path(prompt_dir).expanduser().resolve()
    return PromptTemplates(
        system=_read(root / mode / "system_template.md"),
        user=_read(root / mode / "user_template.md"),
        conditions_field=_read(root / "shared" / "conditions_field.md"),
        description_field=_read(root / "shared" / "description_field.md"),
    )


# ---- pieces built from the config ----------------------------------------------------------

def _code_list(items, last="or"):
    items = [f"`{i}`" for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f", {last} " + items[-1]


def _method_list(cfg):
    names = [m for m in cfg["methods"] if m != "unspecified"]
    parts = [f"`{m}`" + (f" ({cfg['method_help'][m]})" if m in cfg["method_help"] else "") for m in names]
    text = ", ".join(parts)
    if "unspecified" in cfg["methods"]:
        text += ", or `unspecified` (if not stated)"
    return text


def _subtype_field(cfg):
    lines = [f"- **subtype:** {cfg['subtype_help']}. Use only if the paper states it; never infer it."]
    lines += [f"  - {g}" for g in cfg["subtype_glossary"]]
    groups = {}
    for method, subtypes in cfg["methods"].items():
        groups.setdefault(tuple(subtypes) if subtypes else None, []).append(method)
    single = len(groups) == 1
    for subtypes, methods in groups.items():
        who = ("Any method" if single else " or ".join(methods) if len(methods) <= 2
               else ", ".join(methods[:-1]) + ", or " + methods[-1])
        lines.append(f"  - {who}: " + ("`null`." if subtypes is None else _code_list(subtypes) + "."))
    lines.append("  - A stated subtype not in these lists → `unspecified`, with the detail in `description`.")
    return "\n".join(lines)


def _condition_names(cfg, with_help):
    parts = []
    for name, hint in cfg["conditions"]:
        parts.append(f"`{name}`" + (f" ({hint})" if with_help and hint else ""))
    return ", ".join(parts[:-1]) + ", or " + parts[-1]


def _exclude(cfg):
    items = cfg["exclude"]
    return items[0] if len(items) == 1 else "; ".join(items[:-1]) + "; or " + items[-1]


def _value_table(cfg):
    rows = ["| value_type | Example | value |", "|---|---|---|"]
    for vtype in VALUE_TYPES[:-1]:
        example, value = cfg["value_examples"][vtype]
        rows.append(f"| `{vtype}` | `{example}` | {value} |")
    rows.append(f"| `not_observed` | `{cfg['not_observed_example']}` | `null` |")
    return "\n".join(rows)


def _fill(text, values):
    for key, value in values.items():
        text = text.replace(f"[[{key}]]", value)
    return text


def _check_no_slots(text: str, where: str) -> None:
    left = _SLOT.findall(text)
    if left:
        raise ValueError(f"{where}: unfilled placeholder(s) {sorted(set(left))}")


# ---- public rendering ----------------------------------------------------------------------

def render_system_prompt(prop: str, templates: PromptTemplates) -> str:
    """Return the system prompt of ``templates`` filled for property ``prop``."""
    cfg = PROPERTY_CONFIG[prop]
    blocks = {
        "VALUE_TABLE": _value_table(cfg),
        "CONDITIONS_FIELD": templates.conditions_field,
        "DESCRIPTION_FIELD": templates.description_field,
    }
    values = {
        "LABEL": cfg["label"], "PLAIN": cfg["plain"], "SHORT": cfg["short"], "PROPERTY": prop,
        "METHOD_LIST": _method_list(cfg),
        "SUBTYPE_FIELD": _subtype_field(cfg),
        "SEPARATE_RECORDS": cfg["separate_records"],
        "EXCLUDE": _exclude(cfg),
        "TREND_EXAMPLE": cfg["trend_example"],
        "NOT_OBSERVED_EXAMPLE": cfg["not_observed_example"],
        "CONDITION_NAMES_LONG": _condition_names(cfg, with_help=True),
        "CONDITION_NAMES_SHORT": _condition_names(cfg, with_help=False),
        "CALC_EXAMPLES": cfg["calc_examples"],
        "DESCRIPTION_BULLETS": "\n".join(f"  - {b}" for b in cfg["description_bullets"]),
        "DESCRIPTION_NONE": cfg["description_none"],
        "DESCRIPTION_EXAMPLE": cfg["description_example"],
        "VALUE_RAW_EXAMPLES": cfg["value_raw_examples"],
        "VALUE_NOTE": cfg["value_note"],
        "UNIT_HELP": cfg["unit_help"],
        "COND_NUMBER_EXAMPLES": cfg["condition_examples"]["number"],
        "COND_TEXT_EXAMPLES": cfg["condition_examples"]["text"],
        "COND_UNIT_EXAMPLES": cfg["condition_examples"]["units"],
    }
    text = _fill(templates.system, blocks)  # insert the shared blocks first, then fill everything
    text = _fill(text, values)
    text = re.sub(r"\n{3,}", "\n\n", text)  # an empty block (e.g., no value note) leaves no gap
    text = text.strip() + "\n"
    _check_no_slots(text, f"{prop} system prompt")
    return text


def render_user_template(prop: str, templates: PromptTemplates) -> str:
    """Return the user template filled for ``prop``; request placeholders stay."""
    text = _fill(templates.user, {"PLAIN": PROPERTY_CONFIG[prop]["plain"]})
    _check_no_slots(text, f"{prop} user template")
    return text


def fill_user_prompt(template: str, mode: str, values: dict[str, str]) -> str:
    """Fill the request placeholders (``{name}``) of a rendered user template."""
    fields = USER_FIELDS[mode]
    missing = [f for f in fields if f not in values]
    if missing:
        raise ValueError(f"Missing user prompt value(s) for {mode}: {missing}")
    pattern = re.compile(r"\{(" + "|".join(fields) + r")\}")
    return pattern.sub(lambda m: values[m.group(1)], template)


__all__ = [
    "DEFAULT_PROPERTY_PROMPT_DIR",
    "PROMPT_MODES",
    "PromptMode",
    "PromptTemplates",
    "USER_FIELDS",
    "fill_user_prompt",
    "load_templates",
    "render_system_prompt",
    "render_user_template",
]
