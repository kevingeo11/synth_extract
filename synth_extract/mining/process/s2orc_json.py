"""Convert S2ORC JSON articles to Markdown."""

from __future__ import annotations

import json
from pathlib import Path


def _output_path_for_json(
    json_path: Path,
    output_path: str | Path | None,
) -> Path:
    """Resolve a directory or explicit Markdown output path."""
    if output_path is None:
        return json_path.with_suffix(".md")

    requested_path = Path(output_path)
    if requested_path.suffix.lower() == ".md":
        return requested_path
    return requested_path / f"{json_path.stem}.md"


def _required_text(article: dict, field: str, json_path: Path) -> str:
    """Read and validate a required text field from an S2ORC article."""
    value = article.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Expected a non-empty string field {field!r} in {json_path}"
        )
    return value.strip()


def _render_s2orc_json(json_path: Path) -> str:
    """Read one S2ORC JSON document and return Markdown text."""
    with json_path.open(encoding="utf-8") as file:
        article = json.load(file)

    if not isinstance(article, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")

    title = _required_text(article, "title", json_path)
    abstract = _required_text(article, "abstract", json_path)
    body = _required_text(article, "body", json_path)

    return (
        f"# {title}\n\n"
        f"## Abstract\n\n{abstract}\n\n"
        f"## Body\n\n{body}\n"
    )


def s2orc_json_to_markdown(
    json_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert an S2ORC JSON article to a Markdown file.

    ``output_path`` may be an output directory or an explicit ``.md`` file.
    When omitted, the Markdown file is written beside the JSON file with the
    same stem. The generated Markdown path is returned.
    """
    input_path = Path(json_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    if input_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON file, received: {input_path}")

    markdown_path = _output_path_for_json(input_path, output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_s2orc_json(input_path), encoding="utf-8")
    return markdown_path


json_to_markdown = s2orc_json_to_markdown


__all__ = ["s2orc_json_to_markdown", "json_to_markdown"]
