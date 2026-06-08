from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert metadata values to JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _output_folder_for_pdf(pdf_path: Path, output_markdown_path: Path) -> Path:
    output_base_path = (
        output_markdown_path.parent
        if output_markdown_path.suffix
        else output_markdown_path
    )
    return output_base_path / pdf_path.stem


def pdf_to_markdown(
    pdf_path: str,
    output_markdown_path: str,
) -> None:
    """
    Convert a PDF to Markdown using Marker.

    Parameters
    ----------
    pdf_path : str
        Path to input PDF.
    output_markdown_path : str
        Path where markdown file will be saved.
    """

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    pdf_path = Path(pdf_path)
    output_markdown_path = Path(output_markdown_path)
    output_folder = _output_folder_for_pdf(pdf_path, output_markdown_path)

    converter = PdfConverter(
        artifact_dict=create_model_dict()
    )

    rendered = converter(str(pdf_path))

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_folder / f"{pdf_path.stem}.md").write_text(
        rendered.markdown,
        encoding="utf-8",
    )

    for image_name, image in rendered.images.items():
        image.save(output_folder / image_name)

    (output_folder / f"{pdf_path.stem}_meta.json").write_text(
        json.dumps(_json_safe(rendered.metadata), indent=2),
        encoding="utf-8",
    )


def pdf_to_markdown_using_cli(
    pdf_path: str,
    output_markdown_path: str,
) -> None:
    """Convert a PDF to Markdown by calling Marker CLI with subprocess."""
    pdf_path = Path(pdf_path)
    output_markdown_path = Path(output_markdown_path)
    output_folder = _output_folder_for_pdf(pdf_path, output_markdown_path)
    output_base_path = output_folder.parent

    marker_cli = shutil.which("marker_single")
    if marker_cli is None:
        raise FileNotFoundError(
            "Could not find the Marker CLI command 'marker_single'. "
            "Install marker-pdf or make sure marker_single is on PATH."
        )

    output_base_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        marker_cli,
        str(pdf_path),
        "--output_dir",
        str(output_base_path),
        "--output_format",
        "markdown",
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    if not output_folder.exists():
        raise FileNotFoundError(
            "Marker CLI completed, but the expected output folder was not created: "
            f"{output_folder}"
        )


def load_markdown(path: str) -> str:
    """Load markdown content from a file path."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def normalize_markdown_text(text: str) -> str:
    """Normalize markdown text for LLM input.

    This keeps scientific content intact while removing excessive whitespace.
    """
    pass


def chunk_markdown(text: str, max_chars: int = 8000) -> list[str]:
    """If we need to chunk markdown content, we can split on headers to keep scientific sections together."""
    pass
