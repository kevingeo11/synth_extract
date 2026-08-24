"""Convert PDF documents to Markdown with Marker."""

from __future__ import annotations

from pathlib import Path


def _markdown_output_path(
    pdf_path: Path,
    output_path: str | Path | None,
) -> Path:
    """Resolve a directory or explicit Markdown output path."""
    if output_path is None:
        return pdf_path.with_suffix(".md")

    requested_path = Path(output_path)
    if requested_path.suffix.lower() == ".md":
        return requested_path

    return requested_path / f"{pdf_path.stem}.md"


def _validated_pdf_path(pdf_path: str | Path) -> Path:
    """Return a valid PDF path or raise the public input error."""
    input_path = Path(pdf_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"PDF not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {input_path}")
    return input_path


class PdfMarkdownConverter:
    """Reusable Marker PDF-to-Markdown converter.

    Marker's models are initialized once when this object is created. Reuse
    the same instance for multiple PDFs to avoid reloading the models for
    every conversion.
    """

    def __init__(self) -> None:
        # Keep Marker optional at module-import time and load its heavier
        # dependencies only when a converter is actually requested.
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        config = {
            "disable_ocr": True,
            "disable_image_extraction": True,
            "output_format": "markdown",
        }

        self._converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config=config
        )

    def convert(
        self,
        pdf_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Convert one PDF to Markdown and return the output path."""
        input_path = _validated_pdf_path(pdf_path)

        markdown_path = _markdown_output_path(input_path, output_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        rendered = self._converter(str(input_path))
        markdown_path.write_text(rendered.markdown, encoding="utf-8")
        if not markdown_path.is_file():
            raise RuntimeError(
                "Marker did not create the expected file: "
                f"{markdown_path}"
            )

        return markdown_path


def pdf_to_markdown(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert one PDF to Markdown using Marker.

    Parameters
    ----------
    pdf_path
        PDF file to convert.
    output_path
        Optional output directory or explicit ``.md`` file path. If omitted,
        the Markdown file is written beside the PDF with the same stem.

    Returns
    -------
    pathlib.Path
        Path to the generated Markdown file.
    """
    input_path = _validated_pdf_path(pdf_path)
    return PdfMarkdownConverter().convert(input_path, output_path)


__all__ = ["PdfMarkdownConverter", "pdf_to_markdown"]
