#!/usr/bin/env python3
"""Convert full-text PDFs referenced by a TSV file to Markdown.

Rows are filtered by canonical source. For each selected row, the PDF is
expected at::

    FULLTEXT_ROOT / CANONICAL_SOURCE / PAPER_UID / (PAPER_UID + ".pdf")

Successful conversions are recorded as ``1`` in a progress column in the
input TSV. Missing files and conversion failures leave that column empty.
Progress is written atomically after every success so an interrupted run can
resume without repeating completed work.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("convert_tsv_to_markdown")
DEFAULT_SOURCE_COLUMN = "canonical_source"
DEFAULT_UID_COLUMN = "paper_uid"
DEFAULT_PROGRESS_COLUMN = "markdown_completed"


@dataclass
class ConversionSummary:
    selected: int = 0
    converted: int = 0
    already_completed: int = 0
    existing_markdown: int = 0
    missing_pdf: int = 0
    invalid_uid: int = 0
    failed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PDFs for one canonical source in a TSV to Markdown and "
            "record successful rows in the TSV."
        )
    )
    parser.add_argument("--tsv", type=Path, required=True, help="Input TSV path")
    parser.add_argument(
        "--canonical-source",
        required=True,
        help="Canonical source to process, for example 'arxiv'",
    )
    parser.add_argument(
        "--fulltext-root",
        type=Path,
        required=True,
        help="Directory containing source/paper_uid/paper_uid.pdf paths",
    )
    parser.add_argument(
        "--source-column",
        default=DEFAULT_SOURCE_COLUMN,
        help="Canonical-source column (default: %(default)s)",
    )
    parser.add_argument(
        "--uid-column",
        default=DEFAULT_UID_COLUMN,
        help="Paper UID column (default: %(default)s)",
    )
    parser.add_argument(
        "--progress-column",
        default=DEFAULT_PROGRESS_COLUMN,
        help="Markdown progress column (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def load_tsv(tsv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a TSV without coercing identifiers or existing values."""
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {tsv_path}")
        return list(reader.fieldnames), list(reader)


def save_tsv_atomic(
    tsv_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """Replace the TSV atomically with the current in-memory rows."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{tsv_path.name}.",
            suffix=".tmp",
            dir=tsv_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(tsv_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def is_completed(value: str | None) -> bool:
    """Return whether a TSV progress value represents completion."""
    return (value or "").strip().casefold() in {"1", "true"}


def is_safe_uid(paper_uid: str) -> bool:
    """Reject empty UIDs and values capable of escaping their source folder."""
    return bool(paper_uid) and Path(paper_uid).name == paper_uid


def pdf_path_for_row(
    fulltext_root: Path,
    canonical_source: str,
    paper_uid: str,
) -> Path:
    """Construct the expected PDF path for one paper."""
    return fulltext_root / canonical_source / paper_uid / f"{paper_uid}.pdf"


def convert_pdf(pdf_path: Path) -> None:
    """Convert one PDF, importing Marker-related project code only when needed."""
    from synth_extract.utils.markdown_helpers import pdf_to_markdown

    pdf_to_markdown(pdf_path, markdown_only=True)


def convert_tsv_source(
    tsv_path: Path,
    canonical_source: str,
    fulltext_root: Path,
    source_column: str = DEFAULT_SOURCE_COLUMN,
    uid_column: str = DEFAULT_UID_COLUMN,
    progress_column: str = DEFAULT_PROGRESS_COLUMN,
) -> ConversionSummary:
    """Convert pending PDFs for one source and persist progress to the TSV."""
    if not tsv_path.is_file():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")
    if not fulltext_root.is_dir():
        raise FileNotFoundError(f"Full-text root not found: {fulltext_root}")

    fieldnames, rows = load_tsv(tsv_path)
    for required_column in (source_column, uid_column):
        if required_column not in fieldnames:
            raise ValueError(
                f"Required column {required_column!r} is absent from {tsv_path}"
            )

    if progress_column not in fieldnames:
        fieldnames.append(progress_column)
        for row in rows:
            row[progress_column] = ""
        save_tsv_atomic(tsv_path, fieldnames, rows)

    summary = ConversionSummary()
    for row_number, row in enumerate(rows, start=2):
        if row.get(source_column) != canonical_source:
            continue

        summary.selected += 1
        if is_completed(row.get(progress_column)):
            summary.already_completed += 1
            continue

        paper_uid = (row.get(uid_column) or "").strip()
        if not is_safe_uid(paper_uid):
            summary.invalid_uid += 1
            LOG.error("Row %d has an invalid paper UID: %r", row_number, paper_uid)
            continue

        pdf_path = pdf_path_for_row(fulltext_root, canonical_source, paper_uid)
        markdown_path = pdf_path.with_suffix(".md")
        if markdown_path.is_file():
            row[progress_column] = "1"
            save_tsv_atomic(tsv_path, fieldnames, rows)
            summary.existing_markdown += 1
            LOG.info("Already converted: %s", markdown_path)
            continue

        if not pdf_path.is_file():
            summary.missing_pdf += 1
            LOG.warning("PDF not found: %s", pdf_path)
            continue

        LOG.info("Converting row %d: %s", row_number, pdf_path)
        try:
            convert_pdf(pdf_path)
        except Exception:
            summary.failed += 1
            LOG.exception("Conversion failed: %s", pdf_path)
            continue

        if not markdown_path.is_file():
            summary.failed += 1
            LOG.error("Conversion returned without creating %s", markdown_path)
            continue

        row[progress_column] = "1"
        save_tsv_atomic(tsv_path, fieldnames, rows)
        summary.converted += 1
        LOG.info("Converted: %s", markdown_path)

    return summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        summary = convert_tsv_source(
            tsv_path=args.tsv,
            canonical_source=args.canonical_source,
            fulltext_root=args.fulltext_root,
            source_column=args.source_column,
            uid_column=args.uid_column,
            progress_column=args.progress_column,
        )
    except (OSError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2

    LOG.info(
        "Done | selected=%d converted=%d existing_markdown=%d "
        "already_completed=%d missing_pdf=%d invalid_uid=%d failed=%d",
        summary.selected,
        summary.converted,
        summary.existing_markdown,
        summary.already_completed,
        summary.missing_pdf,
        summary.invalid_uid,
        summary.failed,
    )
    return 1 if summary.failed or summary.invalid_uid else 0


if __name__ == "__main__":
    raise SystemExit(main())
