#!/usr/bin/env python3
"""Convert PDFs in a tracking SQLite database to Markdown.

Pending rows are those whose ``convert_md`` value is NULL. For each pending
row, the PDF is expected at::

    FULLTEXT_ROOT / CANONICAL_SOURCE / PAPER_UID / (PAPER_UID + ".pdf")

The Markdown file is written beside the PDF. Successful conversions and
already-existing non-empty Markdown files are recorded as ``convert_md = 1``.
Missing PDFs and conversion failures remain NULL so a later run can retry
them. A single Marker converter is initialized and reused for the full run.

Usage
-----
    python scripts/process/pdf_to_md.py \
        --db data/process/arxiv_track.db \
        --fulltext-root data/fulltext

    python scripts/process/pdf_to_md.py \
        --db data/process/wiley_track.db \
        --fulltext-root data/fulltext
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synth_extract.mining.process.pdf_to_md import PdfMarkdownConverter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("pdf_to_md")
DEFAULT_COMMIT_EVERY = 25


@dataclass
class ConversionSummary:
    selected: int = 0
    converted: int = 0
    existing_markdown: int = 0
    missing_pdf: int = 0
    invalid_row: int = 0
    failed: int = 0
    remaining: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert pending PDFs from a tracking SQLite database to "
            "Markdown and persist progress in convert_md."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Tracking SQLite database containing the papers table.",
    )
    parser.add_argument(
        "--fulltext-root",
        type=Path,
        required=True,
        help="Directory containing source/paper_uid/paper_uid.pdf paths.",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=DEFAULT_COMMIT_EVERY,
        help=(
            "Commit progress after this many completed rows "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def _safe_component(value: str) -> bool:
    """Return whether a database value is safe as one path component."""
    return (
        bool(value)
        and value not in {".", ".."}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
    )


def _validate_schema(conn: sqlite3.Connection) -> None:
    table_exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'papers'
        """
    ).fetchone()
    if table_exists is None:
        raise ValueError(
            "Tracking database does not contain a papers table"
        )

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(papers)")
    }
    required = {
        "paper_id",
        "paper_uid",
        "canonical_source",
        "convert_md",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            "Tracking papers table is missing required column(s): "
            + ", ".join(missing)
        )


def _load_pending_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT paper_id, paper_uid, canonical_source
        FROM papers
        WHERE convert_md IS NULL
        ORDER BY paper_id
        """
    ).fetchall()


def _mark_completed(
    conn: sqlite3.Connection,
    paper_id: int,
) -> None:
    cursor = conn.execute(
        """
        UPDATE papers
        SET convert_md = 1
        WHERE paper_id = ?
          AND convert_md IS NULL
        """,
        (paper_id,),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(
            f"Could not mark paper_id={paper_id} as completed"
        )


def convert_tracking_database(
    db_path: Path,
    fulltext_root: Path,
    commit_every: int = DEFAULT_COMMIT_EVERY,
) -> ConversionSummary:
    """Convert pending tracking rows and persist resumable progress."""
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Tracking database not found: {db_path}"
        )
    if not fulltext_root.is_dir():
        raise FileNotFoundError(
            f"Full-text root not found: {fulltext_root}"
        )
    if commit_every < 1:
        raise ValueError("commit_every must be at least 1")

    summary = ConversionSummary()
    converter: PdfMarkdownConverter | None = None
    uncommitted_updates = 0

    with closing(sqlite3.connect(db_path, timeout=60)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 60000")
        _validate_schema(conn)

        pending_rows = _load_pending_rows(conn)
        summary.selected = len(pending_rows)
        LOG.info("Pending tracking rows: %d", summary.selected)

        for row in pending_rows:
            paper_id = row["paper_id"]
            paper_uid = (row["paper_uid"] or "").strip()
            canonical_source = (
                row["canonical_source"] or ""
            ).strip()

            if not _safe_component(paper_uid):
                summary.invalid_row += 1
                LOG.error(
                    "paper_id=%s has an unsafe paper_uid: %r",
                    paper_id,
                    paper_uid,
                )
                continue
            if not _safe_component(canonical_source):
                summary.invalid_row += 1
                LOG.error(
                    "paper_id=%s has an unsafe canonical_source: %r",
                    paper_id,
                    canonical_source,
                )
                continue

            paper_dir = (
                fulltext_root / canonical_source / paper_uid
            )
            pdf_path = paper_dir / f"{paper_uid}.pdf"
            markdown_path = paper_dir / f"{paper_uid}.md"

            if (
                markdown_path.is_file()
                and markdown_path.stat().st_size > 0
            ):
                _mark_completed(conn, paper_id)
                uncommitted_updates += 1
                summary.existing_markdown += 1
                LOG.info("Already converted: %s", markdown_path)
            elif not pdf_path.is_file():
                summary.missing_pdf += 1
                LOG.warning("PDF not found: %s", pdf_path)
                continue
            else:
                if converter is None:
                    from synth_extract.mining.process.pdf_to_md import (
                        PdfMarkdownConverter,
                    )

                    LOG.info("Initializing Marker converter.")
                    converter = PdfMarkdownConverter()

                LOG.info(
                    "Converting paper_id=%s: %s",
                    paper_id,
                    pdf_path,
                )
                try:
                    output_path = converter.convert(pdf_path)
                except Exception:
                    summary.failed += 1
                    LOG.exception(
                        "Conversion failed: %s",
                        pdf_path,
                    )
                    continue

                if (
                    not output_path.is_file()
                    or output_path.stat().st_size == 0
                ):
                    summary.failed += 1
                    LOG.error(
                        "Conversion did not create non-empty "
                        "Markdown: %s",
                        output_path,
                    )
                    continue

                _mark_completed(conn, paper_id)
                uncommitted_updates += 1
                summary.converted += 1
                LOG.info("Converted: %s", output_path)

            if uncommitted_updates >= commit_every:
                conn.commit()
                uncommitted_updates = 0
                LOG.debug("Committed conversion progress.")

        conn.commit()
        summary.remaining = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE convert_md IS NULL"
        ).fetchone()[0]

    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ),
    )

    try:
        summary = convert_tracking_database(
            db_path=args.db.resolve(),
            fulltext_root=args.fulltext_root.resolve(),
            commit_every=args.commit_every,
        )
    except (
        OSError,
        sqlite3.Error,
        RuntimeError,
        ValueError,
    ) as exc:
        LOG.exception("Pipeline failed: %s", exc)
        return 2

    LOG.info(
        "Done | selected=%d converted=%d existing_markdown=%d "
        "missing_pdf=%d invalid_row=%d failed=%d remaining=%d",
        summary.selected,
        summary.converted,
        summary.existing_markdown,
        summary.missing_pdf,
        summary.invalid_row,
        summary.failed,
        summary.remaining,
    )
    return 1 if summary.remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
