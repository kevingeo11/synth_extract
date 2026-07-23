#!/usr/bin/env python3
"""Wiley full-text downloader.

Reads papers from the central SQLite coordination database, downloads the
full text of every paper whose ``canonical_source`` is currently ``"wiley"``
and whose ``download_status`` is ``"pending"``, and writes the outcome of
each attempt back to that same row.

This script only ever attempts the *current* canonical source. It never
advances ``canonical_source`` or falls back to another source -- that
decision belongs to whatever orchestrates the multi-source workflow.

Designed to run unattended as a SLURM batch/array job: all logging goes to
stdout (which SLURM already redirects to a job log file), progress is
committed to the database after every single paper (not batched) so a job
that gets killed partway through loses no work, and SIGTERM/SIGINT trigger a
clean stop after the in-flight paper finishes.

Usage
-----
    export WILEY_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    python wiley_downloader.py --db /path/to/central.sqlite \\
        --output-dir /path/to/data/fulltext/wiley

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pypdf import PasswordType, PdfReader
from wiley_tdm import DownloadResult, DownloadStatus, TDMClient

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SOURCE_NAME = "wiley"
DEFAULT_TABLE = "papers"
DEFAULT_RATE_LIMIT_SECONDS = 11.0
WILEY_API_READ_TIMEOUT_SECONDS = 60
PDF_EOF_SCAN_BYTES = 4096

# Wiley's own existing-file shortcut is disabled below, so only a fresh,
# completed download is treated as success.
_SUCCESS_STATUSES = {DownloadStatus.SUCCESS}

# Extra, human-readable context appended to `last_error` for each failure
# status. These summarise Wiley's TDM error table; they don't change how the
# result is classified (wiley_tdm / DownloadStatus already did that), they
# just make the stored error message more useful when someone reads it later.
_ERROR_HINTS: dict[DownloadStatus, str] = {
    DownloadStatus.ACCESS_DENIED: (
        "Wiley TDM denied access (HTTP 403). Usually an invalid/unregistered "
        "API token, or no institutional entitlement from this IP."
    ),
    DownloadStatus.UNKNOWN_DOI: (
        "Wiley TDM returned HTTP 404. Either the DOI is not on Wiley, or "
        "your institution/organization has no full-text access to it."
    ),
    DownloadStatus.API_ERROR: (
        "Wiley TDM API error. Check the HTTP status: 400 usually means a "
        "missing client token header, 429 means the rate limit was hit."
    ),
    DownloadStatus.STORAGE_ERROR: "Could not write the PDF to local disk.",
    DownloadStatus.INVALID_DOI: "DOI failed format validation.",
    DownloadStatus.NETWORK_ERROR: "Network/connection error while calling the Wiley TDM API.",
    DownloadStatus.KNOWN_ISSUE: "Wiley TDM flagged a known issue for this DOI.",
}

logger = logging.getLogger("wiley_downloader")

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def setup_logging(level: str = "INFO") -> None:
    """Configure logging to stdout only.

    Job schedulers like SLURM already capture stdout/stderr into their own
    per-job log files, so we deliberately don't write our own log files here
    -- that would just create a second, easily-forgotten copy. Everything is
    written to stdout with a timestamp/level/logger name so it can be
    grepped straight out of the SLURM output file.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)

    # wiley_tdm's own modules log via logging.getLogger(__name__); letting
    # them propagate to our handler surfaces its rate-limit / IP-check /
    # "downloading to" messages in the same stream instead of them being
    # silently dropped (the library itself never calls basicConfig()).
    logging.getLogger("wiley_tdm").setLevel(level.upper())


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json_list(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        logger.warning("Could not parse JSON list %r, treating as empty.", raw)
        return []
    return value if isinstance(value, list) else []


def validate_pdf(path: Path) -> tuple[bool, str]:
    """Perform dependency-free checks that a path contains a plausible PDF."""
    try:
        if not path.exists():
            return False, "file does not exist"
        if not path.is_file():
            return False, "path is not a regular file"

        size = path.stat().st_size
        if size == 0:
            return False, "file is empty"

        with path.open("rb") as pdf_file:
            if pdf_file.read(5) != b"%PDF-":
                return False, "file does not begin with the %PDF- signature"
            pdf_file.seek(max(0, size - PDF_EOF_SCAN_BYTES))
            if b"%%EOF" not in pdf_file.read():
                return False, "file has no %%EOF marker near the end"

        reader = PdfReader(path, strict=False)
        if (
            reader.is_encrypted
            and reader.decrypt("") == PasswordType.NOT_DECRYPTED
        ):
            return False, "PDF requires a password"
        page_count = len(reader.pages)
        if page_count == 0:
            return False, "PDF contains no pages"
    except OSError as exc:
        return False, f"could not inspect file: {exc}"
    except Exception as exc:  # pypdf uses several exception types for malformed PDFs
        return False, f"pypdf could not parse file: {exc}"

    return True, f"valid PDF structure ({page_count} pages)"


def build_error_message(result: DownloadResult) -> str:
    parts = [str(result.status)]
    if result.api_status is not None:
        parts.append(f"HTTP {int(result.api_status)}")
    if result.comment:
        parts.append(str(result.comment))
    hint = _ERROR_HINTS.get(result.status)
    if hint:
        parts.append(hint)
    return " | ".join(parts)


class RateLimiter:
    """Guarantees at least `interval` seconds between successive `wait()` calls.

    This is a client-side courtesy delay on top of whatever TDMClient does
    internally. We call `download_pdf()` one paper at a time (so we can
    write each result to the database immediately), so we do our own pacing
    rather than relying on TDMClient.download_pdfs()'s batch-mode sleep.
    """

    def __init__(self, interval: float):
        self.interval = interval
        self._last_call: Optional[float] = None

    def wait(self) -> None:
        if self._last_call is not None:
            remaining = self.interval - (time.monotonic() - self._last_call)
            if remaining > 0:
                logger.debug("Rate limit: sleeping %.1fs before next request.", remaining)
                time.sleep(remaining)
        self._last_call = time.monotonic()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Paper:
    paper_uid: str
    doi: Optional[str]
    canonical_source: str
    attempt_count: int
    attempted_sources: list = field(default_factory=list)
    failure_history: list = field(default_factory=list)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Paper":
        return cls(
            paper_uid=row["paper_uid"],
            doi=row["doi"],
            canonical_source=row["canonical_source"],
            attempt_count=row["attempt_count"] or 0,
            attempted_sources=_load_json_list(row["attempted_sources"]),
            failure_history=_load_json_list(row["failure_history"]),
        )


# --------------------------------------------------------------------------- #
# Database layer
# --------------------------------------------------------------------------- #


class PaperStore:
    """Thin wrapper around the central coordination SQLite database."""

    def __init__(self, db_path: Path, table: str = DEFAULT_TABLE):
        if not _TABLE_NAME_RE.match(table):
            raise ValueError(f"Unsafe table name: {table!r}")
        self.table = table
        # autocommit (isolation_level=None) -- each UPDATE below lands on
        # disk immediately, so a killed job never loses more than the paper
        # it was in the middle of.
        self.conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        journal_mode = self.conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        if journal_mode != "delete":
            raise RuntimeError(
                f"Expected SQLite journal_mode='delete', found {journal_mode!r}"
            )
        self.conn.execute("PRAGMA busy_timeout=30000")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "PaperStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def fetch_pending(self, limit: Optional[int] = None) -> list[Paper]:
        query = (
            f"SELECT paper_uid, doi, canonical_source, attempt_count, "
            f"attempted_sources, failure_history "
            f"FROM {self.table} "
            f"WHERE canonical_source = ? AND download_status = 'pending' "
            f"ORDER BY paper_uid"
        )
        params: list[Any] = [SOURCE_NAME]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [Paper.from_row(r) for r in rows]

    def mark_success(self, paper: Paper, fulltext_path: Path, fulltext_format: str) -> None:
        now = utcnow_iso()
        attempted_sources = [*paper.attempted_sources, SOURCE_NAME]
        self.conn.execute(
            f"""
            UPDATE {self.table}
            SET download_status       = 'success',
                downloaded_from       = ?,
                fulltext_path         = ?,
                fulltext_format       = ?,
                downloaded_at         = ?,
                last_attempted_source = ?,
                last_attempted_at     = ?,
                last_error            = NULL,
                updated_at            = ?,
                attempted_sources     = ?,
                attempt_count         = attempt_count + 1
            WHERE paper_uid = ?
            """,
            (
                SOURCE_NAME,
                str(fulltext_path),
                fulltext_format,
                now,
                SOURCE_NAME,
                now,
                now,
                json.dumps(attempted_sources),
                paper.paper_uid,
            ),
        )

    def mark_failure(self, paper: Paper, status: DownloadStatus, error_message: str) -> None:
        now = utcnow_iso()
        attempted_sources = [*paper.attempted_sources, SOURCE_NAME]
        failure_entry = {
            "source": SOURCE_NAME,
            "status": status.name,
            "error": error_message,
            "attempt": paper.attempt_count + 1,
            "timestamp": now,
        }
        failure_history = [*paper.failure_history, failure_entry]
        self.conn.execute(
            f"""
            UPDATE {self.table}
            SET download_status       = 'failed',
                last_attempted_source = ?,
                last_attempted_at     = ?,
                last_error            = ?,
                updated_at            = ?,
                attempted_sources     = ?,
                failure_history       = ?,
                attempt_count         = attempt_count + 1
            WHERE paper_uid = ?
            """,
            (
                SOURCE_NAME,
                now,
                error_message,
                now,
                json.dumps(attempted_sources),
                json.dumps(failure_history),
                paper.paper_uid,
            ),
        )


# --------------------------------------------------------------------------- #
# Core download logic
# --------------------------------------------------------------------------- #


def process_paper(
    tdm: TDMClient,
    store: PaperStore,
    paper: Paper,
    output_dir: Path,
    rate_limiter: RateLimiter,
    dry_run: bool = False,
) -> str:
    """Handle a single paper end to end. Returns 'success', 'skipped' or 'failed'."""

    if not paper.doi or not paper.doi.strip():
        message = "canonical_source is 'wiley' but this paper has no DOI."
        logger.error("%s | %s", paper.paper_uid, message)
        if not dry_run:
            store.mark_failure(paper, DownloadStatus.INVALID_DOI, message)
        return "failed"

    paper_dir = output_dir / paper.paper_uid
    target_path = paper_dir / f"{paper.paper_uid}.pdf"

    # A valid final file can be left behind if the process is killed after
    # the move but before the database update. Reconcile that state. An
    # invalid file is removed so Wiley can be retried normally below.
    if target_path.exists():
        is_valid, validation_message = validate_pdf(target_path)
        if is_valid:
            logger.info(
                "%s | Valid PDF already exists at %s; reconciling database to success.",
                paper.paper_uid,
                target_path,
            )
            if not dry_run:
                store.mark_success(paper, target_path, target_path.suffix.lstrip("."))
                return "success"
            return "skipped"

        logger.warning(
            "%s | Invalid existing PDF at %s (%s); removing it and retrying Wiley.",
            paper.paper_uid,
            target_path,
            validation_message,
        )
        if not dry_run:
            try:
                target_path.unlink()
            except OSError as exc:
                message = (
                    f"Invalid existing PDF at {target_path} ({validation_message}) "
                    f"could not be removed: {exc}"
                )
                logger.error("%s | %s", paper.paper_uid, message)
                store.mark_failure(paper, DownloadStatus.STORAGE_ERROR, message)
                return "failed"

    if dry_run:
        logger.info("%s | [dry-run] would download DOI %s", paper.paper_uid, paper.doi)
        return "skipped"

    try:
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Point TDMClient straight at this paper's own folder so it saves the
        # file where we want it, instead of a shared scratch dir we'd then have
        # to move it out of.
        tdm.download_dir = paper_dir
    except OSError as exc:
        message = f"Could not prepare download directory {paper_dir}: {exc}"
        logger.exception("%s | %s", paper.paper_uid, message)
        store.mark_failure(paper, DownloadStatus.STORAGE_ERROR, message)
        return "failed"

    logger.info("%s | Requesting PDF for DOI %s", paper.paper_uid, paper.doi)

    rate_limiter.wait()
    try:
        result = tdm.download_pdf(paper.doi)
    except ValueError as exc:
        # Defensive: DOI was non-empty above, but download_pdf() re-validates.
        message = f"TDMClient rejected the DOI: {exc}"
        logger.error("%s | %s", paper.paper_uid, message)
        store.mark_failure(paper, DownloadStatus.INVALID_DOI, message)
        return "failed"
    except Exception as exc:  # noqa: BLE001 - batch job must survive one bad paper
        message = f"Unexpected exception calling TDMClient.download_pdf: {exc!r}"
        logger.exception("%s | %s", paper.paper_uid, message)
        store.mark_failure(paper, DownloadStatus.API_ERROR, message)
        return "failed"

    if result.status in _SUCCESS_STATUSES:
        original_path = Path(result.path)
        try:
            if original_path != target_path:
                if target_path.exists():
                    # Very unlikely race: something else wrote the target
                    # between our check and now. Keep the existing file, drop
                    # the one we just fetched.
                    logger.warning(
                        "%s | %s appeared during download; discarding freshly "
                        "downloaded %s and keeping the existing file.",
                        paper.paper_uid, target_path, original_path,
                    )
                    original_path.unlink(missing_ok=True)
                else:
                    shutil.move(str(original_path), str(target_path))
        except (OSError, shutil.Error) as exc:
            message = (
                f"Downloaded PDF could not be moved from {original_path} "
                f"to {target_path}: {exc}"
            )
            logger.exception("%s | %s", paper.paper_uid, message)
            store.mark_failure(paper, DownloadStatus.STORAGE_ERROR, message)
            return "failed"

        is_valid, validation_message = validate_pdf(target_path)
        if not is_valid:
            message = (
                f"Downloaded file at {target_path} failed PDF validation: "
                f"{validation_message}"
            )
            logger.error("%s | %s", paper.paper_uid, message)
            try:
                target_path.unlink(missing_ok=True)
            except OSError as exc:
                message = f"{message}; invalid file could not be removed: {exc}"
                logger.exception("%s | Could not remove invalid PDF.", paper.paper_uid)
            store.mark_failure(paper, DownloadStatus.KNOWN_ISSUE, message)
            return "failed"

        logger.info(
            "%s | Download OK and PDF validated (%s, %.1fs) -> %s",
            paper.paper_uid, result.status, result.duration or 0.0, target_path,
        )
        store.mark_success(paper, target_path, target_path.suffix.lstrip("."))
        return "success"

    message = build_error_message(result)
    logger.error("%s | Download failed: %s", paper.paper_uid, message)
    store.mark_failure(paper, result.status, message)
    return "failed"


# --------------------------------------------------------------------------- #
# Graceful shutdown (SLURM sends SIGTERM before killing a job)
# --------------------------------------------------------------------------- #

_stop_requested = False


def _handle_stop_signal(signum: int, _frame: Any) -> None:
    global _stop_requested
    logger.warning(
        "Received signal %s, will stop after the in-flight paper finishes.", signum
    )
    _stop_requested = True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def rate_limit_seconds(value: str) -> float:
    interval = float(value)
    if interval < DEFAULT_RATE_LIMIT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"rate limit must be at least {DEFAULT_RATE_LIMIT_SECONDS:g} seconds"
        )
    return interval


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pending Wiley full texts listed in the central coordination database.",
    )
    parser.add_argument(
        "--db", type=Path, required=True,
        help="Path to the central coordination SQLite database.",
    )
    parser.add_argument(
        "--table", default=DEFAULT_TABLE,
        help=f"Name of the coordination table (default: {DEFAULT_TABLE}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../data/fulltext/wiley"),
        help="Root directory for downloaded PDFs. A sub-folder named after "
             "each paper_uid is created inside it (default: ../data/fulltext/wiley).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Wiley TDM API token. Defaults to the WILEY_API_KEY environment variable.",
    )
    parser.add_argument(
        "--rate-limit", type=rate_limit_seconds, default=DEFAULT_RATE_LIMIT_SECONDS,
        help=f"Minimum seconds between Wiley TDM API calls (default: {DEFAULT_RATE_LIMIT_SECONDS}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on the number of pending papers to process this run "
             "(handy for SLURM array jobs, smoke-testing, or splitting a big backlog).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be downloaded without calling the Wiley API or touching the database.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    api_key = args.api_key or os.environ.get("WILEY_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("No Wiley API key provided (use --api-key or set WILEY_API_KEY).")
        return 2

    if not args.db.exists():
        logger.error("Database not found at %s", args.db)
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    tdm: Optional[TDMClient] = None
    if not args.dry_run:
        # The library defaults to 30 seconds, which can be too short for
        # Wiley to begin/continue returning larger PDFs.
        TDMClient.API_READ_TIMEOUT = WILEY_API_READ_TIMEOUT_SECONDS
        tdm = TDMClient(api_token=api_key, download_dir=output_dir)
        # A DOI-named file may be stale or left by an interrupted download.
        # Always make a fresh request instead of accepting it as EXISTING_FILE.
        tdm.skip_existing_files = False
        # Validates >= TDMClient.API_RATE_LIMIT (5.0s); raises otherwise.
        tdm.api_rate_limit = max(args.rate_limit, TDMClient.API_RATE_LIMIT)
    rate_limiter = RateLimiter(args.rate_limit)

    counts = {"success": 0, "failed": 0, "skipped": 0}
    start = time.monotonic()

    with PaperStore(args.db, table=args.table) as store:
        papers = store.fetch_pending(limit=args.limit)
        logger.info(
            "Found %d pending paper(s) with canonical_source='%s'.%s",
            len(papers), SOURCE_NAME, " [dry-run]" if args.dry_run else "",
        )

        for i, paper in enumerate(papers, start=1):
            if _stop_requested:
                logger.warning(
                    "Stopping before paper %d/%d (%s) due to shutdown signal.",
                    i, len(papers), paper.paper_uid,
                )
                break

            logger.info("[%d/%d] Processing %s", i, len(papers), paper.paper_uid)
            try:
                outcome = process_paper(
                    tdm, store, paper, output_dir, rate_limiter, dry_run=args.dry_run
                )
            except Exception:
                logger.exception(
                    "%s | Unhandled error in process_paper, marking as failed and continuing.",
                    paper.paper_uid,
                )
                if not args.dry_run:
                    store.mark_failure(
                        paper, DownloadStatus.API_ERROR,
                        "Unhandled exception in downloader; see job log for traceback.",
                    )
                outcome = "failed"
            counts[outcome] = counts.get(outcome, 0) + 1

    elapsed = time.monotonic() - start
    logger.info(
        "Done in %.1fs | success=%d skipped=%d failed=%d",
        elapsed, counts["success"], counts["skipped"], counts["failed"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
