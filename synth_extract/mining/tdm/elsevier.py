#!/usr/bin/env python3
"""Elsevier full-text downloader.

Reads papers from the central SQLite coordination database, downloads the
full text of every paper whose ``canonical_source`` is currently
``"elsevier"`` and whose ``download_status`` is ``"pending"``, and writes the
outcome of each attempt back to that same row.

This script only ever attempts the *current* canonical source. It never
advances ``canonical_source`` or falls back to another source -- that
decision belongs to whatever orchestrates the multi-source workflow.

Designed to run unattended as a SLURM batch/array job: all logging goes to
stdout (which SLURM already redirects to a job log file), progress is
committed to the database after every single paper (not batched) so a job
that gets killed partway through loses no work, and SIGTERM/SIGINT trigger a
clean stop after the in-flight paper finishes.

There is no official Python SDK for the ScienceDirect Article (Full Text)
Retrieval API comparable to ``wiley_tdm``, so this talks to the plain REST
endpoint directly with ``requests`` (see ``ElsevierClient`` below), following
the DOI-based retrieval shape documented at
https://dev.elsevier.com/documentation/ArticleRetrievalAPI.wadl

Usage
-----
    export SCOPUS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    python elsevier_downloader.py --db /path/to/central.sqlite \\
        --output-dir /path/to/data/fulltext/elsevier

See ``--help`` for all options.

Requires: requests, lxml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import re

import requests
from lxml import etree

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SOURCE_NAME = "elsevier"
DEFAULT_TABLE = "papers"
# Elsevier's per-second throttling is tight compared to Wiley's; 0.125s
# between requests keeps us at/under ~8 req/s, safely inside published
# ScienceDirect full-text throttling limits.
DEFAULT_RATE_LIMIT_SECONDS = 0.125
ELSEVIER_API_READ_TIMEOUT_SECONDS = 60

# Only a clean HTTP 200 with a real full-text body (not a <service-error>
# document) counts as success.
_SUCCESS_STATUSES = {"SUCCESS"}  # placeholder, replaced below once DownloadStatus exists

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

logger = logging.getLogger("elsevier_downloader")


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

    # Quiet down urllib3/requests' own connection-pool chatter at INFO; it
    # doesn't add anything grep-worthy that our own per-paper log lines don't
    # already cover.
    logging.getLogger("urllib3").setLevel(max(logging.WARNING, root.level))


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


def _xpath_local_text(root: "etree._Element", local_tag: str) -> Optional[str]:
    """Find the text of the first descendant named `local_tag`, ignoring namespaces."""
    matches = root.xpath(f"//*[local-name()='{local_tag}']")
    if matches and matches[0].text:
        return matches[0].text.strip()
    return None


def _extract_service_error(content: bytes) -> Optional[tuple[str, Optional[str]]]:
    """If `content` is an Elsevier <service-error> document, return (code, text)."""
    if not content:
        return None
    try:
        root = etree.fromstring(content)
    except etree.XMLSyntaxError:
        return None
    if root is None or etree.QName(root).localname != "service-error":
        return None
    code = _xpath_local_text(root, "statusCode") or "UNKNOWN"
    text = _xpath_local_text(root, "statusText")
    return code, text


def validate_xml(path: Path) -> tuple[bool, str]:
    """Perform checks that a path contains a plausible, non-error full-text XML doc."""
    try:
        if not path.exists():
            return False, "file does not exist"
        if not path.is_file():
            return False, "path is not a regular file"

        size = path.stat().st_size
        if size == 0:
            return False, "file is empty"

        try:
            tree = etree.parse(str(path))
        except etree.XMLSyntaxError as exc:
            return False, f"not well-formed XML: {exc}"

        root = tree.getroot()
        root_tag = etree.QName(root).localname if root is not None else None

        if root_tag == "service-error":
            code = _xpath_local_text(root, "statusCode") or "UNKNOWN"
            text = _xpath_local_text(root, "statusText")
            detail = f"{code}: {text}" if text else code
            return False, f"file is an Elsevier <service-error> document ({detail})"

        if root_tag != "full-text-retrieval-response":
            return False, f"unexpected root element <{root_tag}>"

        if not root.xpath("//*[local-name()='originalText']"):
            return False, "full-text response contains no <originalText> element"
    except OSError as exc:
        return False, f"could not inspect file: {exc}"

    return True, f"well-formed XML (root element <{root_tag}>)"


class RateLimiter:
    """Guarantees at least `interval` seconds between successive `wait()` calls.

    We call the Elsevier API one paper at a time (so we can write each
    result to the database immediately), so we do our own client-side
    pacing on top of whatever server-side throttling Elsevier applies.
    """

    def __init__(self, interval: float):
        self.interval = interval
        self._last_call: Optional[float] = None

    def wait(self) -> None:
        if self._last_call is not None:
            remaining = self.interval - (time.monotonic() - self._last_call)
            if remaining > 0:
                logger.debug("Rate limit: sleeping %.3fs before next request.", remaining)
                time.sleep(remaining)
        self._last_call = time.monotonic()


# --------------------------------------------------------------------------- #
# Elsevier API client
# --------------------------------------------------------------------------- #


class DownloadStatus(Enum):
    """Outcome of a single Elsevier full-text retrieval attempt."""

    SUCCESS = "SUCCESS"
    ACCESS_DENIED = "ACCESS_DENIED"      # HTTP 401 / 403 -- key or entitlement problem
    UNKNOWN_DOI = "UNKNOWN_DOI"          # HTTP 404 -- RESOURCE_NOT_FOUND
    INVALID_DOI = "INVALID_DOI"          # failed our own format check
    RATE_LIMITED = "RATE_LIMITED"        # HTTP 429 -- quota or throttle
    API_ERROR = "API_ERROR"              # HTTP 400 / 5xx / anything else unexpected
    NETWORK_ERROR = "NETWORK_ERROR"      # connection/timeout error
    STORAGE_ERROR = "STORAGE_ERROR"      # could not write the file locally
    KNOWN_ISSUE = "KNOWN_ISSUE"          # HTTP 200 but body is a <service-error>, or invalid XML
    OTHER_ERROR = "OTHER_ERROR"          # unexpected non-API/non-storage program error


_SUCCESS_STATUSES = {DownloadStatus.SUCCESS}

# Extra, human-readable context appended to `last_error` for each failure
# status. Sourced from Elsevier's own "API Error Messages" and API key
# settings documentation (https://dev.elsevier.com/tecdoc_article_access.html,
# https://dev.elsevier.com/api_key_settings.html). These summarise the
# scenarios Elsevier documents; they don't change how the result was
# classified, they just make the stored error message more useful later.
_ERROR_HINTS: dict[DownloadStatus, str] = {
    DownloadStatus.ACCESS_DENIED: (
        "Elsevier API denied access (HTTP 401/403). HTTP 401 usually means the "
        "API key/IP combination is unrecognized or lacks privileges for this "
        "resource; HTTP 403 usually means the requestor's institutional "
        "entitlements (by IP or token) don't cover this document's source."
    ),
    DownloadStatus.UNKNOWN_DOI: (
        "Elsevier API returned HTTP 404 (typically RESOURCE_NOT_FOUND). Either "
        "this DOI is not on ScienceDirect, or the DOI does not exist."
    ),
    DownloadStatus.RATE_LIMITED: (
        "Elsevier API returned HTTP 429. Check the X-ELS-Status response header: "
        "QUOTA_EXCEEDED means the API key's period quota is used up (see "
        "X-RateLimit-Reset), otherwise the per-second throttling rate was "
        "exceeded and a slower --rate-limit is needed."
    ),
    DownloadStatus.API_ERROR: (
        "Elsevier API error. HTTP 400 usually means the 'view=FULL' parameter "
        "was rejected for this DOI's entitlement level; HTTP 5xx means a "
        "problem on Elsevier's side."
    ),
    DownloadStatus.STORAGE_ERROR: "Could not write the XML to local disk.",
    DownloadStatus.INVALID_DOI: "DOI failed format validation.",
    DownloadStatus.NETWORK_ERROR: "Network/connection error while calling the Elsevier API.",
    DownloadStatus.KNOWN_ISSUE: (
        "Elsevier returned HTTP 200 but the body was a <service-error> document, "
        "or the downloaded XML failed validation."
    ),
    DownloadStatus.OTHER_ERROR: "Unexpected error in the Elsevier downloader.",
}


@dataclass
class DownloadResult:
    status: DownloadStatus
    api_status: Optional[int] = None
    comment: Optional[str] = None
    duration: Optional[float] = None
    content: Optional[bytes] = None  # only populated on SUCCESS


class ElsevierClient:
    """Minimal wrapper around the Elsevier Article (Full Text) Retrieval API.

    There's no official SDK for this endpoint comparable to wiley_tdm, so we
    talk to the plain REST resource directly, mirroring the request shape
    Elsevier documents: GET the DOI resource under /content/article/doi/,
    with an X-ELS-APIKey header and view=FULL to ask for the full text.
    """

    BASE_URL = "https://api.elsevier.com/content/article/doi/"

    _STATUS_MAP = {
        400: DownloadStatus.API_ERROR,
        401: DownloadStatus.ACCESS_DENIED,
        403: DownloadStatus.ACCESS_DENIED,
        404: DownloadStatus.UNKNOWN_DOI,
        429: DownloadStatus.RATE_LIMITED,
    }

    def __init__(self, api_key: str, timeout: int = ELSEVIER_API_READ_TIMEOUT_SECONDS):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-ELS-APIKey": api_key,
                "Accept": "text/xml",
            }
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "ElsevierClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def download_xml(self, doi: str) -> DownloadResult:
        # Keep the DOI's internal '/' literal (Elsevier's URL shape expects
        # prefix/suffix separated by a single slash) but escape anything else
        # that would otherwise corrupt the URL.
        url = self.BASE_URL + quote(doi, safe="/")
        start = time.monotonic()
        try:
            response = self.session.get(url, params={"view": "FULL"}, timeout=self.timeout)
        except requests.Timeout as exc:
            return DownloadResult(
                status=DownloadStatus.NETWORK_ERROR,
                comment=f"Timed out after {self.timeout}s: {exc}",
                duration=time.monotonic() - start,
            )
        except requests.RequestException as exc:
            return DownloadResult(
                status=DownloadStatus.NETWORK_ERROR,
                comment=str(exc),
                duration=time.monotonic() - start,
            )

        duration = time.monotonic() - start
        return self._classify_response(response, duration)

    def _classify_response(self, response: requests.Response, duration: float) -> DownloadResult:
        status_code = response.status_code

        if status_code == 200:
            # Elsevier can return HTTP 200 with a <service-error> body in some
            # scenarios; don't trust the status code alone.
            service_error = _extract_service_error(response.content)
            if service_error:
                code, text = service_error
                detail = f"{code}: {text}" if text else code
                return DownloadResult(
                    status=DownloadStatus.KNOWN_ISSUE,
                    api_status=200,
                    comment=f"HTTP 200 but body was a <service-error> ({detail})",
                    duration=duration,
                )
            return DownloadResult(
                status=DownloadStatus.SUCCESS,
                api_status=200,
                comment=response.headers.get("X-ELS-Status"),
                duration=duration,
                content=response.content,
            )

        status = self._STATUS_MAP.get(status_code, DownloadStatus.API_ERROR)
        return DownloadResult(
            status=status,
            api_status=status_code,
            comment=self._build_comment(response),
            duration=duration,
        )

    @staticmethod
    def _build_comment(response: requests.Response) -> str:
        els_status = response.headers.get("X-ELS-Status")
        rate_remaining = response.headers.get("X-RateLimit-Remaining")
        rate_reset = response.headers.get("X-RateLimit-Reset")

        service_error = _extract_service_error(response.content)
        if service_error:
            code, text = service_error
            body_msg = f"{code}: {text}" if text else code
        else:
            snippet = response.text.strip().replace("\n", " ")
            body_msg = snippet[:300] if snippet else None

        parts = [p for p in (els_status, body_msg) if p]
        if rate_remaining is not None:
            parts.append(f"X-RateLimit-Remaining={rate_remaining}")
        if rate_reset is not None:
            parts.append(f"X-RateLimit-Reset={rate_reset}")
        return " | ".join(parts) if parts else "(no response body)"


def build_error_message(result: DownloadResult) -> str:
    parts = [result.status.name]
    if result.api_status is not None:
        parts.append(f"HTTP {int(result.api_status)}")
    if result.comment:
        parts.append(str(result.comment))
    hint = _ERROR_HINTS.get(result.status)
    if hint:
        parts.append(hint)
    return " | ".join(parts)


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
    client: Optional[ElsevierClient],
    store: PaperStore,
    paper: Paper,
    output_dir: Path,
    rate_limiter: RateLimiter,
    dry_run: bool = False,
) -> str:
    """Handle a single paper end to end. Returns 'success', 'skipped' or 'failed'."""

    if not paper.doi or not paper.doi.strip():
        message = "canonical_source is 'elsevier' but this paper has no DOI."
        logger.error("%s | %s", paper.paper_uid, message)
        if not dry_run:
            store.mark_failure(paper, DownloadStatus.INVALID_DOI, message)
        return "failed"

    paper_dir = output_dir / paper.paper_uid
    target_path = paper_dir / f"{paper.paper_uid}.xml"

    # A valid final file can be left behind if the process is killed after
    # the write but before the database update. Reconcile that state. An
    # invalid file is removed so Elsevier can be retried normally below.
    if target_path.exists():
        is_valid, validation_message = validate_xml(target_path)
        if is_valid:
            logger.info(
                "%s | Valid XML already exists at %s; reconciling database to success.",
                paper.paper_uid,
                target_path,
            )
            if not dry_run:
                store.mark_success(paper, target_path, target_path.suffix.lstrip("."))
                return "success"
            return "skipped"

        logger.warning(
            "%s | Invalid existing XML at %s (%s); removing it and retrying Elsevier.",
            paper.paper_uid,
            target_path,
            validation_message,
        )
        if not dry_run:
            try:
                target_path.unlink()
            except OSError as exc:
                message = (
                    f"Invalid existing XML at {target_path} ({validation_message}) "
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
    except OSError as exc:
        message = f"Could not prepare download directory {paper_dir}: {exc}"
        logger.exception("%s | %s", paper.paper_uid, message)
        store.mark_failure(paper, DownloadStatus.STORAGE_ERROR, message)
        return "failed"

    logger.info("%s | Requesting full text for DOI %s", paper.paper_uid, paper.doi)

    rate_limiter.wait()
    try:
        result = client.download_xml(paper.doi)
    except Exception as exc:  # noqa: BLE001 - batch job must survive one bad paper
        message = f"Unexpected exception calling ElsevierClient.download_xml: {exc!r}"
        logger.exception("%s | %s", paper.paper_uid, message)
        store.mark_failure(paper, DownloadStatus.OTHER_ERROR, message)
        return "failed"

    if result.status in _SUCCESS_STATUSES:
        # Write to a temp file in the same directory then rename, so a kill
        # mid-write never leaves a half-written file at the final path.
        tmp_path = target_path.with_name(target_path.name + ".part")
        try:
            tmp_path.write_bytes(result.content or b"")
            os.replace(tmp_path, target_path)
        except OSError as exc:
            message = f"Downloaded XML could not be written to {target_path}: {exc}"
            logger.exception("%s | %s", paper.paper_uid, message)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            store.mark_failure(paper, DownloadStatus.STORAGE_ERROR, message)
            return "failed"

        is_valid, validation_message = validate_xml(target_path)
        if not is_valid:
            message = (
                f"Downloaded file at {target_path} failed XML validation: "
                f"{validation_message}"
            )
            logger.error("%s | %s", paper.paper_uid, message)
            try:
                target_path.unlink(missing_ok=True)
            except OSError as exc:
                message = f"{message}; invalid file could not be removed: {exc}"
                logger.exception("%s | Could not remove invalid XML.", paper.paper_uid)
            store.mark_failure(paper, DownloadStatus.KNOWN_ISSUE, message)
            return "failed"

        logger.info(
            "%s | Download OK and XML validated (%.1fs) -> %s",
            paper.paper_uid, result.duration or 0.0, target_path,
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
        description="Download pending Elsevier full texts listed in the central coordination database.",
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
        "--output-dir", type=Path, default=Path("../data/fulltext/elsevier"),
        help="Root directory for downloaded XML files. A sub-folder named after "
             "each paper_uid is created inside it (default: ../data/fulltext/elsevier).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Elsevier Developer Portal API key. Defaults to the SCOPUS_API_KEY "
             "environment variable (the same key covers ScienceDirect full-text access).",
    )
    parser.add_argument(
        "--rate-limit", type=rate_limit_seconds, default=DEFAULT_RATE_LIMIT_SECONDS,
        help=f"Minimum seconds between Elsevier API calls (default: {DEFAULT_RATE_LIMIT_SECONDS:g}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on the number of pending papers to process this run "
             "(handy for SLURM array jobs, smoke-testing, or splitting a big backlog).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be downloaded without calling the Elsevier API or touching the database.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    api_key = args.api_key or os.environ.get("SCOPUS_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("No Elsevier API key provided (use --api-key or set SCOPUS_API_KEY).")
        return 2

    if not args.db.exists():
        logger.error("Database not found at %s", args.db)
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    client: Optional[ElsevierClient] = None
    if not args.dry_run:
        client = ElsevierClient(api_key=api_key, timeout=ELSEVIER_API_READ_TIMEOUT_SECONDS)
    rate_limiter = RateLimiter(args.rate_limit)

    counts = {"success": 0, "failed": 0, "skipped": 0}
    start = time.monotonic()

    try:
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
                        client, store, paper, output_dir, rate_limiter, dry_run=args.dry_run
                    )
                except sqlite3.Error:
                    logger.exception(
                        "%s | DATABASE_ERROR: SQLite failed while processing this paper; "
                        "the failure cannot be reliably written to the database.",
                        paper.paper_uid,
                    )
                    raise
                except Exception as exc:
                    message = (
                        f"Unhandled exception in downloader: {exc!r}; "
                        "see job log for traceback."
                    )
                    logger.exception("%s | %s", paper.paper_uid, message)
                    if not args.dry_run:
                        store.mark_failure(
                            paper, DownloadStatus.OTHER_ERROR,
                            message,
                        )
                    outcome = "failed"
                counts[outcome] = counts.get(outcome, 0) + 1
    finally:
        if client is not None:
            client.close()

    elapsed = time.monotonic() - start
    logger.info(
        "Done in %.1fs | success=%d skipped=%d failed=%d",
        elapsed, counts["success"], counts["skipped"], counts["failed"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
