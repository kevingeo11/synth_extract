#!/usr/bin/env python3
"""Springer Nature open-access JATS full-text downloader.

Reads papers from the central SQLite coordination database, downloads the
full text of every paper whose ``canonical_source`` is ``"springer_nature"``
and whose ``download_status`` is ``"pending"``, and writes each outcome back
to the same row immediately.

The downloader only attempts the current canonical source. It does not move
the row to another source after failure; that belongs to the multi-source
workflow orchestrator.

API requests are grouped into batches of at most 20 DOI terms joined with
``OR``. A per-run request budget defaults to 400 and cannot exceed Springer
Nature's documented daily maximum of 500. Every attempted API call consumes
one unit of that budget, regardless of its result.

Usage
-----
    export SPRINGER_NATURE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    python -m synth_extract.mining.tdm.springer_nature \
        --db /path/to/central.sqlite \
        --output-dir /path/to/data/fulltext/springer_nature

Requires: requests, lxml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import requests
from lxml import etree

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SOURCE_NAME = "springer_nature"
DEFAULT_TABLE = "papers"
MIN_RATE_LIMIT_SECONDS = 1.0
DEFAULT_RATE_LIMIT_SECONDS = 2.0
REQUEST_BATCH_SIZE = 20
DEFAULT_REQUEST_LIMIT = 400
MAX_REQUEST_LIMIT = 500
API_TIMEOUT_SECONDS = 60
API_URL = "https://api.springernature.com/openaccess/jats"

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

logger = logging.getLogger("springer_nature_downloader")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def setup_logging(level: str = "INFO") -> None:
    """Configure timestamped logging to stdout for SLURM job capture."""
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
    logging.getLogger("urllib3").setLevel(max(logging.WARNING, root.level))


# --------------------------------------------------------------------------- #
# Helpers and validation
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


def _local_name(element: "etree._Element") -> str:
    return etree.QName(element).localname


def _first_local_text(root: "etree._Element", local_tag: str) -> Optional[str]:
    matches = root.xpath(f"//*[local-name()='{local_tag}']")
    if not matches:
        return None
    text = "".join(matches[0].itertext()).strip()
    return text or None


def _record_doi(record: "etree._Element") -> Optional[str]:
    """Extract the primary DOI from one JATS article or BITS book part."""
    xpaths = (
        ".//*[local-name()='article-meta']/*[local-name()='article-id' "
        "and @pub-id-type='doi']",
        ".//*[local-name()='book-part-meta']/*[local-name()='book-part-id' "
        "and @book-part-id-type='doi']",
        ".//*[local-name()='book-meta']/*[local-name()='book-id' "
        "and @book-id-type='doi']",
    )
    for xpath in xpaths:
        matches = record.xpath(xpath)
        if matches:
            text = "".join(matches[0].itertext()).strip()
            if text:
                return text
    return None


def _response_records(root: "etree._Element") -> list["etree._Element"]:
    return root.xpath(
        "//*[local-name()='records']/*[local-name()='article' "
        "or local-name()='book-part-wrapper']"
    )


def extract_jats_records(content: bytes) -> tuple[dict[str, bytes], str]:
    """Parse a batch response and return standalone XML records keyed by DOI."""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    try:
        root = etree.fromstring(content, parser)
    except (etree.XMLSyntaxError, ValueError) as exc:
        return {}, f"response is not well-formed XML: {exc}"

    if _local_name(root) != "response":
        return {}, f"unexpected root element <{_local_name(root)}>"

    total_text = _first_local_text(root, "total")
    try:
        total = int(total_text) if total_text is not None else 0
    except ValueError:
        return {}, f"response has invalid result total {total_text!r}"

    records: dict[str, bytes] = {}
    unmapped = 0
    for record in _response_records(root):
        doi = _record_doi(record)
        if not doi:
            unmapped += 1
            continue
        records.setdefault(
            doi.casefold(),
            etree.tostring(record, encoding="UTF-8", xml_declaration=True),
        )

    if total > 0 and not records:
        return {}, "response reports records, but none has a mappable DOI"

    detail = f"response total={total}, mapped records={len(records)}"
    if unmapped:
        detail += f", records without DOI={unmapped}"
    return records, detail


def validate_jats_xml(
    path: Path, expected_doi: Optional[str] = None
) -> tuple[bool, str]:
    """Check that a file contains a plausible JATS article or BITS book part."""
    try:
        if not path.exists():
            return False, "file does not exist"
        if not path.is_file():
            return False, "path is not a regular file"
        if path.stat().st_size == 0:
            return False, "file is empty"

        parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
        try:
            tree = etree.parse(str(path), parser)
        except etree.XMLSyntaxError as exc:
            return False, f"not well-formed XML: {exc}"

        root = tree.getroot()
        if root is None:
            return False, "XML has no root element"

        root_name = _local_name(root)
        if root_name in {"article", "book-part-wrapper"}:
            records = [root]
        elif root_name == "response":
            records = _response_records(root)
            if not records:
                return False, "response contains no JATS article or BITS book part"
        else:
            return False, f"unexpected root element <{root_name}>"

        record_dois = [doi for record in records if (doi := _record_doi(record))]
        if not record_dois:
            return False, "JATS/BITS record contains no DOI"
        if expected_doi is not None and expected_doi.casefold() not in {
            doi.casefold() for doi in record_dois
        }:
            return False, f"file DOI does not match expected DOI {expected_doi!r}"
    except OSError as exc:
        return False, f"could not inspect file: {exc}"
    except Exception as exc:  # lxml can expose several malformed-document errors
        return False, f"could not validate Springer Nature XML: {exc}"

    record_types = sorted({_local_name(record) for record in records})
    return True, f"valid Springer Nature XML ({', '.join(record_types)})"


class RateLimiter:
    """Guarantee a minimum interval between successive request start times."""

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
# Springer Nature API client
# --------------------------------------------------------------------------- #


class DownloadStatus(Enum):
    """Outcome of one Springer Nature JATS retrieval attempt."""

    SUCCESS = "SUCCESS"
    ACCESS_DENIED = "ACCESS_DENIED"
    UNKNOWN_DOI = "UNKNOWN_DOI"
    RATE_LIMITED = "RATE_LIMITED"
    API_ERROR = "API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    KNOWN_ISSUE = "KNOWN_ISSUE"
    OTHER_ERROR = "OTHER_ERROR"


_SUCCESS_STATUSES = {DownloadStatus.SUCCESS}

_ERROR_HINTS: dict[DownloadStatus, str] = {
    DownloadStatus.ACCESS_DENIED: (
        "Springer Nature denied access. Check SPRINGER_NATURE_API_KEY and the "
        "API account's access."
    ),
    DownloadStatus.UNKNOWN_DOI: (
        "No open-access JATS record matched this DOI. The API may return HTTP "
        "404 for a wholly unmatched request or omit the DOI from a successful batch."
    ),
    DownloadStatus.RATE_LIMITED: (
        "Springer Nature rate-limited the request. Use a slower --rate-limit "
        "or retry after the API quota resets."
    ),
    DownloadStatus.API_ERROR: (
        "Springer Nature API error. HTTP 5xx indicates a server-side problem."
    ),
    DownloadStatus.NETWORK_ERROR: (
        "Network/connection error while calling the Springer Nature API."
    ),
    DownloadStatus.STORAGE_ERROR: "Could not write the XML to local disk.",
    DownloadStatus.KNOWN_ISSUE: (
        "The API response did not contain a valid Springer Nature JATS/BITS record."
    ),
    DownloadStatus.OTHER_ERROR: "Unexpected error in the Springer Nature downloader.",
}


@dataclass
class BatchDownloadResult:
    status: DownloadStatus
    api_status: Optional[int] = None
    comment: Optional[str] = None
    duration: Optional[float] = None
    records: dict[str, bytes] = field(default_factory=dict)


class SpringerNatureClient:
    """Minimal client for the Springer Nature Open Access JATS API."""

    def __init__(self, api_key: str, timeout: int = API_TIMEOUT_SECONDS):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/xml",
                "User-Agent": "synth-extract-springer-nature-downloader/1.0",
            }
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "SpringerNatureClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _safe_request_error(self, exc: requests.RequestException) -> str:
        return str(exc).replace(self.api_key, "[REDACTED]")

    @staticmethod
    def _doi_query_term(doi: str) -> str:
        escaped = doi.replace("\\", "\\\\").replace('"', '\\"')
        return f'doi:"{escaped}"'

    def download_batch(self, dois: list[str]) -> BatchDownloadResult:
        if not dois:
            raise ValueError("DOI batch cannot be empty")
        if len(dois) > REQUEST_BATCH_SIZE:
            raise ValueError(
                f"DOI batch cannot exceed {REQUEST_BATCH_SIZE} entries"
            )

        unique_dois = list(dict.fromkeys(dois))
        query = "(" + " OR ".join(
            self._doi_query_term(doi) for doi in unique_dois
        ) + ")"
        start = time.monotonic()
        try:
            response = self.session.get(
                API_URL,
                params={
                    "api_key": self.api_key,
                    "callback": "",
                    "s": 1,
                    "p": REQUEST_BATCH_SIZE,
                    "q": query,
                },
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            return BatchDownloadResult(
                status=DownloadStatus.NETWORK_ERROR,
                comment=(
                    f"Timed out after {self.timeout}s: "
                    f"{self._safe_request_error(exc)}"
                ),
                duration=time.monotonic() - start,
            )
        except requests.RequestException as exc:
            return BatchDownloadResult(
                status=DownloadStatus.NETWORK_ERROR,
                comment=self._safe_request_error(exc),
                duration=time.monotonic() - start,
            )

        duration = time.monotonic() - start
        return self._classify_response(response, duration)

    def _classify_response(
        self, response: requests.Response, duration: float
    ) -> BatchDownloadResult:
        status_code = response.status_code
        if status_code == 200:
            records, parse_comment = extract_jats_records(response.content)
            total_text = self._success_comment(response)
            comment = " | ".join(
                part for part in (total_text, parse_comment) if part
            )
            if not records and "total=0" not in parse_comment:
                return BatchDownloadResult(
                    status=DownloadStatus.KNOWN_ISSUE,
                    api_status=200,
                    comment=comment,
                    duration=duration,
                )
            return BatchDownloadResult(
                status=DownloadStatus.SUCCESS,
                api_status=200,
                comment=comment,
                duration=duration,
                records=records,
            )

        if status_code in (401, 403):
            status = DownloadStatus.ACCESS_DENIED
        elif status_code == 404:
            status = DownloadStatus.UNKNOWN_DOI
        elif status_code == 429:
            status = DownloadStatus.RATE_LIMITED
        else:
            status = DownloadStatus.API_ERROR

        return BatchDownloadResult(
            status=status,
            api_status=status_code,
            comment=self._error_comment(response),
            duration=duration,
        )

    @staticmethod
    def _success_comment(response: requests.Response) -> Optional[str]:
        dois_downloaded = response.headers.get("dois-downloaded")
        if dois_downloaded is None:
            return None
        return f"dois-downloaded={dois_downloaded}"

    @staticmethod
    def _error_comment(response: requests.Response) -> str:
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type.casefold():
            try:
                payload = response.json()
            except (requests.JSONDecodeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                parts: list[str] = []
                for key in ("status", "message"):
                    value = payload.get(key)
                    if value:
                        parts.append(str(value))
                error = payload.get("error")
                if isinstance(error, dict):
                    for key in ("error", "error_description"):
                        value = error.get(key)
                        if value:
                            parts.append(str(value))
                elif error:
                    parts.append(str(error))
                if parts:
                    return " | ".join(parts)

        snippet = response.text.strip().replace("\n", " ")
        return snippet[:500] if snippet else "(no response body)"


def build_error_message(result: BatchDownloadResult) -> str:
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
# Data model and database layer
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


class PaperStore:
    """Thin wrapper around the central coordination SQLite database."""

    def __init__(self, db_path: Path, table: str = DEFAULT_TABLE):
        if not _TABLE_NAME_RE.match(table):
            raise ValueError(f"Unsafe table name: {table!r}")
        self.table = table
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
        return [Paper.from_row(row) for row in rows]

    def mark_success(
        self, paper: Paper, fulltext_path: Path, fulltext_format: str
    ) -> None:
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

    def mark_failure(
        self, paper: Paper, status: DownloadStatus, error_message: str
    ) -> None:
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


def prepare_paper(
    store: PaperStore,
    paper: Paper,
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[Optional[str], Optional[Path]]:
    """Reconcile local state and return (outcome, target) for batch planning."""
    if not paper.doi or not paper.doi.strip():
        message = "canonical_source is 'springer_nature' but this paper has no DOI."
        logger.error("%s | %s", paper.paper_uid, message)
        if not dry_run:
            store.mark_failure(paper, DownloadStatus.OTHER_ERROR, message)
        return "failed", None

    paper_dir = output_dir / paper.paper_uid
    target_path = paper_dir / f"{paper.paper_uid}.xml"

    # Reconcile a valid file left by a kill between the atomic rename and DB
    # update. Remove an invalid final file and make a fresh request.
    if target_path.exists():
        is_valid, validation_message = validate_jats_xml(target_path, paper.doi)
        if is_valid:
            logger.info(
                "%s | Valid XML already exists at %s; reconciling database to success.",
                paper.paper_uid,
                target_path,
            )
            if not dry_run:
                store.mark_success(paper, target_path, "xml")
                return "success", None
            return "skipped", None

        logger.warning(
            "%s | Invalid existing XML at %s (%s); removing and retrying.",
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
                return "failed", None

    if dry_run:
        logger.info(
            "%s | [dry-run] would include DOI %s in a batch request",
            paper.paper_uid,
            paper.doi,
        )
        return "skipped", None

    return None, target_path


def store_downloaded_record(
    store: PaperStore,
    paper: Paper,
    target_path: Path,
    content: bytes,
    duration: Optional[float],
) -> str:
    """Atomically store and validate one record returned by a batch request."""
    tmp_path = target_path.with_name(target_path.name + ".part")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(content)
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

    is_valid, validation_message = validate_jats_xml(target_path, paper.doi)
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
        "%s | Download OK and XML validated (batch %.1fs) -> %s",
        paper.paper_uid,
        duration or 0.0,
        target_path,
    )
    store.mark_success(paper, target_path, "xml")
    return "success"


# --------------------------------------------------------------------------- #
# Shutdown and CLI
# --------------------------------------------------------------------------- #


_stop_requested = False


def _handle_stop_signal(signum: int, _frame: Any) -> None:
    global _stop_requested
    logger.warning(
        "Received signal %s, will stop after the in-flight paper finishes.", signum
    )
    _stop_requested = True


def rate_limit_seconds(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rate limit must be a number") from exc
    if not math.isfinite(interval) or interval < MIN_RATE_LIMIT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"rate limit must be a finite value greater than "
            f"{MIN_RATE_LIMIT_SECONDS:g} second"
        )
    return interval


def request_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("request limit must be an integer") from exc
    if not 1 <= limit <= MAX_REQUEST_LIMIT:
        raise argparse.ArgumentTypeError(
            f"request limit must be between 1 and {MAX_REQUEST_LIMIT}"
        )
    return limit


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download pending Springer Nature open-access JATS full texts "
            "from the central coordination database."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to the central coordination SQLite database.",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Name of the coordination table (default: {DEFAULT_TABLE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/fulltext/springer_nature"),
        help=(
            "Root directory for downloaded XML. A paper_uid subdirectory is "
            "created for each paper."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Springer Nature API key. Defaults to SPRINGER_NATURE_API_KEY."
        ),
    )
    parser.add_argument(
        "--rate-limit",
        type=rate_limit_seconds,
        default=DEFAULT_RATE_LIMIT_SECONDS,
        help=(
            f"Minimum seconds between API calls "
            f"(default: {DEFAULT_RATE_LIMIT_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pending papers to process in this run.",
    )
    parser.add_argument(
        "--request-limit",
        type=request_limit,
        default=DEFAULT_REQUEST_LIMIT,
        help=(
            f"Maximum API requests in this run (default: {DEFAULT_REQUEST_LIMIT}; "
            f"hard maximum: {MAX_REQUEST_LIMIT}). Each request can contain up to "
            f"{REQUEST_BATCH_SIZE} DOIs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List work without calling the API or changing the database.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    api_key = args.api_key or os.environ.get("SPRINGER_NATURE_API_KEY")
    if not api_key and not args.dry_run:
        logger.error(
            "No Springer Nature API key provided "
            "(use --api-key or set SPRINGER_NATURE_API_KEY)."
        )
        return 2

    if not args.db.exists():
        logger.error("Database not found at %s", args.db)
        return 2

    output_dir = args.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Could not create output directory %s: %s", output_dir, exc)
        return 2

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    client: Optional[SpringerNatureClient] = None
    if not args.dry_run:
        client = SpringerNatureClient(api_key=api_key, timeout=API_TIMEOUT_SECONDS)
    rate_limiter = RateLimiter(args.rate_limit)

    counts = {"success": 0, "failed": 0, "skipped": 0}
    requests_used = 0
    start = time.monotonic()

    try:
        with PaperStore(args.db, table=args.table) as store:
            papers = store.fetch_pending(limit=args.limit)
            logger.info(
                "Found %d pending paper(s) with canonical_source='%s'.%s",
                len(papers),
                SOURCE_NAME,
                " [dry-run]" if args.dry_run else "",
            )

            ready: list[tuple[Paper, Path]] = []
            for index, paper in enumerate(papers, start=1):
                if _stop_requested:
                    logger.warning(
                        "Stopping preflight before paper %d/%d (%s).",
                        index,
                        len(papers),
                        paper.paper_uid,
                    )
                    break

                logger.info(
                    "[%d/%d] Processing %s", index, len(papers), paper.paper_uid
                )
                try:
                    outcome, target_path = prepare_paper(
                        store,
                        paper,
                        output_dir,
                        dry_run=args.dry_run,
                    )
                except sqlite3.Error:
                    logger.exception(
                        "%s | DATABASE_ERROR: SQLite failed while processing this "
                        "paper; the failure cannot be reliably written to the database.",
                        paper.paper_uid,
                    )
                    raise
                except Exception as exc:  # noqa: BLE001 - continue the batch
                    message = (
                        f"Unhandled exception in downloader: {exc!r}; "
                        "see job log for traceback."
                    )
                    logger.exception("%s | %s", paper.paper_uid, message)
                    if not args.dry_run:
                        store.mark_failure(paper, DownloadStatus.OTHER_ERROR, message)
                    outcome = "failed"
                    target_path = None

                if outcome is not None:
                    counts[outcome] = counts.get(outcome, 0) + 1
                elif target_path is not None:
                    ready.append((paper, target_path))

            processed_ready = 0
            for batch_start in range(0, len(ready), REQUEST_BATCH_SIZE):
                if _stop_requested:
                    logger.warning(
                        "Stopping before the next batch due to shutdown signal."
                    )
                    break
                if requests_used >= args.request_limit:
                    logger.warning(
                        "API request limit reached (%d); leaving remaining papers pending.",
                        args.request_limit,
                    )
                    break

                batch = ready[batch_start : batch_start + REQUEST_BATCH_SIZE]
                batch_dois = [paper.doi for paper, _target in batch if paper.doi]
                request_number = requests_used + 1
                logger.info(
                    "Batch request %d/%d | papers=%d unique_dois=%d",
                    request_number,
                    args.request_limit,
                    len(batch),
                    len(set(batch_dois)),
                )

                rate_limiter.wait()
                requests_used += 1
                try:
                    result = client.download_batch(batch_dois)
                except Exception as exc:  # noqa: BLE001 - count and fail this batch
                    message = (
                        f"Unexpected exception calling SpringerNatureClient: {exc!r}"
                    )
                    logger.exception("Batch %d | %s", request_number, message)
                    for paper, _target_path in batch:
                        store.mark_failure(
                            paper, DownloadStatus.OTHER_ERROR, message
                        )
                        counts["failed"] += 1
                    processed_ready += len(batch)
                    continue

                if result.status not in _SUCCESS_STATUSES:
                    message = build_error_message(result)
                    logger.error("Batch %d failed: %s", request_number, message)
                    for paper, _target_path in batch:
                        store.mark_failure(paper, result.status, message)
                        counts["failed"] += 1
                    processed_ready += len(batch)
                    continue

                for paper, target_path in batch:
                    record = result.records.get(paper.doi.casefold())
                    if record is None:
                        missing_result = BatchDownloadResult(
                            status=DownloadStatus.UNKNOWN_DOI,
                            api_status=result.api_status,
                            comment=(
                                f"DOI {paper.doi!r} was not returned by the "
                                f"batch response | {result.comment}"
                            ),
                        )
                        message = build_error_message(missing_result)
                        logger.error("%s | %s", paper.paper_uid, message)
                        store.mark_failure(
                            paper, DownloadStatus.UNKNOWN_DOI, message
                        )
                        counts["failed"] += 1
                        continue

                    outcome = store_downloaded_record(
                        store,
                        paper,
                        target_path,
                        record,
                        result.duration,
                    )
                    counts[outcome] = counts.get(outcome, 0) + 1
                processed_ready += len(batch)

            remaining = len(ready) - processed_ready
            if remaining:
                logger.warning(
                    "%d download-ready paper(s) remain pending and unrequested.",
                    remaining,
                )
    finally:
        if client is not None:
            client.close()

    elapsed = time.monotonic() - start
    logger.info(
        "Done in %.1fs | requests=%d/%d success=%d skipped=%d failed=%d",
        elapsed,
        requests_used,
        args.request_limit,
        counts["success"],
        counts["skipped"],
        counts["failed"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
