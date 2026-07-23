#!/usr/bin/env python3
"""Europe PMC open-access full-text XML downloader.

Reads pending papers whose ``canonical_source`` is ``"europepmc"`` from the
central SQLite coordination database, retrieves one article at a time by
PMCID, and writes every result back to the same database row immediately.

The downloader only attempts the current canonical source. It does not move
failed rows to another source; that belongs to the multi-source workflow
orchestrator.

Europe PMC documents ``/{id}/fullTextXML`` as providing JATS XML for the open
access full-text subset. No API key is required.

Usage
-----
    python -m synth_extract.mining.tdm.europepmc \
        --db /path/to/central.sqlite \
        --output-dir /path/to/data/fulltext/europepmc

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
from urllib.parse import quote

import requests
from lxml import etree

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SOURCE_NAME = "europepmc"
DEFAULT_TABLE = "papers"
MIN_RATE_LIMIT_SECONDS = 0.15
DEFAULT_RATE_LIMIT_SECONDS = 0.15
API_TIMEOUT_SECONDS = 60
API_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

logger = logging.getLogger("europepmc_downloader")


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


def _article_pmcids(root: "etree._Element") -> set[str]:
    matches = root.xpath(
        "//*[local-name()='article-meta']/*[local-name()='article-id' "
        "and (@pub-id-type='pmcid' or @pub-id-type='pmc')]"
    )
    return {
        "".join(element.itertext()).strip().casefold()
        for element in matches
        if "".join(element.itertext()).strip()
    }


def validate_xml(
    path: Path, expected_pmcid: Optional[str] = None
) -> tuple[bool, str]:
    """Check that a file is plausible JATS XML for the expected PMCID."""
    try:
        if not path.exists():
            return False, "file does not exist"
        if not path.is_file():
            return False, "path is not a regular file"
        if path.stat().st_size == 0:
            return False, "file is empty"

        parser = etree.XMLParser(
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
            huge_tree=True,
        )
        try:
            tree = etree.parse(str(path), parser)
        except (etree.XMLSyntaxError, ValueError) as exc:
            return False, f"not well-formed XML: {exc}"

        root = tree.getroot()
        if root is None:
            return False, "XML has no root element"
        if _local_name(root) != "article":
            return False, f"unexpected root element <{_local_name(root)}>"

        pmcids = _article_pmcids(root)
        if not pmcids:
            return False, "JATS article contains no PMCID"
        if (
            expected_pmcid is not None
            and expected_pmcid.strip().casefold() not in pmcids
        ):
            return False, (
                f"file PMCID does not match expected PMCID "
                f"{expected_pmcid!r}"
            )
    except OSError as exc:
        return False, f"could not inspect file: {exc}"
    except Exception as exc:
        return False, f"could not validate Europe PMC XML: {exc}"

    return True, "valid Europe PMC JATS XML"


class RateLimiter:
    """Guarantee a minimum interval between successive request start times."""

    def __init__(self, interval: float):
        self.interval = interval
        self._last_call: Optional[float] = None

    def wait(self) -> None:
        if self._last_call is not None:
            remaining = self.interval - (time.monotonic() - self._last_call)
            if remaining > 0:
                logger.debug(
                    "Rate limit: sleeping %.3fs before next request.", remaining
                )
                time.sleep(remaining)
        self._last_call = time.monotonic()


# --------------------------------------------------------------------------- #
# Europe PMC client
# --------------------------------------------------------------------------- #


class DownloadStatus(Enum):
    """Outcome of one Europe PMC full-text retrieval attempt."""

    SUCCESS = "SUCCESS"
    ACCESS_DENIED = "ACCESS_DENIED"
    UNKNOWN_PMCID = "UNKNOWN_PMCID"
    RATE_LIMITED = "RATE_LIMITED"
    API_ERROR = "API_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    INVALID_PMCID = "INVALID_PMCID"
    KNOWN_ISSUE = "KNOWN_ISSUE"
    OTHER_ERROR = "OTHER_ERROR"


_SUCCESS_STATUSES = {DownloadStatus.SUCCESS}

_ERROR_HINTS: dict[DownloadStatus, str] = {
    DownloadStatus.ACCESS_DENIED: (
        "Europe PMC denied the request (HTTP 401/403). The fullTextXML "
        "endpoint requires no API key, so this may be a service-side policy "
        "or access problem."
    ),
    DownloadStatus.UNKNOWN_PMCID: (
        "Europe PMC did not return open-access full-text XML for this PMCID. "
        "The PMCID may be unknown, removed, or outside the open-access "
        "full-text subset."
    ),
    DownloadStatus.RATE_LIMITED: (
        "Europe PMC rate-limited the request. Retry later or increase "
        "--rate-limit."
    ),
    DownloadStatus.API_ERROR: (
        "Europe PMC returned an unexpected HTTP error. HTTP 5xx indicates "
        "a server-side problem."
    ),
    DownloadStatus.NETWORK_ERROR: (
        "Network/connection error while calling Europe PMC."
    ),
    DownloadStatus.STORAGE_ERROR: "Could not write the XML to local disk.",
    DownloadStatus.INVALID_PMCID: "The database row has no usable PMCID.",
    DownloadStatus.KNOWN_ISSUE: (
        "Europe PMC returned HTTP 200, but the downloaded body was not valid "
        "JATS XML for the requested PMCID."
    ),
    DownloadStatus.OTHER_ERROR: "Unexpected error in the Europe PMC downloader.",
}


@dataclass
class DownloadResult:
    status: DownloadStatus
    api_status: Optional[int] = None
    comment: Optional[str] = None
    duration: Optional[float] = None
    content: Optional[bytes] = None


class EuropePMCClient:
    """Minimal client for Europe PMC's PMCID-based fullTextXML endpoint."""

    _STATUS_MAP = {
        400: DownloadStatus.API_ERROR,
        401: DownloadStatus.ACCESS_DENIED,
        403: DownloadStatus.ACCESS_DENIED,
        404: DownloadStatus.UNKNOWN_PMCID,
        410: DownloadStatus.UNKNOWN_PMCID,
        429: DownloadStatus.RATE_LIMITED,
    }

    def __init__(self, timeout: int = API_TIMEOUT_SECONDS):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/xml, text/xml",
                "User-Agent": "synth-extract-europepmc-downloader/1.0",
            }
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "EuropePMCClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def download_xml(self, pmcid: str) -> DownloadResult:
        encoded_pmcid = quote(pmcid, safe="")
        url = f"{API_BASE_URL}/{encoded_pmcid}/fullTextXML"
        start = time.monotonic()
        try:
            response = self.session.get(url, timeout=self.timeout)
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

    def _classify_response(
        self, response: requests.Response, duration: float
    ) -> DownloadResult:
        status_code = response.status_code
        if status_code == 200:
            return DownloadResult(
                status=DownloadStatus.SUCCESS,
                api_status=200,
                comment=response.headers.get("Content-Type"),
                duration=duration,
                content=response.content,
            )

        status = self._STATUS_MAP.get(status_code, DownloadStatus.API_ERROR)
        return DownloadResult(
            status=status,
            api_status=status_code,
            comment=self._error_comment(response),
            duration=duration,
        )

    @staticmethod
    def _error_comment(response: requests.Response) -> str:
        content_type = response.headers.get("Content-Type")
        snippet = response.text.strip().replace("\n", " ")
        body = snippet[:500] if snippet else "(no response body)"
        if content_type:
            return f"Content-Type={content_type} | {body}"
        return body


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
# Data model and database layer
# --------------------------------------------------------------------------- #


@dataclass
class Paper:
    paper_uid: str
    pmcid: Optional[str]
    canonical_source: str
    attempt_count: int
    attempted_sources: list = field(default_factory=list)
    failure_history: list = field(default_factory=list)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Paper":
        return cls(
            paper_uid=row["paper_uid"],
            pmcid=row["pmcid"],
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
        self.conn = sqlite3.connect(
            str(db_path), timeout=30, isolation_level=None
        )
        self.conn.row_factory = sqlite3.Row
        journal_mode = self.conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
        if journal_mode != "delete":
            raise RuntimeError(
                f"Expected SQLite journal_mode='delete', found "
                f"{journal_mode!r}"
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
            f"SELECT paper_uid, pmcid, canonical_source, attempt_count, "
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


def process_paper(
    client: Optional[EuropePMCClient],
    store: PaperStore,
    paper: Paper,
    output_dir: Path,
    rate_limiter: RateLimiter,
    dry_run: bool = False,
) -> str:
    """Handle one paper end to end."""
    if not paper.pmcid or not paper.pmcid.strip():
        message = (
            "canonical_source is 'europepmc' but this paper has no PMCID."
        )
        logger.error("%s | %s", paper.paper_uid, message)
        if not dry_run:
            store.mark_failure(
                paper, DownloadStatus.INVALID_PMCID, message
            )
        return "failed"

    pmcid = paper.pmcid.strip()
    paper_dir = output_dir / paper.paper_uid
    target_path = paper_dir / f"{paper.paper_uid}.xml"

    # Reconcile a valid file left by a kill between atomic rename and the
    # database update. Remove an invalid final file and retry normally.
    if target_path.exists():
        is_valid, validation_message = validate_xml(target_path, pmcid)
        if is_valid:
            logger.info(
                "%s | Valid XML already exists at %s; reconciling database "
                "to success.",
                paper.paper_uid,
                target_path,
            )
            if not dry_run:
                store.mark_success(paper, target_path, "xml")
                return "success"
            return "skipped"

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
                    f"Invalid existing XML at {target_path} "
                    f"({validation_message}) could not be removed: {exc}"
                )
                logger.error("%s | %s", paper.paper_uid, message)
                store.mark_failure(
                    paper, DownloadStatus.STORAGE_ERROR, message
                )
                return "failed"

    if dry_run:
        logger.info(
            "%s | [dry-run] would download PMCID %s",
            paper.paper_uid,
            pmcid,
        )
        return "skipped"

    logger.info(
        "%s | Requesting full text for PMCID %s", paper.paper_uid, pmcid
    )
    rate_limiter.wait()
    try:
        result = client.download_xml(pmcid)
    except Exception as exc:
        message = (
            f"Unexpected exception calling EuropePMCClient.download_xml: "
            f"{exc!r}"
        )
        logger.exception("%s | %s", paper.paper_uid, message)
        store.mark_failure(paper, DownloadStatus.OTHER_ERROR, message)
        return "failed"

    if result.status not in _SUCCESS_STATUSES:
        message = build_error_message(result)
        logger.error("%s | Download failed: %s", paper.paper_uid, message)
        store.mark_failure(paper, result.status, message)
        return "failed"

    tmp_path = target_path.with_name(target_path.name + ".part")
    try:
        paper_dir.mkdir(parents=True, exist_ok=True)
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

    is_valid, validation_message = validate_xml(target_path, pmcid)
    if not is_valid:
        message = (
            f"Downloaded file at {target_path} failed XML validation: "
            f"{validation_message}"
        )
        logger.error("%s | %s", paper.paper_uid, message)
        try:
            target_path.unlink(missing_ok=True)
        except OSError as exc:
            message = (
                f"{message}; invalid file could not be removed: {exc}"
            )
            logger.exception(
                "%s | Could not remove invalid XML.", paper.paper_uid
            )
        store.mark_failure(paper, DownloadStatus.KNOWN_ISSUE, message)
        return "failed"

    logger.info(
        "%s | Download OK and XML validated (%.1fs) -> %s",
        paper.paper_uid,
        result.duration or 0.0,
        target_path,
    )
    store.mark_success(paper, target_path, "xml")
    return "success"


# --------------------------------------------------------------------------- #
# Graceful shutdown
# --------------------------------------------------------------------------- #

_stop_requested = False


def _handle_stop_signal(signum: int, _frame: Any) -> None:
    global _stop_requested
    logger.warning(
        "Received signal %s, will stop after the in-flight paper finishes.",
        signum,
    )
    _stop_requested = True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def rate_limit_seconds(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rate limit must be a number"
        ) from exc
    if (
        not math.isfinite(interval)
        or interval < MIN_RATE_LIMIT_SECONDS
    ):
        raise argparse.ArgumentTypeError(
            f"rate limit must be a finite value of at least "
            f"{MIN_RATE_LIMIT_SECONDS:g} seconds"
        )
    return interval


def positive_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "limit must be an integer"
        ) from exc
    if limit < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return limit


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download pending Europe PMC open-access full-text XML from the "
            "central coordination database."
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
        default=Path("../data/fulltext/europepmc"),
        help=(
            "Root directory for downloaded XML. A paper_uid subdirectory is "
            "created for each successful download."
        ),
    )
    parser.add_argument(
        "--rate-limit",
        type=rate_limit_seconds,
        default=DEFAULT_RATE_LIMIT_SECONDS,
        help=(
            f"Minimum seconds between Europe PMC requests "
            f"(default and minimum: {DEFAULT_RATE_LIMIT_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=positive_limit,
        default=None,
        help="Maximum number of pending papers to process in this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List work without calling Europe PMC or changing the database."
        ),
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

    if not args.db.exists():
        logger.error("Database not found at %s", args.db)
        return 2

    output_dir = args.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "Could not create output directory %s: %s", output_dir, exc
        )
        return 2

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    client: Optional[EuropePMCClient] = None
    if not args.dry_run:
        client = EuropePMCClient(timeout=API_TIMEOUT_SECONDS)
    rate_limiter = RateLimiter(args.rate_limit)

    counts = {"success": 0, "failed": 0, "skipped": 0}
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

            for index, paper in enumerate(papers, start=1):
                if _stop_requested:
                    logger.warning(
                        "Stopping before paper %d/%d (%s) due to shutdown "
                        "signal.",
                        index,
                        len(papers),
                        paper.paper_uid,
                    )
                    break

                logger.info(
                    "[%d/%d] Processing %s",
                    index,
                    len(papers),
                    paper.paper_uid,
                )
                try:
                    outcome = process_paper(
                        client,
                        store,
                        paper,
                        output_dir,
                        rate_limiter,
                        dry_run=args.dry_run,
                    )
                except sqlite3.Error:
                    logger.exception(
                        "%s | DATABASE_ERROR: SQLite failed while processing "
                        "this paper; the failure cannot be reliably written "
                        "to the database.",
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
                            paper, DownloadStatus.OTHER_ERROR, message
                        )
                    outcome = "failed"
                counts[outcome] = counts.get(outcome, 0) + 1
    finally:
        if client is not None:
            client.close()

    elapsed = time.monotonic() - start
    logger.info(
        "Done in %.1fs | success=%d skipped=%d failed=%d",
        elapsed,
        counts["success"],
        counts["skipped"],
        counts["failed"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
