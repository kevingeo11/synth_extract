#!/usr/bin/env python3
"""Extract pending S2ORC full text from the local filtered corpus.

The central coordination database supplies pending papers whose
``canonical_source`` is ``"s2orc"``. Each DOI is mapped to a S2ORC corpus ID
through ``papers_dup`` in the local S2ORC SQLite database. The compressed
JSONL corpus is then scanned once for all requested corpus IDs.

Each successful record is reduced to these fields:

``doi``, ``corpus_id``, ``title``, ``abstract``, and ``body``.

The body is taken from ``record["body"]["text"]``. Output is written
atomically to ``<output_dir>/<paper_uid>/<paper_uid>.json`` and the central
database is updated after every paper.

Usage
-----
    python -m synth_extract.mining.tdm.s2orc \
        --db data/central_papers.db \
        --s2orc-db data/s2orc.db \
        --corpus-file data/s2orc/s2orc_filtered_polymer.jsonl.gz \
        --output-dir data/fulltext/s2orc
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SOURCE_NAME = "s2orc"
DEFAULT_TABLE = "papers"
DEFAULT_S2ORC_TABLE = "papers_dup"
DEFAULT_S2ORC_DB = Path("data/s2orc.db")
DEFAULT_CORPUS_FILE = Path(
    "data/s2orc/s2orc_filtered_polymer.jsonl.gz"
)
DEFAULT_OUTPUT_DIR = Path("data/fulltext/s2orc")
DEFAULT_PROGRESS_EVERY = 10_000

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

logger = logging.getLogger("s2orc_downloader")


# --------------------------------------------------------------------------- #
# Logging and helpers
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


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json_list(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        logger.warning(
            "Could not parse JSON list %r, treating as empty.", raw
        )
        return []
    return value if isinstance(value, list) else []


def _parse_corpus_id(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


# --------------------------------------------------------------------------- #
# Output record validation
# --------------------------------------------------------------------------- #


def build_output_record(
    record: dict[str, Any],
    doi: str,
    expected_corpus_id: int,
) -> tuple[Optional[dict[str, Any]], str]:
    """Reduce one S2ORC record to the fields stored by this downloader."""
    corpus_id = _parse_corpus_id(record.get("corpusid"))
    if corpus_id != expected_corpus_id:
        return None, (
            f"record corpusid {record.get('corpusid')!r} does not match "
            f"expected corpus ID {expected_corpus_id}"
        )

    title = record.get("title")
    if title is not None and not isinstance(title, str):
        return None, "record title is neither text nor null"

    abstract = record.get("abstract")
    if abstract is not None and not isinstance(abstract, str):
        return None, "record abstract is neither text nor null"

    body = record.get("body")
    if not isinstance(body, dict):
        return None, "record body is not an object"
    body_text = body.get("text")
    if not isinstance(body_text, str) or not body_text.strip():
        return None, "record body.text is missing or empty"

    return {
        "doi": doi,
        "corpus_id": corpus_id,
        "title": title,
        "abstract": abstract,
        "body": body_text,
    }, "valid S2ORC record"


def validate_json_file(
    path: Path,
    expected_doi: Optional[str] = None,
    expected_corpus_id: Optional[int] = None,
) -> tuple[bool, str]:
    """Validate a stored reduced S2ORC JSON file."""
    try:
        if not path.exists():
            return False, "file does not exist"
        if not path.is_file():
            return False, "path is not a regular file"
        if path.stat().st_size == 0:
            return False, "file is empty"

        with path.open("r", encoding="utf-8") as input_file:
            record = json.load(input_file)
    except (OSError, UnicodeError) as exc:
        return False, f"could not read JSON file: {exc}"
    except json.JSONDecodeError as exc:
        return False, f"file is not valid JSON: {exc}"

    if not isinstance(record, dict):
        return False, "JSON root is not an object"

    required_fields = {"doi", "corpus_id", "title", "abstract", "body"}
    missing_fields = required_fields - record.keys()
    if missing_fields:
        return False, (
            "JSON is missing field(s): "
            + ", ".join(sorted(missing_fields))
        )

    doi = record.get("doi")
    if not isinstance(doi, str) or not doi.strip():
        return False, "JSON DOI is missing or empty"
    if (
        expected_doi is not None
        and doi.casefold() != expected_doi.casefold()
    ):
        return False, f"JSON DOI does not match expected DOI {expected_doi!r}"

    corpus_id = _parse_corpus_id(record.get("corpus_id"))
    if corpus_id is None:
        return False, "JSON corpus_id is missing or invalid"
    if (
        expected_corpus_id is not None
        and corpus_id != expected_corpus_id
    ):
        return False, (
            f"JSON corpus_id {corpus_id} does not match expected corpus ID "
            f"{expected_corpus_id}"
        )

    if record.get("title") is not None and not isinstance(
        record.get("title"), str
    ):
        return False, "JSON title is neither text nor null"
    if record.get("abstract") is not None and not isinstance(
        record.get("abstract"), str
    ):
        return False, "JSON abstract is neither text nor null"

    body = record.get("body")
    if not isinstance(body, str) or not body.strip():
        return False, "JSON body is missing or empty"

    return True, "valid reduced S2ORC JSON"


# --------------------------------------------------------------------------- #
# Status, data model, and central database
# --------------------------------------------------------------------------- #


class DownloadStatus(Enum):
    SUCCESS = "SUCCESS"
    INVALID_DOI = "INVALID_DOI"
    DOI_NOT_IN_LOOKUP = "DOI_NOT_IN_LOOKUP"
    CORPUS_ID_NOT_FOUND = "CORPUS_ID_NOT_FOUND"
    INVALID_CORPUS_RECORD = "INVALID_CORPUS_RECORD"
    STORAGE_ERROR = "STORAGE_ERROR"
    OTHER_ERROR = "OTHER_ERROR"


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
        if not _TABLE_NAME_RE.fullmatch(table):
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
            self.conn.close()
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
        self, paper: Paper, fulltext_path: Path
    ) -> None:
        now = utcnow_iso()
        attempted_sources = [*paper.attempted_sources, SOURCE_NAME]
        cursor = self.conn.execute(
            f"""
            UPDATE {self.table}
            SET download_status       = 'success',
                downloaded_from       = ?,
                fulltext_path         = ?,
                fulltext_format       = 'json',
                downloaded_at         = ?,
                last_attempted_source = ?,
                last_attempted_at     = ?,
                last_error            = NULL,
                updated_at            = ?,
                attempted_sources     = ?,
                attempt_count         = attempt_count + 1
            WHERE paper_uid = ?
              AND canonical_source = ?
              AND download_status = 'pending'
            """,
            (
                SOURCE_NAME,
                str(fulltext_path),
                now,
                SOURCE_NAME,
                now,
                now,
                json.dumps(attempted_sources),
                paper.paper_uid,
                SOURCE_NAME,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f"{paper.paper_uid}: success update affected "
                f"{cursor.rowcount} rows"
            )

    def mark_failure(
        self,
        paper: Paper,
        status: DownloadStatus,
        error_message: str,
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
        cursor = self.conn.execute(
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
              AND canonical_source = ?
              AND download_status = 'pending'
            """,
            (
                SOURCE_NAME,
                now,
                error_message,
                now,
                json.dumps(attempted_sources),
                json.dumps(failure_history),
                paper.paper_uid,
                SOURCE_NAME,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f"{paper.paper_uid}: failure update affected "
                f"{cursor.rowcount} rows"
            )


# --------------------------------------------------------------------------- #
# Local S2ORC lookup and storage
# --------------------------------------------------------------------------- #


def lookup_corpus_ids(
    db_path: Path,
    table: str,
    wanted_dois: set[str],
) -> dict[str, int]:
    """Scan the lookup table once and return requested DOI-to-corpus-ID rows."""
    if not _TABLE_NAME_RE.fullmatch(table):
        raise ValueError(f"Unsafe S2ORC table name: {table!r}")
    if not wanted_dois:
        return {}

    mapping: dict[str, int] = {}
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with closing(
        sqlite3.connect(uri, uri=True, timeout=30)
    ) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        rows = conn.execute(
            f"""
            SELECT corpusid, doi
            FROM {table}
            WHERE doi IS NOT NULL
            """
        )
        for raw_corpus_id, doi in rows:
            if doi not in wanted_dois:
                continue
            corpus_id = _parse_corpus_id(raw_corpus_id)
            if corpus_id is None:
                raise RuntimeError(
                    f"S2ORC lookup has an invalid corpusid for DOI {doi!r}"
                )
            if doi in mapping:
                raise RuntimeError(
                    f"S2ORC lookup contains duplicate DOI {doi!r}"
                )
            mapping[doi] = corpus_id

    return mapping


def write_output_record(
    store: PaperStore,
    paper: Paper,
    target_path: Path,
    output_record: dict[str, Any],
) -> str:
    """Atomically write, validate, and record one reduced S2ORC paper."""
    tmp_path = target_path.with_name(target_path.name + ".part")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                output_record,
                output_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            output_file.write("\n")
        os.replace(tmp_path, target_path)
    except OSError as exc:
        message = f"Could not write S2ORC JSON to {target_path}: {exc}"
        logger.exception("%s | %s", paper.paper_uid, message)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        store.mark_failure(
            paper, DownloadStatus.STORAGE_ERROR, message
        )
        return "failed"

    is_valid, validation_message = validate_json_file(
        target_path,
        expected_doi=output_record["doi"],
        expected_corpus_id=output_record["corpus_id"],
    )
    if not is_valid:
        message = (
            f"Stored file at {target_path} failed JSON validation: "
            f"{validation_message}"
        )
        logger.error("%s | %s", paper.paper_uid, message)
        try:
            target_path.unlink(missing_ok=True)
        except OSError as exc:
            message = (
                f"{message}; invalid file could not be removed: {exc}"
            )
        store.mark_failure(
            paper, DownloadStatus.INVALID_CORPUS_RECORD, message
        )
        return "failed"

    store.mark_success(paper, target_path)
    logger.info(
        "%s | Extracted corpus ID %s -> %s",
        paper.paper_uid,
        output_record["corpus_id"],
        target_path,
    )
    return "success"


# --------------------------------------------------------------------------- #
# Corpus scan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PendingTarget:
    paper: Paper
    target_path: Path
    corpus_id: int


@dataclass
class ScanResult:
    completed: bool
    line_count: int = 0
    invalid_json_lines: int = 0
    invalid_corpus_id_lines: int = 0
    outcomes: dict[str, int] = field(
        default_factory=lambda: {"success": 0, "failed": 0}
    )
    invalid_target_records: dict[int, str] = field(default_factory=dict)
    error: Optional[str] = None


def scan_corpus(
    corpus_file: Path,
    targets: dict[int, list[PendingTarget]],
    store: PaperStore,
    progress_every: int,
) -> ScanResult:
    """Scan the gzip JSONL file once, extracting all requested corpus IDs."""
    result = ScanResult(completed=False)
    start = time.monotonic()

    try:
        with gzip.open(corpus_file, "rt", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                result.line_count = line_number
                if _stop_requested:
                    logger.warning(
                        "Stopping corpus scan after line %d; %d corpus ID(s) "
                        "remain pending.",
                        line_number - 1,
                        len(targets),
                    )
                    return result

                if progress_every and line_number % progress_every == 0:
                    logger.info(
                        "Corpus scan | lines=%d remaining_ids=%d "
                        "success=%d failed=%d elapsed=%.1fs",
                        line_number,
                        len(targets),
                        result.outcomes["success"],
                        result.outcomes["failed"],
                        time.monotonic() - start,
                    )

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    result.invalid_json_lines += 1
                    if result.invalid_json_lines <= 10:
                        logger.warning(
                            "Invalid JSON at corpus line %d: %s",
                            line_number,
                            exc,
                        )
                    continue

                if not isinstance(record, dict):
                    result.invalid_json_lines += 1
                    continue

                corpus_id = _parse_corpus_id(record.get("corpusid"))
                if corpus_id is None:
                    result.invalid_corpus_id_lines += 1
                    continue
                if corpus_id not in targets:
                    continue

                target_group = targets[corpus_id]
                first_target = target_group[0]
                output_record, validation_message = build_output_record(
                    record,
                    first_target.paper.doi or "",
                    corpus_id,
                )
                if output_record is None:
                    result.invalid_target_records[corpus_id] = (
                        f"Corpus ID {corpus_id} at line {line_number} is "
                        f"invalid: {validation_message}"
                    )
                    logger.error(
                        "Corpus ID %s | %s",
                        corpus_id,
                        validation_message,
                    )
                    continue

                # A corpus ID should map to one DOI, but support multiple
                # central rows defensively by changing the DOI per output.
                for target in target_group:
                    paper_record = dict(output_record)
                    paper_record["doi"] = target.paper.doi
                    outcome = write_output_record(
                        store,
                        target.paper,
                        target.target_path,
                        paper_record,
                    )
                    result.outcomes[outcome] += 1

                del targets[corpus_id]
                result.invalid_target_records.pop(corpus_id, None)
                if not targets:
                    result.completed = True
                    logger.info(
                        "All requested corpus IDs found after %d line(s).",
                        line_number,
                    )
                    return result
    except (OSError, EOFError, UnicodeError) as exc:
        result.error = f"Could not complete corpus scan: {exc}"
        logger.exception(result.error)
        return result

    result.completed = True
    return result


# --------------------------------------------------------------------------- #
# Graceful shutdown and CLI
# --------------------------------------------------------------------------- #

_stop_requested = False


def _handle_stop_signal(signum: int, _frame: Any) -> None:
    global _stop_requested
    logger.warning(
        "Received signal %s; will stop after the current corpus line.",
        signum,
    )
    _stop_requested = True


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be an integer"
        ) from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pending S2ORC full text from a local compressed JSONL "
            "corpus."
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
        help=f"Central table name (default: {DEFAULT_TABLE}).",
    )
    parser.add_argument(
        "--s2orc-db",
        type=Path,
        default=DEFAULT_S2ORC_DB,
        help=f"S2ORC lookup database (default: {DEFAULT_S2ORC_DB}).",
    )
    parser.add_argument(
        "--s2orc-table",
        default=DEFAULT_S2ORC_TABLE,
        help=(
            f"S2ORC DOI lookup table "
            f"(default: {DEFAULT_S2ORC_TABLE})."
        ),
    )
    parser.add_argument(
        "--corpus-file",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help=(
            "Filtered S2ORC gzip JSONL corpus "
            f"(default: {DEFAULT_CORPUS_FILE})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output root containing one paper_uid directory per paper "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum number of pending central rows to process.",
    )
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=DEFAULT_PROGRESS_EVERY,
        help=(
            "Log scan progress after this many JSONL lines "
            f"(default: {DEFAULT_PROGRESS_EVERY:,})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve DOI mappings and inspect existing files without "
            "writing files, scanning the corpus, or changing the database."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main control flow
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    for path, label in (
        (args.db, "central database"),
        (args.s2orc_db, "S2ORC lookup database"),
        (args.corpus_file, "S2ORC corpus file"),
    ):
        if not path.is_file():
            logger.error("%s not found at %s", label.capitalize(), path)
            return 2

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "Could not create output directory %s: %s",
            args.output_dir,
            exc,
        )
        return 2

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

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
            if not papers:
                return 0

            wanted_dois = {
                paper.doi.strip()
                for paper in papers
                if paper.doi is not None and paper.doi.strip()
            }
            doi_to_corpus_id = lookup_corpus_ids(
                args.s2orc_db,
                args.s2orc_table,
                wanted_dois,
            )
            logger.info(
                "Resolved %d/%d requested DOI(s) to S2ORC corpus IDs.",
                len(doi_to_corpus_id),
                len(wanted_dois),
            )

            targets: dict[int, list[PendingTarget]] = {}
            for index, paper in enumerate(papers, start=1):
                logger.info(
                    "[%d/%d] Preparing %s",
                    index,
                    len(papers),
                    paper.paper_uid,
                )

                if not paper.doi or not paper.doi.strip():
                    message = (
                        "canonical_source is 's2orc' but this paper has no "
                        "DOI."
                    )
                    logger.error("%s | %s", paper.paper_uid, message)
                    if not args.dry_run:
                        store.mark_failure(
                            paper, DownloadStatus.INVALID_DOI, message
                        )
                    counts["failed"] += 1
                    continue

                doi = paper.doi.strip()
                corpus_id = doi_to_corpus_id.get(doi)
                if corpus_id is None:
                    message = (
                        f"DOI {doi!r} is not present in "
                        f"{args.s2orc_db}:{args.s2orc_table}."
                    )
                    logger.error("%s | %s", paper.paper_uid, message)
                    if not args.dry_run:
                        store.mark_failure(
                            paper,
                            DownloadStatus.DOI_NOT_IN_LOOKUP,
                            message,
                        )
                    counts["failed"] += 1
                    continue

                target_path = (
                    args.output_dir
                    / paper.paper_uid
                    / f"{paper.paper_uid}.json"
                )
                if target_path.exists():
                    is_valid, validation_message = validate_json_file(
                        target_path,
                        expected_doi=doi,
                        expected_corpus_id=corpus_id,
                    )
                    if is_valid:
                        logger.info(
                            "%s | Valid JSON already exists at %s; "
                            "reconciling database to success.",
                            paper.paper_uid,
                            target_path,
                        )
                        if not args.dry_run:
                            store.mark_success(paper, target_path)
                            counts["success"] += 1
                        else:
                            counts["skipped"] += 1
                        continue

                    logger.warning(
                        "%s | Invalid existing JSON at %s (%s); "
                        "removing before corpus extraction.",
                        paper.paper_uid,
                        target_path,
                        validation_message,
                    )
                    if not args.dry_run:
                        try:
                            target_path.unlink()
                        except OSError as exc:
                            message = (
                                f"Invalid existing JSON at {target_path} "
                                f"could not be removed: {exc}"
                            )
                            logger.error(
                                "%s | %s", paper.paper_uid, message
                            )
                            store.mark_failure(
                                paper,
                                DownloadStatus.STORAGE_ERROR,
                                message,
                            )
                            counts["failed"] += 1
                            continue

                if args.dry_run:
                    logger.info(
                        "%s | [dry-run] would search corpus ID %s for DOI %s",
                        paper.paper_uid,
                        corpus_id,
                        doi,
                    )
                    counts["skipped"] += 1
                    continue

                targets.setdefault(corpus_id, []).append(
                    PendingTarget(
                        paper=paper,
                        target_path=target_path,
                        corpus_id=corpus_id,
                    )
                )

            if args.dry_run or not targets:
                scan_result = None
            else:
                logger.info(
                    "Scanning %s once for %d corpus ID(s) covering %d "
                    "paper(s).",
                    args.corpus_file,
                    len(targets),
                    sum(len(group) for group in targets.values()),
                )
                scan_result = scan_corpus(
                    args.corpus_file,
                    targets,
                    store,
                    args.progress_every,
                )
                counts["success"] += scan_result.outcomes["success"]
                counts["failed"] += scan_result.outcomes["failed"]

                if scan_result.completed:
                    for corpus_id, target_group in list(targets.items()):
                        invalid_message = (
                            scan_result.invalid_target_records.get(corpus_id)
                        )
                        if invalid_message:
                            status = DownloadStatus.INVALID_CORPUS_RECORD
                            message = invalid_message
                        else:
                            status = DownloadStatus.CORPUS_ID_NOT_FOUND
                            message = (
                                f"Corpus ID {corpus_id} was mapped from the "
                                "DOI lookup but was not found in "
                                f"{args.corpus_file}."
                            )

                        for target in target_group:
                            logger.error(
                                "%s | %s",
                                target.paper.paper_uid,
                                message,
                            )
                            store.mark_failure(
                                target.paper, status, message
                            )
                            counts["failed"] += 1
                    targets.clear()
                else:
                    remaining_papers = sum(
                        len(group) for group in targets.values()
                    )
                    logger.warning(
                        "Corpus scan did not complete; leaving %d unmatched "
                        "paper(s) pending.",
                        remaining_papers,
                    )

                logger.info(
                    "Corpus scan summary | completed=%s lines=%d "
                    "invalid_json=%d invalid_corpus_id=%d",
                    scan_result.completed,
                    scan_result.line_count,
                    scan_result.invalid_json_lines,
                    scan_result.invalid_corpus_id_lines,
                )
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        logger.exception("Fatal S2ORC downloader error: %s", exc)
        return 1

    elapsed = time.monotonic() - start
    logger.info(
        "Done in %.1fs | success=%d skipped=%d failed=%d",
        elapsed,
        counts["success"],
        counts["skipped"],
        counts["failed"],
    )
    if scan_result is not None and scan_result.error:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
