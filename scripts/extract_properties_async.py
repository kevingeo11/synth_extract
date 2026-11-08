#!/usr/bin/env python3
"""Extract polymer properties for pending (paper, property) pairs into ``polymer_samples.db``.

Only paper UIDs listed in the required manifest are considered. For each
manifest paper, every property in ``--properties`` that property presence
flags for the paper is one work item: a row of ``property_extraction_progress``
keyed by (paper_uid, property). Missing progress rows are added at start.

Papers are claimed in batches of ``--batch-size`` papers, together with all
their eligible properties, in short transactions. Each (paper, property) pair
runs with prompt 2 (``--mode p2``, per paper, values linked to samples) or
prompt 3 (``--mode p3``, one request per sample); one shared limit of
``--max-parallel-requests`` applies to every model request of the job.

A pair succeeds only when every request of it succeeds (all or nothing). For a
successful pair, its rows in ``properties`` are replaced and its progress row is
marked ``success`` with the prompt, prompt hash, model, counts, and tokens, all
in one transaction per batch. Values that match no sample are counted
(``unlinked``) but not stored. A failed pair keeps any previously stored rows and
is marked ``failed`` with an error message.

Example
-------
python scripts/extract_properties_async.py \\
    --uid-file data/uid_manifest/property_extraction_uids_0001.txt \\
    --mode p2 \\
    --model qwen3.8-27b \\
    --base-url http://127.0.0.1:8000/v1 \\
    --api-key none \\
    --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synth_extract.agents.extraction import (  # noqa: E402
    PaperContext,
    PaperPropertyExtractor,
    PaperPropertyResult,
    SamplePropertyExtractor,
    SampleRef,
)
from synth_extract.agents.extraction.property_config import PROPERTY_CONFIG  # noqa: E402
from synth_extract.agents.extraction.property_prompts import (  # noqa: E402
    DEFAULT_PROPERTY_PROMPT_DIR,
)
from synth_extract.agents.llm import LLMBackend  # noqa: E402


LOG = logging.getLogger("property_extraction_async")
PAPERS_TABLE = "papers"
SAMPLES_TABLE = "samples"
CLASSIFICATION_TABLE = "property_classification"
PROPERTIES_TABLE = "properties"
PROGRESS_TABLE = "property_extraction_progress"

MODES = {"p2": "per_paper", "p3": "with_context"}
EXTRACTOR_CLASSES = {"p2": PaperPropertyExtractor, "p3": SamplePropertyExtractor}
PROPERTY_COLUMNS = [
    "sample_id", "paper_uid", "property", "method", "subtype", "value_raw", "value",
    "value_type", "unit", "conditions", "origin", "description", "flags",
]
MAX_ERROR_CHARS = 4000


@dataclass(frozen=True, slots=True)
class PaperRow:
    paper_uid: str
    canonical_source: str
    fulltext_path: str


@dataclass(frozen=True, slots=True)
class PairClaim:
    paper: PaperRow
    property: str
    previous_status: str


@dataclass(slots=True)
class PairItem:
    claim: PairClaim
    result: PaperPropertyResult | None
    latency_seconds: float
    local_error: str | None = None
    markdown_missing: bool = False

    @property
    def succeeded(self) -> bool:
        return self.local_error is None and self.result is not None and self.result.complete


@dataclass(slots=True)
class Progress:
    eligible_at_start: int = 0
    eligible_papers_at_start: int = 0
    claimed: int = 0
    claimed_papers: int = 0
    attempted: int = 0
    successful_pairs: int = 0
    failed_pairs: int = 0
    inserted_properties: int = 0
    unlinked_values: int = 0
    zero_record_pairs: int = 0
    missing_markdown: int = 0
    database_errors: int = 0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---- helpers copied from scripts/extract_samples_async.py ----------------------------------


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def safe_path_component(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
    )


def _uids_from_json_value(value: Any, source: str) -> list[str]:
    if isinstance(value, dict):
        if "uids" not in value:
            raise ValueError(f"{source} must contain a top-level 'uids' array")
        value = value["uids"]
    if not isinstance(value, list):
        raise ValueError(f"{source} must contain a JSON array of UIDs")
    if not all(isinstance(uid, str) for uid in value):
        raise ValueError(f"Every UID in {source} must be a string")
    return value


def load_uid_file(uid_file: Path) -> list[str]:
    """Load and strictly validate a text, JSON, or JSONL UID manifest."""
    uid_file = uid_file.expanduser().resolve()
    if not uid_file.is_file():
        raise FileNotFoundError(f"UID file does not exist: {uid_file}")

    if uid_file.suffix.lower() == ".json":
        with uid_file.open(encoding="utf-8-sig") as handle:
            uids = _uids_from_json_value(json.load(handle), str(uid_file))
    elif uid_file.suffix.lower() == ".jsonl":
        uids = []
        with uid_file.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, str):
                    uid = value
                elif isinstance(value, dict) and isinstance(
                    value.get("paper_uid"), str
                ):
                    uid = value["paper_uid"]
                else:
                    raise ValueError(
                        f"Invalid JSONL UID at {uid_file}:{line_number}"
                    )
                uids.append(uid)
    else:
        with uid_file.open(encoding="utf-8-sig") as handle:
            uids = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]

    uids = [uid.strip() for uid in uids]
    if not uids:
        raise ValueError(f"UID file is empty: {uid_file}")
    invalid = [uid for uid in uids if not safe_path_component(uid)]
    if invalid:
        raise ValueError(f"Invalid UID path component(s): {invalid[:5]}")
    duplicates = sorted(uid for uid, count in Counter(uids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate UID(s) in manifest: {duplicates[:5]}")
    return uids


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(
        level=getattr(logging, level), handlers=[handler], force=True
    )


def required_columns(
    connection: sqlite3.Connection,
    table: str,
    required: set[str],
) -> None:
    columns = {
        str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')
    }
    if not columns:
        raise RuntimeError(f"Required table {table!r} does not exist")
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(
            f"Table {table!r} is missing columns: {', '.join(missing)}"
        )


def retryable_sqlite_lock_error(exc: sqlite3.Error) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        primary_code = error_code & 0xFF
        if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


def sqlite_retry_delay(args: argparse.Namespace, attempt: int) -> float:
    maximum = min(
        args.sqlite_retry_max_delay,
        args.sqlite_retry_base_delay * (2**attempt),
    )
    return maximum * random.uniform(0.8, 1.2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract polymer properties for pending (paper, property) pairs and "
            "populate the properties and property_extraction_progress tables."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=REPO_ROOT / "data" / "polymer_samples.db",
        help="SQLite database (default: %(default)s)",
    )
    parser.add_argument(
        "--fulltext-root",
        type=Path,
        default=REPO_ROOT / "data" / "fulltext",
        help="Fallback root containing source/paper_uid/paper_uid.md",
    )
    parser.add_argument(
        "--uid-file",
        type=Path,
        required=True,
        help="UID manifest: one UID per line, JSON, or JSONL",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        required=True,
        help="p2: per paper, values linked to samples; p3: one request per sample",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        choices=list(PROPERTY_CONFIG),
        default=list(PROPERTY_CONFIG),
        metavar="PROPERTY",
        help="Properties to extract (default: all configured: %(default)s)",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=None,
        help=f"Optional property prompt template folder (default: {DEFAULT_PROPERTY_PROMPT_DIR})",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID exposed by the OpenAI-compatible server",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default="none",
        help="API key or placeholder for servers without authentication",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help=(
            "Maximum papers to attempt once during this run, each with all its "
            "eligible properties (default: all)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=5,
        help=(
            "Papers claimed and committed per batch, with all their eligible "
            "properties (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--p2-batch-size",
        type=positive_int,
        default=15,
        help="Prompt 2 only: target samples per request (default: %(default)s)",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=positive_int,
        default=int(os.getenv("MAX_PARALLEL_REQUESTS", "4")),
        help="Maximum in-flight model requests for the whole job (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=1800.0,
        help="Model request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=40000,
        help="Maximum completion tokens per request (default: %(default)s)",
    )
    parser.add_argument(
        "--extra-body",
        type=json_object,
        default=None,
        help="Optional JSON object forwarded as OpenAI extra_body",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also attempt pairs currently marked failed",
    )
    parser.add_argument(
        "--retry-in-progress",
        action="store_true",
        help=(
            "Also reclaim pairs left in_progress by an interrupted run; do "
            "not use while another job is processing the same manifest"
        ),
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the endpoint model-list check",
    )
    parser.add_argument(
        "--sqlite-timeout",
        type=positive_float,
        default=60.0,
        help="SQLite lock timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--sqlite-write-retries",
        type=nonnegative_int,
        default=5,
        help="Retries for locked SQLite writes (default: %(default)s)",
    )
    parser.add_argument(
        "--sqlite-retry-base-delay",
        type=positive_float,
        default=1.0,
        help="Initial SQLite retry delay in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--sqlite-retry-max-delay",
        type=positive_float,
        default=30.0,
        help="Maximum SQLite retry delay in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    args.properties = list(dict.fromkeys(args.properties))  # unique, in the given order
    return args


def validate_database(connection: sqlite3.Connection) -> None:
    """Validate the exact tables and columns needed by this worker."""
    required_columns(
        connection,
        PAPERS_TABLE,
        {"paper_uid", "canonical_source", "fulltext_path"},
    )
    required_columns(
        connection,
        SAMPLES_TABLE,
        {"sample_id", "paper_uid", "name", "identifier", "aliases", "description"},
    )
    required_columns(
        connection,
        CLASSIFICATION_TABLE,
        {"paper_uid", "status", "properties_json"},
    )
    required_columns(connection, PROPERTIES_TABLE, set(PROPERTY_COLUMNS))
    required_columns(
        connection,
        PROGRESS_TABLE,
        {
            "paper_uid", "property", "status", "attempt_count", "mode", "prompt_hash",
            "model", "requests", "records", "unlinked", "prompt_tokens",
            "completion_tokens", "last_error", "started_at", "completed_at", "updated_at",
        },
    )


def run_with_write_retries(connection, args, description, work):
    """Run ``work(connection)`` in one IMMEDIATE transaction, retrying SQLite locks."""
    for attempt in range(args.sqlite_write_retries + 1):
        try:
            connection.execute("BEGIN IMMEDIATE")
            value = work(connection)
            connection.commit()
            return value
        except sqlite3.Error as exc:
            connection.rollback()
            if retryable_sqlite_lock_error(exc) and attempt < args.sqlite_write_retries:
                delay = sqlite_retry_delay(args, attempt)
                LOG.warning(
                    "SQLite lock while %s; retry=%d/%d delay=%.2fs",
                    description,
                    attempt + 1,
                    args.sqlite_write_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
        except Exception:
            connection.rollback()
            raise
    raise AssertionError("unreachable")


def create_manifest_table(connection: sqlite3.Connection, uids: list[str]) -> None:
    """Load manifest UIDs into SQLite and require every UID to exist."""
    connection.execute(
        """
        CREATE TEMP TABLE requested_property_uids (
            position INTEGER PRIMARY KEY,
            paper_uid TEXT NOT NULL UNIQUE
        )
        """
    )
    connection.executemany(
        "INSERT INTO requested_property_uids(position, paper_uid) VALUES (?, ?)",
        enumerate(uids),
    )
    missing = [
        str(row[0])
        for row in connection.execute(
            f"""
            SELECT requested.paper_uid
            FROM requested_property_uids AS requested
            LEFT JOIN {PAPERS_TABLE} AS papers USING (paper_uid)
            WHERE papers.paper_uid IS NULL
            ORDER BY requested.position
            """
        )
    ]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"{len(missing)} manifest UID(s) are absent from {PAPERS_TABLE}: "
            f"{preview}{suffix}"
        )


def create_selected_properties_table(connection: sqlite3.Connection, properties: list[str]) -> None:
    """Load the run's properties with their property-presence labels."""
    connection.execute(
        """
        CREATE TEMP TABLE selected_properties (
            position INTEGER PRIMARY KEY,
            property TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL UNIQUE
        )
        """
    )
    connection.executemany(
        "INSERT INTO selected_properties(position, property, label) VALUES (?, ?, ?)",
        [(i, prop, PROPERTY_CONFIG[prop]["presence_label"]) for i, prop in enumerate(properties)],
    )


def create_run_tracking_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS property_extraction_run_attempted (
            paper_uid TEXT NOT NULL,
            property TEXT NOT NULL,
            PRIMARY KEY (paper_uid, property)
        ) WITHOUT ROWID
        """
    )


def initialize_progress_rows(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    """Add missing progress rows for the manifest's flagged (paper, property) pairs.

    The same selection as the setup notebook: property presence succeeded and
    flags the property, and the paper has at least one sample.
    """

    def work(con: sqlite3.Connection) -> int:
        cursor = con.execute(
            f"""
            INSERT OR IGNORE INTO {PROGRESS_TABLE}(paper_uid, property)
            SELECT DISTINCT requested.paper_uid, selected.property
            FROM requested_property_uids AS requested
            JOIN {CLASSIFICATION_TABLE} AS classification USING (paper_uid)
            JOIN json_each(classification.properties_json) AS flagged
            JOIN selected_properties AS selected ON selected.label = flagged.value
            WHERE classification.status = 'success'
              AND EXISTS (
                  SELECT 1 FROM {SAMPLES_TABLE} AS samples
                  WHERE samples.paper_uid = requested.paper_uid
              )
            """
        )
        return max(cursor.rowcount, 0)

    return run_with_write_retries(connection, args, "initializing progress rows", work)


def eligibility_sql(args: argparse.Namespace) -> tuple[str, list[str]]:
    statuses = ["pending"]
    if args.retry_failed:
        statuses.append("failed")
    if args.retry_in_progress:
        statuses.append("in_progress")
    placeholders = ", ".join("?" for _ in statuses)
    return f"progress.status IN ({placeholders})", statuses


ELIGIBLE_PAIRS_SQL = f"""
    FROM requested_property_uids AS requested
    JOIN {PAPERS_TABLE} AS papers USING (paper_uid)
    JOIN {PROGRESS_TABLE} AS progress USING (paper_uid)
    JOIN selected_properties AS selected ON selected.property = progress.property
    WHERE {{clause}}
      AND NOT EXISTS (
          SELECT 1
          FROM property_extraction_run_attempted AS attempted
          WHERE attempted.paper_uid = progress.paper_uid
            AND attempted.property = progress.property
      )
"""


def count_eligible(connection: sqlite3.Connection, args: argparse.Namespace) -> tuple[int, int]:
    """Return the eligible (paper, property) pairs and the papers they belong to."""
    clause, parameters = eligibility_sql(args)
    pairs, papers = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT requested.paper_uid) "
        + ELIGIBLE_PAIRS_SQL.format(clause=clause),
        parameters,
    ).fetchone()
    return int(pairs), int(papers)


def claim_next_batch(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
    max_papers: int,
) -> list[PairClaim]:
    """Atomically claim the next papers (at most ``max_papers``) with all their eligible pairs."""
    clause, statuses = eligibility_sql(args)

    def work(con: sqlite3.Connection) -> list[PairClaim]:
        paper_uids = [
            str(row[0])
            for row in con.execute(
                "SELECT requested.paper_uid "
                + ELIGIBLE_PAIRS_SQL.format(clause=clause)
                + " GROUP BY requested.paper_uid ORDER BY MIN(requested.position) LIMIT ?",
                [*statuses, max_papers],
            )
        ]
        if not paper_uids:
            return []
        placeholders = ", ".join("?" for _ in paper_uids)
        rows = con.execute(
            "SELECT papers.paper_uid, papers.canonical_source, papers.fulltext_path, "
            "progress.property, progress.status "
            + ELIGIBLE_PAIRS_SQL.format(clause=clause)
            + f" AND requested.paper_uid IN ({placeholders})"
            + " ORDER BY requested.position, selected.position",
            [*statuses, *paper_uids],
        ).fetchall()

        claimed: list[PairClaim] = []
        for paper_uid, source, fulltext_path, prop, previous_status in rows:
            cursor = con.execute(
                f"""
                UPDATE {PROGRESS_TABLE}
                SET status = 'in_progress',
                    attempt_count = attempt_count + 1,
                    last_error = NULL,
                    started_at = CURRENT_TIMESTAMP,
                    completed_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE paper_uid = ? AND property = ? AND status = ?
                """,
                (paper_uid, prop, previous_status),
            )
            if cursor.rowcount != 1:
                continue
            con.execute(
                "INSERT INTO property_extraction_run_attempted(paper_uid, property) VALUES (?, ?)",
                (paper_uid, prop),
            )
            claimed.append(
                PairClaim(
                    paper=PaperRow(str(paper_uid), str(source), str(fulltext_path)),
                    property=str(prop),
                    previous_status=str(previous_status),
                )
            )
        return claimed

    return run_with_write_retries(connection, args, "claiming a batch", work)


def load_samples(connection: sqlite3.Connection, paper_uids: list[str]) -> dict[str, list[SampleRef]]:
    """Read the samples of the given papers, in sample_id order."""
    placeholders = ", ".join("?" for _ in paper_uids)
    samples: dict[str, list[SampleRef]] = {uid: [] for uid in paper_uids}
    for paper_uid, sample_id, name, identifier, aliases, description in connection.execute(
        f"""
        SELECT paper_uid, sample_id, name, identifier, aliases, description
        FROM {SAMPLES_TABLE}
        WHERE paper_uid IN ({placeholders})
        ORDER BY paper_uid, sample_id
        """,
        paper_uids,
    ):
        samples[str(paper_uid)].append(
            SampleRef(
                sample_id=int(sample_id),
                name=name,
                identifier=identifier,
                aliases=aliases,
                description=description,
            )
        )
    return samples


def resolve_markdown_path(paper: PaperRow, fulltext_root: Path) -> Path:
    """Resolve the Markdown sibling of the stored source-document path."""
    stored = Path(paper.fulltext_path).expanduser()
    if not stored.is_absolute():
        stored = REPO_ROOT / stored
    stored = stored.resolve()

    candidates = [
        stored if stored.suffix.lower() == ".md" else stored.with_suffix(".md")
    ]
    candidates.append(
        fulltext_root
        / paper.canonical_source
        / paper.paper_uid
        / f"{paper.paper_uid}.md"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


async def extract_paper_pairs(
    args: argparse.Namespace,
    extractors: dict[str, PaperPropertyExtractor | SamplePropertyExtractor],
    semaphore: asyncio.Semaphore,
    fulltext_root: Path,
    claims: list[PairClaim],
    samples: list[SampleRef],
) -> list[PairItem]:
    """Run every claimed property of one paper; the paper's text is read once."""
    paper = claims[0].paper

    def failed(error: str, missing: bool = False) -> list[PairItem]:
        return [PairItem(c, None, 0.0, local_error=error, markdown_missing=missing) for c in claims]

    markdown_path = resolve_markdown_path(paper, fulltext_root)
    if not markdown_path.is_file():
        return failed(f"Markdown file not found: {markdown_path}", missing=True)
    if not samples:
        return failed("Paper has no samples")
    try:
        markdown = await asyncio.to_thread(
            markdown_path.read_text, encoding="utf-8", errors="replace"
        )
        context = PaperContext(paper_uid=paper.paper_uid, samples=samples, markdown_full_text=markdown)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return failed(f"{type(exc).__name__}: {exc}")

    async def one(claim: PairClaim) -> PairItem:
        extractor = extractors[claim.property]
        started = time.monotonic()
        try:
            if args.mode == "p2":
                result = await extractor.aextract_paper(
                    context, batch_size=args.p2_batch_size, semaphore=semaphore
                )
            else:
                result = await extractor.aextract_paper(context, semaphore=semaphore)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return PairItem(claim, None, time.monotonic() - started,
                            local_error=f"{type(exc).__name__}: {exc}")
        return PairItem(claim, result, time.monotonic() - started)

    return list(await asyncio.gather(*(one(c) for c in claims)))


def failure_message(item: PairItem) -> str:
    if item.local_error is not None:
        return item.local_error
    if item.result is not None and item.result.failures:
        first = item.result.failures[0]
        message = (
            f"{len(item.result.failures)}/{item.result.requests} request(s) failed; "
            f"first (samples {first.sample_ids}): {first.error_type}: {first.message}"
        )
        if first.raw_response:
            message += f" | raw_response={first.raw_response[:2000]}"
        return message[:MAX_ERROR_CHARS]
    return "unknown extraction failure"


def property_rows(result: PaperPropertyResult) -> list[tuple[Any, ...]]:
    """Column values of the result's stored records, in PROPERTY_COLUMNS order."""
    rows = []
    for record in result.to_properties():
        data = record.model_dump(mode="json")
        data["conditions"] = json.dumps(data["conditions"], ensure_ascii=False)
        data["flags"] = json.dumps(data["flags"], ensure_ascii=False)
        rows.append(tuple(data[column] for column in PROPERTY_COLUMNS))
    return rows


def unlinked_count(result: PaperPropertyResult) -> int:
    return sum(not record.linked for record in result.records)


def persist_batch(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
    items: list[PairItem],
    progress: Progress,
) -> bool:
    """Atomically persist property rows and terminal status for every pair."""
    mode = MODES[args.mode]
    rows_by_item = [property_rows(item.result) if item.succeeded else [] for item in items]
    insert_sql = (
        f"INSERT INTO {PROPERTIES_TABLE}({', '.join(PROPERTY_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in PROPERTY_COLUMNS)})"
    )

    def work(con: sqlite3.Connection) -> None:
        for item, rows in zip(items, rows_by_item):
            paper_uid, prop = item.claim.paper.paper_uid, item.claim.property
            if item.succeeded:
                result = item.result
                con.execute(
                    f"DELETE FROM {PROPERTIES_TABLE} WHERE paper_uid = ? AND property = ?",
                    (paper_uid, prop),
                )
                con.executemany(insert_sql, rows)
                cursor = con.execute(
                    f"""
                    UPDATE {PROGRESS_TABLE}
                    SET status = 'success',
                        mode = ?,
                        prompt_hash = ?,
                        model = ?,
                        requests = ?,
                        records = ?,
                        unlinked = ?,
                        prompt_tokens = ?,
                        completion_tokens = ?,
                        last_error = NULL,
                        completed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE paper_uid = ? AND property = ? AND status = 'in_progress'
                    """,
                    (
                        mode, result.prompt_hash, args.model, result.requests, len(rows),
                        unlinked_count(result), result.prompt_tokens, result.completion_tokens,
                        paper_uid, prop,
                    ),
                )
            else:
                # Stored rows and their run details stay as they were.
                cursor = con.execute(
                    f"""
                    UPDATE {PROGRESS_TABLE}
                    SET status = 'failed',
                        last_error = ?,
                        completed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE paper_uid = ? AND property = ? AND status = 'in_progress'
                    """,
                    (failure_message(item), paper_uid, prop),
                )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "Progress row changed concurrently for "
                    f"paper_uid={paper_uid} property={prop}; refusing a partial commit"
                )

    try:
        run_with_write_retries(connection, args, f"persisting {len(items)} results", work)
    except sqlite3.Error:
        LOG.exception(
            "SQLite transaction failed for %d extraction results; the entire batch was rolled back",
            len(items),
        )
        progress.database_errors += len(items)
        return False
    except Exception:
        LOG.exception("Could not persist %d extraction results; batch rolled back", len(items))
        progress.database_errors += len(items)
        return False

    progress.attempted += len(items)
    for item, rows in zip(items, rows_by_item):
        if item.result is not None:
            progress.requests += item.result.requests
            progress.prompt_tokens += item.result.prompt_tokens
            progress.completion_tokens += item.result.completion_tokens
        if item.succeeded:
            progress.successful_pairs += 1
            progress.inserted_properties += len(rows)
            progress.unlinked_values += unlinked_count(item.result)
            if not rows:
                progress.zero_record_pairs += 1
        else:
            progress.failed_pairs += 1
            if item.markdown_missing:
                progress.missing_markdown += 1
    return True


def release_claims(connection: sqlite3.Connection, claims: list[PairClaim]) -> None:
    """Return an interrupted, unpersisted batch to pending status."""
    if not claims:
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.executemany(
            f"""
            UPDATE {PROGRESS_TABLE}
            SET status = 'pending',
                last_error = 'Interrupted before extraction results were persisted',
                completed_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_uid = ? AND property = ? AND status = 'in_progress'
            """,
            ((c.paper.paper_uid, c.property) for c in claims),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def log_item(item: PairItem) -> None:
    prefix = f"paper_uid={item.claim.paper.paper_uid} property={item.claim.property}"
    if item.succeeded:
        result = item.result
        LOG.info(
            "%s requests=%d records=%d unlinked=%d duplicates_dropped=%d tokens_input=%d "
            "tokens_output=%d latency=%.2fs",
            prefix,
            result.requests,
            len(result.to_properties()),
            unlinked_count(result),
            result.duplicates_dropped,
            result.prompt_tokens,
            result.completion_tokens,
            item.latency_seconds,
        )
    else:
        LOG.error(
            "%s extraction_failure=%s latency=%.2fs",
            prefix,
            failure_message(item),
            item.latency_seconds,
        )


def log_progress(progress: Progress, target_papers: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    LOG.info(
        "Progress papers=%d/%d pairs_attempted=%d success=%d failed=%d properties=%d unlinked=%d "
        "zero_record_pairs=%d missing_markdown=%d database_errors=%d requests=%d "
        "tokens_input=%d tokens_output=%d rate=%.3f pairs/s",
        progress.claimed_papers,
        target_papers,
        progress.attempted,
        progress.successful_pairs,
        progress.failed_pairs,
        progress.inserted_properties,
        progress.unlinked_values,
        progress.zero_record_pairs,
        progress.missing_markdown,
        progress.database_errors,
        progress.requests,
        progress.prompt_tokens,
        progress.completion_tokens,
        progress.attempted / elapsed,
    )


async def run_with_extractors(
    args: argparse.Namespace,
    db_path: Path,
    extractors: dict[str, PaperPropertyExtractor | SamplePropertyExtractor],
) -> int:
    fulltext_root = args.fulltext_root.expanduser().resolve()
    if not fulltext_root.is_dir():
        LOG.error("Full-text root does not exist: %s", fulltext_root)
        return 1

    try:
        uids = load_uid_file(args.uid_file)
    except (OSError, ValueError, json.JSONDecodeError):
        LOG.exception("Could not load UID manifest: %s", args.uid_file)
        return 1

    connection = sqlite3.connect(db_path, timeout=args.sqlite_timeout, isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(args.sqlite_timeout * 1000)}")
    progress = Progress()
    started = time.monotonic()

    try:
        validate_database(connection)
        create_manifest_table(connection, uids)
        create_selected_properties_table(connection, args.properties)
        added = initialize_progress_rows(connection, args)
        if added:
            LOG.info("Added %d missing progress rows", added)
        create_run_tracking_table(connection)
        progress.eligible_at_start, progress.eligible_papers_at_start = count_eligible(connection, args)
        target = progress.eligible_papers_at_start
        if args.limit is not None:
            target = min(target, args.limit)

        LOG.info(
            "Manifest_uids=%d eligible_pairs=%d eligible_papers=%d target_papers=%d mode=%s "
            "properties=%s database=%s fulltext_root=%s retry_failed=%s retry_in_progress=%s",
            len(uids),
            progress.eligible_at_start,
            progress.eligible_papers_at_start,
            target,
            args.mode,
            ",".join(args.properties),
            db_path,
            fulltext_root,
            args.retry_failed,
            args.retry_in_progress,
        )
        if target == 0:
            LOG.info("No eligible (paper, property) pairs to extract")
            return 0

        first = next(iter(extractors.values()))
        config = first.backend.config()
        LOG.info(
            "Extractor model=%s base_url=%s timeout=%s max_tokens=%s extra_body=%s "
            "max_parallel_requests=%d batch_size=%d papers p2_batch_size=%d",
            config.get("model"),
            config.get("base_url"),
            config.get("timeout"),
            config.get("max_tokens"),
            config.get("extra_body"),
            args.max_parallel_requests,
            args.batch_size,
            args.p2_batch_size,
        )
        for prop, extractor in extractors.items():
            LOG.info("Prompt property=%s mode=%s prompt_hash=%s", prop, extractor.mode, extractor.prompt_hash())
        if not args.skip_health_check:
            try:
                first.health_check()
            except Exception:
                LOG.exception("Model endpoint health check failed")
                return 1

        semaphore = asyncio.Semaphore(args.max_parallel_requests)
        batch_number = 0
        while progress.claimed_papers < target:
            batch = claim_next_batch(
                connection, args, min(args.batch_size, target - progress.claimed_papers)
            )
            if not batch:
                break
            batch_number += 1
            by_paper: dict[str, list[PairClaim]] = {}
            for claim in batch:
                by_paper.setdefault(claim.paper.paper_uid, []).append(claim)
            progress.claimed += len(batch)
            progress.claimed_papers += len(by_paper)
            LOG.info(
                "Starting batch=%d papers=%d pairs=%d claimed_papers=%d/%d",
                batch_number,
                len(by_paper),
                len(batch),
                progress.claimed_papers,
                target,
            )
            try:
                samples = load_samples(connection, list(by_paper))
                grouped = await asyncio.gather(
                    *(
                        extract_paper_pairs(
                            args=args,
                            extractors=extractors,
                            semaphore=semaphore,
                            fulltext_root=fulltext_root,
                            claims=claims,
                            samples=samples[paper_uid],
                        )
                        for paper_uid, claims in by_paper.items()
                    )
                )
            except asyncio.CancelledError:
                try:
                    release_claims(connection, batch)
                except Exception:
                    LOG.exception(
                        "Could not return interrupted batch to pending; rerun with --retry-in-progress"
                    )
                raise

            items = [item for paper_items in grouped for item in paper_items]
            for item in items:
                log_item(item)
            persist_batch(connection, args, items, progress)
            log_progress(progress, target, started)

    except asyncio.CancelledError:
        LOG.warning("Cancelled; previously committed batches are preserved")
        raise
    except (RuntimeError, sqlite3.Error, OSError):
        LOG.exception("Fatal database or filesystem error")
        return 1
    finally:
        connection.close()

    LOG.info(
        "Finished attempted=%d pairs (%d papers) success=%d failed=%d inserted_properties=%d "
        "unlinked=%d requests=%d",
        progress.attempted,
        progress.claimed_papers,
        progress.successful_pairs,
        progress.failed_pairs,
        progress.inserted_properties,
        progress.unlinked_values,
        progress.requests,
    )
    if progress.failed_pairs or progress.database_errors:
        LOG.error(
            "Job incomplete: failed_pairs=%d database_errors=%d; use "
            "--retry-failed or --retry-in-progress as appropriate",
            progress.failed_pairs,
            progress.database_errors,
        )
        return 2
    return 0


async def run_async(args: argparse.Namespace) -> int:
    db_path = args.db_path.expanduser().resolve()
    if not db_path.is_file():
        LOG.error("Database does not exist: %s", db_path)
        return 1

    backend = LLMBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        extra_body=args.extra_body,
    )
    extractor_kwargs: dict[str, Any] = {}
    if args.prompt_dir is not None:
        extractor_kwargs["prompt_dir"] = args.prompt_dir

    try:
        cls = EXTRACTOR_CLASSES[args.mode]
        extractors = {prop: cls(backend, prop, **extractor_kwargs) for prop in args.properties}
        return await run_with_extractors(args, db_path, extractors)
    finally:
        backend.close()
        await backend.aclose()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        return asyncio.run(run_async(args))
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
