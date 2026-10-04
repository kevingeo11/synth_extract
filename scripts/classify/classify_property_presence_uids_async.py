#!/usr/bin/env python3
"""Classify target-property presence for manifest papers asynchronously.

Only UIDs in the required manifest are considered. Eligible progress rows are
claimed as ``in_progress`` in short SQLite transactions. Each successful result
atomically stores ``has_property``, ``properties_json``, and ``success`` in the
``property_classification`` table; failures store a terminal error.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synth_extract.agents.classification import (  # noqa: E402
    PropertyPresenceClassifier,
    PropertyPresenceResult,
)
from synth_extract.agents.llm import (  # noqa: E402
    LLMBackend,
    StructuredFailure,
    StructuredOutcome,
    StructuredSuccess,
)


LOG = logging.getLogger("property_presence_classification_async")
PAPERS_TABLE = "papers"
RESULT_TABLE = "property_classification"


@dataclass(frozen=True, slots=True)
class PaperRow:
    paper_uid: str
    canonical_source: str
    fulltext_path: str
    previous_status: str | None


@dataclass(frozen=True, slots=True)
class ClassificationItem:
    paper: PaperRow
    markdown_path: Path
    outcome: StructuredOutcome[PropertyPresenceResult] | None
    latency_seconds: float
    local_error: str | None = None
    markdown_missing: bool = False


@dataclass(slots=True)
class Progress:
    requested: int = 0
    eligible_at_start: int = 0
    claimed: int = 0
    attempted: int = 0
    successful: int = 0
    positive: int = 0
    negative: int = 0
    failed: int = 0
    missing_markdown: int = 0
    database_errors: int = 0
    properties: Counter[str] = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
    """Load a text, JSON, or JSONL UID manifest without accepting duplicates."""
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify fixed property presence for manifest papers and update "
            f"the {RESULT_TABLE} table."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=REPO_ROOT / "data" / "polymer_samples.db",
        help="SQLite database (default: %(default)s)",
    )
    parser.add_argument(
        "--uid-file",
        type=Path,
        required=True,
        help="UID manifest: one UID per line, JSON, or JSONL",
    )
    parser.add_argument(
        "--fulltext-root",
        type=Path,
        default=REPO_ROOT / "data" / "fulltext",
        help="Fallback root containing source/paper_uid/paper_uid.md",
    )
    parser.add_argument("--system-prompt-path", type=Path, default=None)
    parser.add_argument("--user-template-path", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=25)
    parser.add_argument(
        "--max-parallel-requests",
        type=positive_int,
        default=int(os.getenv("MAX_PARALLEL_REQUESTS", "8")),
    )
    parser.add_argument("--timeout", type=positive_float, default=300.0)
    parser.add_argument("--max-tokens", type=positive_int, default=16384)
    parser.add_argument("--extra-body", type=json_object, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--retry-in-progress",
        action="store_true",
        help=(
            "Reclaim interrupted in_progress rows; do not use while another "
            "worker for the same manifests is active"
        ),
    )
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--sqlite-timeout", type=positive_float, default=60.0)
    parser.add_argument(
        "--sqlite-write-retries", type=nonnegative_int, default=5
    )
    parser.add_argument(
        "--sqlite-retry-base-delay", type=positive_float, default=1.0
    )
    parser.add_argument(
        "--sqlite-retry-max-delay", type=positive_float, default=30.0
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


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


def validate_database(connection: sqlite3.Connection) -> None:
    required_columns(
        connection,
        PAPERS_TABLE,
        {"paper_uid", "canonical_source", "fulltext_path"},
    )
    required_columns(
        connection,
        RESULT_TABLE,
        {
            "paper_uid",
            "status",
            "attempt_count",
            "has_property",
            "properties_json",
            "last_error",
            "started_at",
            "completed_at",
            "updated_at",
        },
    )


def retryable_sqlite_lock_error(exc: sqlite3.Error) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        primary_code = error_code & 0xFF
        if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


def retry_delay(args: argparse.Namespace, attempt: int) -> float:
    maximum = min(
        args.sqlite_retry_max_delay,
        args.sqlite_retry_base_delay * (2**attempt),
    )
    return maximum * random.uniform(0.8, 1.2)


def create_manifest_table(connection: sqlite3.Connection, uids: list[str]) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE requested_property_presence_uids (
            position INTEGER PRIMARY KEY,
            paper_uid TEXT NOT NULL UNIQUE
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO requested_property_presence_uids(position, paper_uid)
        VALUES (?, ?)
        """,
        enumerate(uids),
    )
    missing = [
        str(row[0])
        for row in connection.execute(
            f"""
            SELECT requested.paper_uid
            FROM requested_property_presence_uids AS requested
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


def initialize_progress_rows(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> int:
    for attempt in range(args.sqlite_write_retries + 1):
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"""
                INSERT OR IGNORE INTO {RESULT_TABLE}(paper_uid)
                SELECT papers.paper_uid
                FROM requested_property_presence_uids AS requested
                JOIN {PAPERS_TABLE} AS papers USING (paper_uid)
                """
            )
            connection.commit()
            return max(cursor.rowcount, 0)
        except sqlite3.Error as exc:
            connection.rollback()
            if (
                retryable_sqlite_lock_error(exc)
                and attempt < args.sqlite_write_retries
            ):
                delay = retry_delay(args, attempt)
                LOG.warning(
                    "SQLite lock initializing progress; retry=%d/%d delay=%.2fs",
                    attempt + 1,
                    args.sqlite_write_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
    raise AssertionError("unreachable")


def create_run_tracking_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE property_presence_run_attempted (
            paper_uid TEXT PRIMARY KEY
        ) WITHOUT ROWID
        """
    )


def eligibility_sql(args: argparse.Namespace) -> tuple[str, list[str]]:
    statuses = ["pending"]
    if args.retry_failed:
        statuses.append("failed")
    if args.retry_in_progress:
        statuses.append("in_progress")
    placeholders = ", ".join("?" for _ in statuses)
    return f"(result.status IS NULL OR result.status IN ({placeholders}))", statuses


def count_eligible(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    clause, parameters = eligibility_sql(args)
    return int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM requested_property_presence_uids AS requested
            JOIN {RESULT_TABLE} AS result USING (paper_uid)
            WHERE {clause}
            """,
            parameters,
        ).fetchone()[0]
    )


def claim_next_batch(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
    batch_size: int,
) -> list[PaperRow]:
    clause, statuses = eligibility_sql(args)
    for attempt in range(args.sqlite_write_retries + 1):
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""
                SELECT papers.paper_uid, papers.canonical_source,
                       papers.fulltext_path, result.status
                FROM requested_property_presence_uids AS requested
                JOIN {PAPERS_TABLE} AS papers USING (paper_uid)
                JOIN {RESULT_TABLE} AS result USING (paper_uid)
                WHERE {clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM property_presence_run_attempted AS attempted
                      WHERE attempted.paper_uid = papers.paper_uid
                  )
                ORDER BY requested.position
                LIMIT ?
                """,
                [*statuses, batch_size],
            ).fetchall()

            claimed: list[PaperRow] = []
            for paper_uid, source, fulltext_path, previous_status in rows:
                cursor = connection.execute(
                    f"""
                    UPDATE {RESULT_TABLE}
                    SET status = 'in_progress',
                        attempt_count = attempt_count + 1,
                        has_property = NULL,
                        properties_json = NULL,
                        last_error = NULL,
                        started_at = CURRENT_TIMESTAMP,
                        completed_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE paper_uid = ? AND status IS ?
                    """,
                    (paper_uid, previous_status),
                )
                if cursor.rowcount != 1:
                    continue
                connection.execute(
                    "INSERT INTO property_presence_run_attempted VALUES (?)",
                    (paper_uid,),
                )
                claimed.append(
                    PaperRow(
                        paper_uid=str(paper_uid),
                        canonical_source=str(source),
                        fulltext_path=str(fulltext_path),
                        previous_status=(
                            None if previous_status is None else str(previous_status)
                        ),
                    )
                )
            connection.commit()
            return claimed
        except sqlite3.Error as exc:
            connection.rollback()
            if (
                retryable_sqlite_lock_error(exc)
                and attempt < args.sqlite_write_retries
            ):
                delay = retry_delay(args, attempt)
                LOG.warning(
                    "SQLite lock claiming batch; retry=%d/%d delay=%.2fs",
                    attempt + 1,
                    args.sqlite_write_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
    raise AssertionError("unreachable")


def resolve_markdown_path(paper: PaperRow, fulltext_root: Path) -> Path:
    stored = Path(paper.fulltext_path).expanduser()
    if not stored.is_absolute():
        stored = REPO_ROOT / stored
    stored = stored.resolve()
    candidates = [
        stored if stored.suffix.lower() == ".md" else stored.with_suffix(".md"),
        fulltext_root
        / paper.canonical_source
        / paper.paper_uid
        / f"{paper.paper_uid}.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


async def classify_paper(
    classifier: PropertyPresenceClassifier,
    semaphore: asyncio.Semaphore,
    fulltext_root: Path,
    paper: PaperRow,
) -> ClassificationItem:
    markdown_path = resolve_markdown_path(paper, fulltext_root)
    if not markdown_path.is_file():
        return ClassificationItem(
            paper=paper,
            markdown_path=markdown_path,
            outcome=None,
            latency_seconds=0.0,
            local_error=f"Markdown file not found: {markdown_path}",
            markdown_missing=True,
        )

    started = time.monotonic()
    try:
        markdown = await asyncio.to_thread(
            markdown_path.read_text,
            encoding="utf-8",
            errors="replace",
        )
        async with semaphore:
            outcome = await classifier.aclassify(markdown)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return ClassificationItem(
            paper=paper,
            markdown_path=markdown_path,
            outcome=None,
            latency_seconds=time.monotonic() - started,
            local_error=f"{type(exc).__name__}: {exc}",
        )
    return ClassificationItem(
        paper=paper,
        markdown_path=markdown_path,
        outcome=outcome,
        latency_seconds=time.monotonic() - started,
    )


def failure_message(item: ClassificationItem) -> str:
    if item.local_error is not None:
        return item.local_error
    if isinstance(item.outcome, StructuredFailure):
        message = f"{item.outcome.error_type}: {item.outcome.message}"
        if item.outcome.raw_response:
            message += f" | raw_response={item.outcome.raw_response[:4000]}"
        return message
    return "unknown classification failure"


def persist_batch(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
    items: list[ClassificationItem],
    progress: Progress,
) -> bool:
    for attempt in range(args.sqlite_write_retries + 1):
        try:
            connection.execute("BEGIN IMMEDIATE")
            for item in items:
                paper_uid = item.paper.paper_uid
                if isinstance(item.outcome, StructuredSuccess):
                    result = item.outcome.result
                    properties_json = json.dumps(
                        result.properties,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    cursor = connection.execute(
                        f"""
                        UPDATE {RESULT_TABLE}
                        SET status = 'success',
                            has_property = ?,
                            properties_json = ?,
                            last_error = NULL,
                            completed_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE paper_uid = ? AND status = 'in_progress'
                        """,
                        (int(result.has_property), properties_json, paper_uid),
                    )
                else:
                    cursor = connection.execute(
                        f"""
                        UPDATE {RESULT_TABLE}
                        SET status = 'failed',
                            has_property = NULL,
                            properties_json = NULL,
                            last_error = ?,
                            completed_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE paper_uid = ? AND status = 'in_progress'
                        """,
                        (failure_message(item), paper_uid),
                    )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "Progress row changed concurrently for "
                        f"paper_uid={paper_uid}; batch rolled back"
                    )
            connection.commit()
            break
        except sqlite3.Error as exc:
            connection.rollback()
            if (
                retryable_sqlite_lock_error(exc)
                and attempt < args.sqlite_write_retries
            ):
                delay = retry_delay(args, attempt)
                LOG.warning(
                    "SQLite lock persisting %d results; retry=%d/%d delay=%.2fs",
                    len(items),
                    attempt + 1,
                    args.sqlite_write_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            LOG.exception("Database transaction failed; batch rolled back")
            progress.database_errors += len(items)
            return False
        except Exception:
            connection.rollback()
            LOG.exception("Could not persist results; batch rolled back")
            progress.database_errors += len(items)
            return False

    progress.attempted += len(items)
    for item in items:
        if isinstance(item.outcome, StructuredSuccess):
            progress.successful += 1
            result = item.outcome.result
            if result.has_property:
                progress.positive += 1
            else:
                progress.negative += 1
            progress.properties.update(result.properties)
            usage = item.outcome.metadata.usage
            if usage is not None:
                progress.prompt_tokens += usage.prompt_tokens or 0
                progress.completion_tokens += usage.completion_tokens or 0
                progress.total_tokens += usage.total_tokens or 0
        else:
            progress.failed += 1
            if item.markdown_missing:
                progress.missing_markdown += 1
    return True


def release_claims(
    connection: sqlite3.Connection,
    paper_uids: list[str],
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.executemany(
            f"""
            UPDATE {RESULT_TABLE}
            SET status = 'pending',
                has_property = NULL,
                properties_json = NULL,
                last_error = 'Interrupted before results were persisted',
                completed_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_uid = ? AND status = 'in_progress'
            """,
            ((paper_uid,) for paper_uid in paper_uids),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def log_item(item: ClassificationItem) -> None:
    prefix = f"paper_uid={item.paper.paper_uid}"
    if isinstance(item.outcome, StructuredSuccess):
        LOG.info(
            "%s has_property=%s properties=%s latency=%.2fs",
            prefix,
            item.outcome.result.has_property,
            item.outcome.result.properties,
            item.latency_seconds,
        )
    else:
        LOG.error(
            "%s classification_failure=%s latency=%.2fs",
            prefix,
            failure_message(item),
            item.latency_seconds,
        )


def log_progress(progress: Progress, target: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    LOG.info(
        "Progress attempted=%d/%d success=%d positive=%d negative=%d "
        "failed=%d missing_markdown=%d database_errors=%d "
        "tokens_input=%d tokens_output=%d tokens_total=%d rate=%.2f papers/s",
        progress.attempted,
        target,
        progress.successful,
        progress.positive,
        progress.negative,
        progress.failed,
        progress.missing_markdown,
        progress.database_errors,
        progress.prompt_tokens,
        progress.completion_tokens,
        progress.total_tokens,
        progress.attempted / elapsed,
    )


async def run_with_classifier(
    args: argparse.Namespace,
    db_path: Path,
    classifier: PropertyPresenceClassifier,
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

    connection = sqlite3.connect(
        db_path,
        timeout=args.sqlite_timeout,
        isolation_level=None,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(args.sqlite_timeout * 1000)}")
    progress = Progress(requested=len(uids))
    started = time.monotonic()

    try:
        validate_database(connection)
        create_manifest_table(connection, uids)
        added = initialize_progress_rows(connection, args)
        if added:
            LOG.info("Added %d missing property-classification rows", added)
        create_run_tracking_table(connection)
        progress.eligible_at_start = count_eligible(connection, args)
        target = progress.eligible_at_start
        if args.limit is not None:
            target = min(target, args.limit)

        LOG.info(
            "Manifest_uids=%d eligible=%d target=%d database=%s "
            "retry_failed=%s retry_in_progress=%s",
            len(uids),
            progress.eligible_at_start,
            target,
            db_path,
            args.retry_failed,
            args.retry_in_progress,
        )
        if target == 0:
            LOG.info("No eligible manifest papers to classify")
            return 0

        config = classifier.llm_config()
        LOG.info(
            "Classifier model=%s base_url=%s timeout=%s max_tokens=%s "
            "prompt_hash=%s extra_body=%s parallel=%d batch_size=%d",
            config.get("model"),
            config.get("base_url"),
            config.get("timeout"),
            config.get("max_tokens"),
            config.get("prompt_hash"),
            config.get("extra_body"),
            args.max_parallel_requests,
            args.batch_size,
        )
        if not args.skip_health_check:
            try:
                classifier.health_check()
            except Exception:
                LOG.exception("Model endpoint health check failed")
                return 1

        semaphore = asyncio.Semaphore(args.max_parallel_requests)
        batch_number = 0
        while progress.claimed < target:
            batch = claim_next_batch(
                connection,
                args,
                min(args.batch_size, target - progress.claimed),
            )
            if not batch:
                break
            batch_number += 1
            progress.claimed += len(batch)
            LOG.info(
                "Starting batch=%d rows=%d claimed=%d/%d",
                batch_number,
                len(batch),
                progress.claimed,
                target,
            )
            try:
                items = await asyncio.gather(
                    *(
                        classify_paper(
                            classifier,
                            semaphore,
                            fulltext_root,
                            paper,
                        )
                        for paper in batch
                    )
                )
            except asyncio.CancelledError:
                try:
                    release_claims(
                        connection,
                        [paper.paper_uid for paper in batch],
                    )
                except Exception:
                    LOG.exception(
                        "Could not release interrupted batch; use "
                        "--retry-in-progress"
                    )
                raise

            for item in items:
                log_item(item)
            persist_batch(connection, args, items, progress)
            log_progress(progress, target, started)

    except asyncio.CancelledError:
        LOG.warning("Cancelled; previously committed batches are preserved")
        raise
    except (RuntimeError, sqlite3.Error, OSError):
        LOG.exception("Fatal database, manifest, or filesystem error")
        return 1
    finally:
        connection.close()

    LOG.info(
        "Finished success=%d positive=%d negative=%d failed=%d properties=%s",
        progress.successful,
        progress.positive,
        progress.negative,
        progress.failed,
        dict(progress.properties),
    )
    if progress.failed or progress.database_errors:
        LOG.error(
            "Job incomplete: failed=%d database_errors=%d",
            progress.failed,
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
    classifier_kwargs: dict[str, Any] = {}
    if args.system_prompt_path is not None:
        classifier_kwargs["system_prompt_path"] = args.system_prompt_path
    if args.user_template_path is not None:
        classifier_kwargs["user_template_path"] = args.user_template_path

    try:
        classifier = PropertyPresenceClassifier(backend, **classifier_kwargs)
        return await run_with_classifier(args, db_path, classifier)
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
