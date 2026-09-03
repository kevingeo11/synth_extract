#!/usr/bin/env python3
"""Classify Markdown papers from a UID manifest and update ``classify``.

Only manifest UIDs whose selected result column is NULL are classified. Papers
are processed concurrently in small batches, while successful labels are
committed to SQLite in one short transaction per batch. Failed classifications
and missing/unreadable Markdown files remain NULL so they can be retried.

UID files may be one-UID-per-line text files, JSON arrays, JSON objects with an
``uids`` array, or JSONL containing strings or objects with ``paper_uid``.

Example
-------
python scripts/classify_fulltext_uids_async.py \
    --db-path data/central_papers.db \
    --uid-file manifests/class_run_1/job_000.txt \
    --result-column class_run_1 \
    --fulltext-root data/fulltext \
    --model qwen3.6-27b \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key none \
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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synth_extract.agents.classification import (  # noqa: E402
    ClassificationFailure,
    ClassificationResult,
    FullTextClassifier,
)
from synth_extract.agents.llm import LLMBackend  # noqa: E402

LOG = logging.getLogger("fulltext_uid_classification_async")
TABLE = "classify"
PROTECTED_COLUMNS = {
    "paper_id",
    "paper_uid",
    "canonical_source",
    "fulltext_path",
}


@dataclass
class Progress:
    requested: int = 0
    already_classified: int = 0
    attempted: int = 0
    updated: int = 0
    true: int = 0
    false: int = 0
    classification_errors: int = 0
    missing_markdown: int = 0
    database_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class PaperRow:
    paper_id: int
    paper_uid: str
    canonical_source: str


@dataclass(frozen=True)
class ClassificationItem:
    paper: PaperRow
    markdown_path: Path
    result: ClassificationResult | ClassificationFailure | None
    latency_seconds: float
    local_error: str | None = None
    markdown_missing: bool = False


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify pending UIDs from a manifest using their Markdown full "
            "text and update successful labels in the classify table."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite database containing the classify table",
    )
    parser.add_argument(
        "--uid-file",
        type=Path,
        required=True,
        help="UID manifest: text, JSON, or JSONL",
    )
    parser.add_argument(
        "--result-column",
        required=True,
        help="Existing nullable INTEGER column that receives 0/1 labels",
    )
    parser.add_argument(
        "--fulltext-root",
        type=Path,
        required=True,
        help="Root containing source/paper_uid/paper_uid.md",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=Path,
        default=None,
        help="Optional custom system prompt file",
    )
    parser.add_argument(
        "--user-template-path",
        type=Path,
        default=None,
        help="Optional custom user prompt template file",
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
        help="Maximum pending manifest UIDs to attempt (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=25,
        help=(
            "Classifications collected per database transaction "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=positive_int,
        default=int(os.getenv("MAX_PARALLEL_REQUESTS", "8")),
        help="Maximum in-flight model requests (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=300.0,
        help="Model request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=8192,
        help="Maximum completion tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--extra-body",
        type=json_object,
        default=None,
        help="Optional JSON object forwarded as OpenAI extra_body",
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
        help=(
            "Retries for SQLITE_BUSY/SQLITE_LOCKED write transactions "
            "(default: %(default)s)"
        ),
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
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=[handler],
        force=True,
    )


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


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
    """Load and strictly validate a UID manifest."""
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


def validate_classify_table(
    connection: sqlite3.Connection,
    result_column: str,
) -> str:
    columns = connection.execute(
        f"PRAGMA table_info({quote_identifier(TABLE)})"
    ).fetchall()
    if not columns:
        raise RuntimeError(f"Table {TABLE!r} does not exist or has no columns")

    by_name = {str(column[1]): column for column in columns}
    required = {"paper_id", "paper_uid", "canonical_source"}
    missing = sorted(required.difference(by_name))
    if missing:
        raise RuntimeError(
            f"Table {TABLE!r} is missing required columns: {', '.join(missing)}"
        )
    if result_column not in by_name:
        raise RuntimeError(
            f"Result column {result_column!r} does not exist in table {TABLE!r}"
        )
    if result_column in PROTECTED_COLUMNS:
        raise RuntimeError(f"Refusing to use protected column {result_column!r}")

    declared_type = str(by_name[result_column][2] or "").upper()
    if "INT" not in declared_type:
        raise RuntimeError(
            f"Result column {result_column!r} must be INTEGER, not {declared_type!r}"
        )
    return result_column


def load_manifest_rows(
    connection: sqlite3.Connection,
    uids: list[str],
    result_column: str,
) -> tuple[list[PaperRow], int]:
    """Resolve all manifest UIDs and return only rows still pending."""
    connection.execute(
        "CREATE TEMP TABLE requested_uids ("
        "position INTEGER PRIMARY KEY, paper_uid TEXT NOT NULL UNIQUE)"
    )
    connection.executemany(
        "INSERT INTO requested_uids(position, paper_uid) VALUES (?, ?)",
        enumerate(uids),
    )
    column_sql = quote_identifier(result_column)
    rows = connection.execute(
        f"""
        SELECT
            requested.position,
            requested.paper_uid,
            classify.paper_id,
            classify.canonical_source,
            classify.{column_sql}
        FROM requested_uids AS requested
        LEFT JOIN {quote_identifier(TABLE)} AS classify
            ON classify.paper_uid = requested.paper_uid
        ORDER BY requested.position
        """
    ).fetchall()

    missing_uids = [str(row[1]) for row in rows if row[2] is None]
    if missing_uids:
        preview = ", ".join(missing_uids[:10])
        suffix = " ..." if len(missing_uids) > 10 else ""
        raise RuntimeError(
            f"{len(missing_uids)} manifest UID(s) are absent from classify: "
            f"{preview}{suffix}"
        )

    pending: list[PaperRow] = []
    already_classified = 0
    for _, paper_uid, paper_id, canonical_source, result_value in rows:
        source = str(canonical_source)
        uid = str(paper_uid)
        if not safe_path_component(source):
            raise RuntimeError(
                f"Unsafe canonical_source for paper_uid={uid}: {source!r}"
            )
        if result_value is not None:
            already_classified += 1
            continue
        pending.append(
            PaperRow(
                paper_id=int(paper_id),
                paper_uid=uid,
                canonical_source=source,
            )
        )
    return pending, already_classified


def chunks(items: list[PaperRow], size: int) -> Iterator[list[PaperRow]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def classify_paper(
    classifier: FullTextClassifier,
    semaphore: asyncio.Semaphore,
    fulltext_root: Path,
    paper: PaperRow,
) -> ClassificationItem:
    markdown_path = (
        fulltext_root
        / paper.canonical_source
        / paper.paper_uid
        / f"{paper.paper_uid}.md"
    )
    if not markdown_path.is_file():
        return ClassificationItem(
            paper=paper,
            markdown_path=markdown_path,
            result=None,
            latency_seconds=0.0,
            local_error=f"Markdown file not found: {markdown_path}",
            markdown_missing=True,
        )

    started = time.monotonic()
    async with semaphore:
        try:
            full_text = await asyncio.to_thread(
                markdown_path.read_text,
                encoding="utf-8",
            )
            result = await classifier.aclassify(full_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = None
            local_error = f"{type(exc).__name__}: {exc}"
        else:
            local_error = None

    return ClassificationItem(
        paper=paper,
        markdown_path=markdown_path,
        result=result,
        latency_seconds=time.monotonic() - started,
        local_error=local_error,
    )


def log_completed_item(item: ClassificationItem) -> None:
    result = item.result
    paper = item.paper
    prefix = f"source={paper.canonical_source} paper_uid={paper.paper_uid}"
    if isinstance(result, ClassificationResult):
        usage = result.metadata.usage
        if usage is None:
            tokens = "tokens=unavailable"
        else:
            tokens = (
                f"tokens_input={usage.prompt_tokens} "
                f"tokens_output={usage.completion_tokens} "
                f"tokens_total={usage.total_tokens}"
            )
        LOG.info(
            "%s prediction=%d %s latency=%.2fs",
            prefix,
            int(result.label),
            tokens,
            item.latency_seconds,
        )
    elif isinstance(result, ClassificationFailure):
        LOG.error(
            "%s classification_failure=%s message=%s latency=%.2fs",
            prefix,
            result.error_type,
            result.message,
            item.latency_seconds,
        )
    else:
        LOG.error(
            "%s local_failure=%s",
            prefix,
            item.local_error or "unknown error",
        )


def add_usage(progress: Progress, result: ClassificationResult) -> None:
    usage = result.metadata.usage
    if usage is None:
        return
    progress.prompt_tokens += usage.prompt_tokens or 0
    progress.completion_tokens += usage.completion_tokens or 0
    progress.total_tokens += usage.total_tokens or 0


def retryable_sqlite_lock_error(exc: sqlite3.Error) -> bool:
    """Return whether an SQLite exception represents lock contention."""
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        primary_code = error_code & 0xFF
        if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


def write_result_batch(
    connection: sqlite3.Connection,
    result_column: str,
    items: list[ClassificationItem],
    progress: Progress,
    sqlite_write_retries: int,
    sqlite_retry_base_delay: float,
    sqlite_retry_max_delay: float,
) -> None:
    """Commit successful labels only, matching both paper ID and UID."""
    progress.attempted += len(items)
    candidates: list[tuple[ClassificationItem, int]] = []
    for item in items:
        result = item.result
        if isinstance(result, ClassificationResult):
            candidates.append((item, int(result.label)))
            add_usage(progress, result)
        else:
            progress.classification_errors += 1
            if item.markdown_missing:
                progress.missing_markdown += 1

    if not candidates:
        return

    column_sql = quote_identifier(result_column)
    update_sql = f"""
        UPDATE {quote_identifier(TABLE)}
        SET {column_sql} = ?
        WHERE paper_id = ?
          AND paper_uid = ?
          AND {column_sql} IS NULL
    """
    applied: list[tuple[ClassificationItem, int]] = []
    skipped: list[ClassificationItem] = []
    for attempt in range(sqlite_write_retries + 1):
        applied = []
        skipped = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            for item, label in candidates:
                cursor = connection.execute(
                    update_sql,
                    (label, item.paper.paper_id, item.paper.paper_uid),
                )
                if cursor.rowcount == 1:
                    applied.append((item, label))
                else:
                    skipped.append(item)
            connection.commit()
            break
        except sqlite3.Error as exc:
            connection.rollback()
            should_retry = (
                retryable_sqlite_lock_error(exc)
                and attempt < sqlite_write_retries
            )
            if should_retry:
                exponential_delay = min(
                    sqlite_retry_max_delay,
                    sqlite_retry_base_delay * (2**attempt),
                )
                delay = exponential_delay * random.uniform(0.8, 1.2)
                LOG.warning(
                    "SQLite write lock for %d results; retry=%d/%d "
                    "delay=%.2fs error=%s",
                    len(candidates),
                    attempt + 1,
                    sqlite_write_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue

            progress.database_errors += len(candidates)
            LOG.exception(
                "SQLite transaction failed for %d successful "
                "classifications after %d attempt(s); the entire batch "
                "was rolled back",
                len(candidates),
                attempt + 1,
            )
            return

    progress.updated += len(applied)
    progress.true += sum(label for _, label in applied)
    progress.false += sum(1 - label for _, label in applied)
    progress.database_errors += len(skipped)
    for item in skipped:
        LOG.error(
            "Database update skipped paper_id=%d paper_uid=%s; the UID or "
            "result column may have changed concurrently",
            item.paper.paper_id,
            item.paper.paper_uid,
        )


def log_progress(progress: Progress, target: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    LOG.info(
        "Progress attempted=%d/%d updated=%d true=%d false=%d "
        "classification_errors=%d missing_markdown=%d database_errors=%d "
        "tokens_input=%d tokens_output=%d tokens_total=%d rate=%.2f papers/s",
        progress.attempted,
        target,
        progress.updated,
        progress.true,
        progress.false,
        progress.classification_errors,
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
    classifier: FullTextClassifier,
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

    progress = Progress(requested=len(uids))
    connection = sqlite3.connect(
        db_path,
        timeout=args.sqlite_timeout,
        isolation_level=None,
    )
    connection.execute(
        f"PRAGMA busy_timeout = {int(args.sqlite_timeout * 1000)}"
    )
    started = time.monotonic()

    try:
        result_column = validate_classify_table(connection, args.result_column)
        pending, already_classified = load_manifest_rows(
            connection,
            uids,
            result_column,
        )
        progress.already_classified = already_classified
        if args.limit is not None:
            pending = pending[: args.limit]
        target = len(pending)
        LOG.info(
            "Manifest requested=%d already_classified=%d pending_target=%d "
            "database=%s table=%s result_column=%s fulltext_root=%s",
            progress.requested,
            progress.already_classified,
            target,
            db_path,
            TABLE,
            result_column,
            fulltext_root,
        )
        if target == 0:
            LOG.info("No pending manifest UIDs to classify")
            return 0

        config = classifier.llm_config()
        LOG.info(
            "Classifier model=%s base_url=%s timeout=%s max_tokens=%s "
            "prompt_hash=%s extra_body=%s max_parallel_requests=%d batch_size=%d",
            config.get("model"),
            config.get("base_url"),
            config.get("timeout"),
            config.get("max_tokens"),
            config.get("prompt_hash"),
            config.get("extra_body"),
            args.max_parallel_requests,
            args.batch_size,
        )
        health = classifier.health_check()
        if isinstance(health, ClassificationFailure):
            LOG.error(
                "Health check failed error_type=%s message=%s",
                health.error_type,
                health.message,
            )
            return 1

        semaphore = asyncio.Semaphore(args.max_parallel_requests)
        processed = 0
        for batch_number, batch in enumerate(
            chunks(pending, args.batch_size),
            start=1,
        ):
            LOG.info(
                "Starting batch=%d rows=%d progress=%d/%d",
                batch_number,
                len(batch),
                processed,
                target,
            )
            items = await asyncio.gather(
                *(
                    classify_paper(
                        classifier=classifier,
                        semaphore=semaphore,
                        fulltext_root=fulltext_root,
                        paper=paper,
                    )
                    for paper in batch
                )
            )
            for item in items:
                log_completed_item(item)
            write_result_batch(
                connection=connection,
                result_column=result_column,
                items=items,
                progress=progress,
                sqlite_write_retries=args.sqlite_write_retries,
                sqlite_retry_base_delay=args.sqlite_retry_base_delay,
                sqlite_retry_max_delay=args.sqlite_retry_max_delay,
            )
            processed += len(batch)
            log_progress(progress, target, started)

    except asyncio.CancelledError:
        LOG.warning("Cancelled; previously committed labels are preserved")
        raise
    except (RuntimeError, sqlite3.Error):
        LOG.exception("Fatal database or manifest validation error")
        return 1
    finally:
        connection.close()

    LOG.info(
        "Finished requested=%d already_classified=%d attempted=%d updated=%d; "
        "failed rows remain NULL and can be retried",
        progress.requested,
        progress.already_classified,
        progress.attempted,
        progress.updated,
    )
    if progress.classification_errors or progress.database_errors:
        LOG.error(
            "Job incomplete: classification_errors=%d database_errors=%d; "
            "rerun the same manifest to retry remaining NULL rows",
            progress.classification_errors,
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
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        extra_body=args.extra_body,
    )
    try:
        classifier = FullTextClassifier(
            backend=backend,
            system_prompt_path=args.system_prompt_path,
            user_template_path=args.user_template_path,
        )
        return await run_with_classifier(args, db_path, classifier)
    finally:
        try:
            backend.close()
        finally:
            await backend.aclose()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        return asyncio.run(run_async(args))
    except KeyboardInterrupt:
        LOG.warning("Interrupted; previously committed labels are preserved")
        return 130
    except Exception:
        LOG.exception("Fatal unhandled error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
