#!/usr/bin/env python3
"""Classify pending papers concurrently and persist labels with one SQLite writer."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synth_extract.agents.classification import (  # noqa: E402
    ClassificationFailure,
    ClassificationResult,
    TitleAbstractClassifier,
)
from synth_extract.agents.llm import LLMBackend  # noqa: E402

LOG = logging.getLogger("paper_classification_async")
TABLE = "papers"
STOP_WRITER = object()


@dataclass
class Progress:
    attempted: int = 0
    updated: int = 0
    true: int = 0
    false: int = 0
    classification_errors: int = 0
    database_errors: int = 0


@dataclass(frozen=True)
class ClassificationItem:
    paper_id: int
    paper_uid: str
    result: ClassificationResult | ClassificationFailure | None
    latency_seconds: float
    unexpected_error: str | None = None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify pending rows concurrently. Completed results flow through "
            "one queue to one SQLite writer; failures remain NULL."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=REPO_ROOT / "data" / "central_workspace.db",
        help="SQLite database path (default: %(default)s)",
    )
    parser.add_argument(
        "--result-column",
        required=True,
        help="Existing database column in which classification labels are stored",
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
        required=True,
        help="API key, or a placeholder value for servers that do not authenticate",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum pending rows to attempt (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1000,
        help="Pending rows selected and launched per read batch (default: %(default)s)",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=positive_int,
        default=int(os.getenv("MAX_PARALLEL_REQUESTS", "8")),
        help="Maximum in-flight model requests (default: %(default)s)",
    )
    parser.add_argument(
        "--write-batch-size",
        type=positive_int,
        default=50,
        help="Maximum completed results grouped per writer transaction (default: %(default)s)",
    )
    parser.add_argument(
        "--write-flush-seconds",
        type=nonnegative_float,
        default=0.25,
        help="Maximum wait to fill a partial write batch (default: %(default)s)",
    )
    parser.add_argument(
        "--log-every",
        type=positive_int,
        default=100,
        help="Progress interval in completed requests (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=60.0,
        help="Model request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=10_000,
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
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=handlers,
        force=True,
    )


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def validate_result_column(
    connection: sqlite3.Connection,
    result_column: str,
) -> str:
    columns = connection.execute(
        f"PRAGMA table_info({quote_identifier(TABLE)})"
    ).fetchall()
    if not columns:
        raise RuntimeError(f"Table {TABLE!r} does not exist or has no columns")

    names = [str(column[1]) for column in columns]
    required = {"paper_id", "paper_uid", "title", "abstract"}
    missing = sorted(required.difference(names))
    if missing:
        raise RuntimeError(
            f"Table {TABLE!r} is missing required columns: {', '.join(missing)}"
        )

    if result_column not in names:
        raise RuntimeError(
            f"Result column {result_column!r} does not exist in table {TABLE!r}"
        )
    return result_column


def normalize(value: object) -> str:
    return "" if value is None else str(value)


def log_progress(progress: Progress, target: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    LOG.info(
        "Progress completed=%d/%d updated=%d true=%d false=%d "
        "classification_errors=%d database_errors=%d rate=%.2f papers/s",
        progress.attempted,
        target,
        progress.updated,
        progress.true,
        progress.false,
        progress.classification_errors,
        progress.database_errors,
        progress.attempted / elapsed,
    )


async def classify_and_enqueue(
    classifier: TitleAbstractClassifier,
    semaphore: asyncio.Semaphore,
    result_queue: asyncio.Queue[ClassificationItem | object],
    paper_id: int,
    paper_uid: str,
    title: str,
    abstract: str,
) -> None:
    async with semaphore:
        request_started = time.monotonic()
        try:
            result = await classifier.aclassify(title=title, abstract=abstract)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = None
            unexpected_error = f"{type(exc).__name__}: {exc}"
        else:
            unexpected_error = None
        latency_seconds = time.monotonic() - request_started

    await result_queue.put(
        ClassificationItem(
            paper_id=paper_id,
            paper_uid=paper_uid,
            result=result,
            latency_seconds=latency_seconds,
            unexpected_error=unexpected_error,
        )
    )


def write_result_batch(
    connection: sqlite3.Connection,
    update_sql: str,
    items: list[ClassificationItem],
    progress: Progress,
) -> None:
    """Write one queue batch in one transaction and update progress counters."""
    progress.attempted += len(items)
    candidates: list[tuple[ClassificationItem, int]] = []

    for item in items:
        result = item.result
        if isinstance(result, ClassificationFailure):
            progress.classification_errors += 1
            LOG.error(
                "Classification failed paper_id=%s paper_uid=%s "
                "error_type=%s message=%s; row was not updated",
                item.paper_id,
                item.paper_uid,
                result.error_type,
                result.message,
            )
        elif isinstance(result, ClassificationResult):
            candidates.append((item, int(result.label)))
        else:
            progress.classification_errors += 1
            LOG.error(
                "Unexpected classifier error paper_id=%s paper_uid=%s "
                "message=%s; row was not updated",
                item.paper_id,
                item.paper_uid,
                item.unexpected_error or "unknown error",
            )

    if not candidates:
        return

    applied: list[tuple[ClassificationItem, int]] = []
    skipped: list[ClassificationItem] = []
    try:
        connection.execute("BEGIN")
        for item, label in candidates:
            cursor = connection.execute(update_sql, (label, item.paper_id))
            if cursor.rowcount == 1:
                applied.append((item, label))
            else:
                skipped.append(item)
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        progress.database_errors += len(candidates)
        LOG.exception(
            "SQLite transaction failed for %d classification results; "
            "the transaction was rolled back",
            len(candidates),
        )
        return

    progress.updated += len(applied)
    progress.true += sum(label for _, label in applied)
    progress.false += sum(1 - label for _, label in applied)
    progress.database_errors += len(skipped)

    for item in skipped:
        LOG.error(
            "Update skipped paper_id=%s paper_uid=%s; "
            "the row may have changed concurrently",
            item.paper_id,
            item.paper_uid,
        )
    LOG.debug(
        "Committed result batch received=%d candidates=%d updated=%d skipped=%d",
        len(items),
        len(candidates),
        len(applied),
        len(skipped),
    )


async def result_writer(
    db_path: Path,
    sqlite_timeout: float,
    update_sql: str,
    result_queue: asyncio.Queue[ClassificationItem | object],
    write_batch_size: int,
    write_flush_seconds: float,
    progress: Progress,
    target: int,
    started: float,
    log_every: int,
) -> None:
    """Consume completed classifications and commit them through one connection."""
    connection = sqlite3.connect(db_path, timeout=sqlite_timeout)
    connection.execute(f"PRAGMA busy_timeout = {int(sqlite_timeout * 1000)}")
    last_logged = 0
    stop_requested = False

    try:
        while not stop_requested:
            first = await result_queue.get()
            if first is STOP_WRITER:
                result_queue.task_done()
                break

            items = [first]
            deadline = asyncio.get_running_loop().time() + write_flush_seconds

            while len(items) < write_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(
                        result_queue.get(),
                        timeout=remaining,
                    )
                except TimeoutError:
                    break

                if next_item is STOP_WRITER:
                    result_queue.task_done()
                    stop_requested = True
                    break
                items.append(next_item)

            write_result_batch(connection, update_sql, items, progress)
            for _ in items:
                result_queue.task_done()

            if (
                progress.attempted - last_logged >= log_every
                or progress.attempted == target
            ):
                log_progress(progress, target, started)
                last_logged = progress.attempted
    finally:
        connection.close()


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
    read_connection: sqlite3.Connection | None = None
    writer_task: asyncio.Task[None] | None = None
    result_queue: asyncio.Queue[ClassificationItem | object] | None = None
    progress = Progress()
    started = time.monotonic()
    target = 0

    try:
        classifier = TitleAbstractClassifier(backend=backend)
        config = classifier.llm_config()
        LOG.info(
            "Classifier model=%s base_url=%s timeout=%ss max_tokens=%s "
            "prompt_hash=%s extra_body=%s max_parallel_requests=%d",
            config["model"],
            config["base_url"],
            config["timeout"],
            config["max_tokens"],
            config["prompt_hash"],
            config["extra_body"],
            args.max_parallel_requests,
        )
        health = classifier.health_check()
        if isinstance(health, ClassificationFailure):
            LOG.error(
                "Health check failed error_type=%s message=%s",
                health.error_type,
                health.message,
            )
            return 1
        LOG.info("Classifier health check passed")

        database_uri = f"file:{db_path}?mode=ro"
        read_connection = sqlite3.connect(
            database_uri,
            uri=True,
            timeout=args.sqlite_timeout,
        )
        read_connection.execute(
            f"PRAGMA busy_timeout = {int(args.sqlite_timeout * 1000)}"
        )
        result_column = validate_result_column(
            read_connection,
            args.result_column,
        )
        table_sql = quote_identifier(TABLE)
        column_sql = quote_identifier(result_column)
        pending = int(
            read_connection.execute(
                f"SELECT COUNT(*) FROM {table_sql} WHERE {column_sql} IS NULL"
            ).fetchone()[0]
        )
        target = pending if args.limit is None else min(pending, args.limit)
        LOG.info(
            "Starting database=%s table=%s result_column=%s pending=%d target=%d "
            "read_batch_size=%d write_batch_size=%d",
            db_path,
            TABLE,
            result_column,
            pending,
            target,
            args.batch_size,
            args.write_batch_size,
        )
        if target == 0:
            LOG.info("No pending rows to classify")
            return 0

        select_sql = f"""
            SELECT paper_id, paper_uid, title, abstract
            FROM {table_sql}
            WHERE {column_sql} IS NULL AND paper_id > ?
            ORDER BY paper_id
            LIMIT ?
        """
        update_sql = f"""
            UPDATE {table_sql}
            SET {column_sql} = ?
            WHERE paper_id = ? AND {column_sql} IS NULL
        """
        result_queue = asyncio.Queue()
        writer_task = asyncio.create_task(
            result_writer(
                db_path=db_path,
                sqlite_timeout=args.sqlite_timeout,
                update_sql=update_sql,
                result_queue=result_queue,
                write_batch_size=args.write_batch_size,
                write_flush_seconds=args.write_flush_seconds,
                progress=progress,
                target=target,
                started=started,
                log_every=args.log_every,
            ),
            name="sqlite-result-writer",
        )
        semaphore = asyncio.Semaphore(args.max_parallel_requests)
        scheduled = 0
        last_paper_id = -1

        while scheduled < target:
            fetch_size = min(args.batch_size, target - scheduled)
            rows = read_connection.execute(
                select_sql,
                (last_paper_id, fetch_size),
            ).fetchall()
            if not rows:
                break

            last_paper_id = int(rows[-1][0])
            scheduled += len(rows)
            LOG.info(
                "Selected batch rows=%d scheduled=%d/%d last_paper_id=%d",
                len(rows),
                scheduled,
                target,
                last_paper_id,
            )
            tasks = [
                asyncio.create_task(
                    classify_and_enqueue(
                        classifier=classifier,
                        semaphore=semaphore,
                        result_queue=result_queue,
                        paper_id=int(paper_id),
                        paper_uid=normalize(paper_uid),
                        title=normalize(title_value),
                        abstract=normalize(abstract_value),
                    )
                )
                for paper_id, paper_uid, title_value, abstract_value in rows
            ]
            await asyncio.gather(*tasks)
            if writer_task.done():
                await writer_task

        if scheduled < target:
            LOG.warning(
                "Selected only %d of %d target rows; no more pending rows were found",
                scheduled,
                target,
            )

        await result_queue.put(STOP_WRITER)
        await writer_task
        writer_task = None
        await result_queue.join()

    except asyncio.CancelledError:
        LOG.warning("Cancelled; previously committed labels are preserved")
        raise
    except (RuntimeError, sqlite3.Error):
        LOG.exception("Fatal database processing error")
        return 1
    finally:
        if writer_task is not None and not writer_task.done():
            writer_task.cancel()
            await asyncio.gather(writer_task, return_exceptions=True)
        if read_connection is not None:
            read_connection.close()
        try:
            backend.close()
        finally:
            await backend.aclose()

    log_progress(progress, target, started)
    LOG.info("Finished. Failed rows remain NULL and can be retried.")
    return 0


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
