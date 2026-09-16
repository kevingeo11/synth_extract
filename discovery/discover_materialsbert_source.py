#!/usr/bin/env python3
"""Extract MaterialBERT property names for every pending paper in one source.

Rows are read from ``polymer_material_papers`` in the discovery workspace.
Only rows matching ``--source`` whose ``materialsbert_pty`` value is NULL are
processed. Successful property-name lists are stored as compact JSON arrays;
missing Markdown and inference failures remain NULL so the command is safe to
rerun.

Each spawned worker loads its own ``pranav-s/PolymerNER`` pipeline. All SQLite
writes happen in the parent process in short transactions.

Example
-------
python discovery/discover_materialsbert_source.py \
    --source elsevier \
    --db-path data/discovery_workspace.db \
    --fulltext-root data/fulltext \
    --workers 8 \
    --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("materialsbert_property_discovery")
TABLE = "polymer_material_papers"
RESULT_COLUMN = "materialsbert_pty"
DEFAULT_MODEL = "pranav-s/PolymerNER"

_NER_PIPELINE: Any | None = None
_MAX_CHUNK_TOKENS = 450
_INFERENCE_BATCH_SIZE = 64


@dataclass(frozen=True)
class PaperTask:
    paper_id: int
    paper_uid: str
    markdown_path: str


@dataclass(frozen=True)
class PaperResult:
    paper_id: int
    paper_uid: str
    properties: list[str] | None
    chunks: int
    latency_seconds: float
    error: str | None = None
    markdown_missing: bool = False


@dataclass
class Progress:
    source_rows: int = 0
    already_completed: int = 0
    attempted: int = 0
    updated: int = 0
    properties: int = 0
    chunks: int = 0
    missing_markdown: int = 0
    inference_errors: int = 0
    database_errors: int = 0


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MaterialBERT PROP_NAME entities for all pending papers "
            "from one canonical source and update discovery_workspace.db."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="canonical_source to process, for example elsevier or arxiv",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=REPO_ROOT / "data" / "discovery_workspace.db",
        help="SQLite discovery workspace (default: %(default)s)",
    )
    parser.add_argument(
        "--fulltext-root",
        type=Path,
        default=REPO_ROOT / "data" / "fulltext",
        help="Root containing source/paper_uid/paper_uid.md (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face token-classification model (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help=(
            "Spawned model workers; each loads a separate model copy "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=64,
        help="Transformer inference batch size per worker (default: %(default)s)",
    )
    parser.add_argument(
        "--paper-batch-size",
        type=positive_int,
        default=8,
        help="Papers grouped into each worker task (default: %(default)s)",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=positive_int,
        default=450,
        help="Tokens per text chunk before model special tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Transformers pipeline device; use -1 for CPU (default: %(default)s)",
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


def validate_table(connection: sqlite3.Connection) -> None:
    columns = connection.execute(
        f"PRAGMA table_info({quote_identifier(TABLE)})"
    ).fetchall()
    if not columns:
        raise RuntimeError(f"Table {TABLE!r} does not exist or has no columns")

    by_name = {str(column[1]): column for column in columns}
    required = {"paper_id", "paper_uid", "canonical_source", RESULT_COLUMN}
    missing = sorted(required.difference(by_name))
    if missing:
        raise RuntimeError(
            f"Table {TABLE!r} is missing required columns: {', '.join(missing)}"
        )

    declared_type = str(by_name[RESULT_COLUMN][2] or "").upper()
    if "TEXT" not in declared_type:
        raise RuntimeError(
            f"Column {RESULT_COLUMN!r} must be TEXT, not {declared_type!r}"
        )


def validate_stored_property_json(value: Any, paper_uid: str) -> None:
    if not isinstance(value, str):
        raise RuntimeError(
            f"Stored {RESULT_COLUMN} for paper_uid={paper_uid} is not text"
        )
    try:
        properties = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Stored {RESULT_COLUMN} for paper_uid={paper_uid} is invalid JSON"
        ) from exc
    if not isinstance(properties, list) or not all(
        isinstance(item, str) and item.strip() for item in properties
    ):
        raise RuntimeError(
            f"Stored {RESULT_COLUMN} for paper_uid={paper_uid} must be a JSON "
            "array of non-empty strings"
        )


def load_source_tasks(
    connection: sqlite3.Connection,
    source: str,
    fulltext_root: Path,
) -> tuple[list[PaperTask], int, int]:
    rows = connection.execute(
        f"""
        SELECT
            paper_id,
            paper_uid,
            {quote_identifier(RESULT_COLUMN)}
        FROM {quote_identifier(TABLE)}
        WHERE canonical_source = ?
        ORDER BY paper_id
        """,
        (source,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"No rows found for canonical_source={source!r}")

    tasks: list[PaperTask] = []
    already_completed = 0
    for paper_id, paper_uid, stored_result in rows:
        uid = str(paper_uid or "").strip()
        if not safe_path_component(uid):
            raise RuntimeError(
                f"Unsafe paper_uid for paper_id={paper_id}: {paper_uid!r}"
            )
        if stored_result is not None:
            validate_stored_property_json(stored_result, uid)
            already_completed += 1
            continue
        markdown_path = fulltext_root / source / uid / f"{uid}.md"
        tasks.append(
            PaperTask(
                paper_id=int(paper_id),
                paper_uid=uid,
                markdown_path=str(markdown_path),
            )
        )
    return tasks, already_completed, len(rows)


def batched(items: list[PaperTask], size: int) -> Iterator[list[PaperTask]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def initialize_worker(
    model_name: str,
    device: int,
    max_chunk_tokens: int,
    inference_batch_size: int,
) -> None:
    """Load one token-classification model inside a spawned worker."""
    global _NER_PIPELINE, _MAX_CHUNK_TOKENS, _INFERENCE_BATCH_SIZE

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512,
    )
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    _NER_PIPELINE = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )
    _MAX_CHUNK_TOKENS = max_chunk_tokens
    _INFERENCE_BATCH_SIZE = inference_batch_size


def chunk_text(text: str) -> Iterator[str]:
    if _NER_PIPELINE is None:
        raise RuntimeError("MaterialBERT worker has not been initialized")
    token_ids = _NER_PIPELINE.tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    for start in range(0, len(token_ids), _MAX_CHUNK_TOKENS):
        chunk_ids = token_ids[start : start + _MAX_CHUNK_TOKENS]
        yield _NER_PIPELINE.tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
        )


def process_paper_batch(tasks: list[PaperTask]) -> list[PaperResult]:
    """Read, chunk, infer, and return results for one small paper batch."""
    if _NER_PIPELINE is None:
        raise RuntimeError("MaterialBERT worker has not been initialized")

    started_by_uid: dict[str, float] = {}
    chunks_by_uid: dict[str, int] = {task.paper_uid: 0 for task in tasks}
    task_by_uid = {task.paper_uid: task for task in tasks}
    results_by_uid: dict[str, PaperResult] = {}
    all_chunks: list[str] = []
    chunk_to_uid: list[str] = []

    for task in tasks:
        started_by_uid[task.paper_uid] = time.monotonic()
        markdown_path = Path(task.markdown_path)
        if not markdown_path.is_file():
            results_by_uid[task.paper_uid] = PaperResult(
                paper_id=task.paper_id,
                paper_uid=task.paper_uid,
                properties=None,
                chunks=0,
                latency_seconds=0.0,
                error=f"Markdown file not found: {markdown_path}",
                markdown_missing=True,
            )
            continue

        try:
            markdown = markdown_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            paper_chunks = list(chunk_text(markdown))
        except Exception as exc:
            results_by_uid[task.paper_uid] = PaperResult(
                paper_id=task.paper_id,
                paper_uid=task.paper_uid,
                properties=None,
                chunks=0,
                latency_seconds=time.monotonic() - started_by_uid[task.paper_uid],
                error=f"{type(exc).__name__}: {exc}",
            )
            continue

        chunks_by_uid[task.paper_uid] = len(paper_chunks)
        all_chunks.extend(paper_chunks)
        chunk_to_uid.extend([task.paper_uid] * len(paper_chunks))

    properties_by_uid: dict[str, list[str]] = {
        task.paper_uid: []
        for task in tasks
        if task.paper_uid not in results_by_uid
    }
    if all_chunks:
        try:
            entities_per_chunk = _NER_PIPELINE(
                all_chunks,
                batch_size=_INFERENCE_BATCH_SIZE,
            )
            for paper_uid, entities in zip(
                chunk_to_uid,
                entities_per_chunk,
                strict=True,
            ):
                for entity in entities:
                    if entity.get("entity_group") != "PROP_NAME":
                        continue
                    property_name = entity.get("word")
                    if isinstance(property_name, str) and property_name.strip():
                        properties_by_uid[paper_uid].append(property_name)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for paper_uid in properties_by_uid:
                task = task_by_uid[paper_uid]
                results_by_uid[paper_uid] = PaperResult(
                    paper_id=task.paper_id,
                    paper_uid=paper_uid,
                    properties=None,
                    chunks=chunks_by_uid[paper_uid],
                    latency_seconds=time.monotonic() - started_by_uid[paper_uid],
                    error=error,
                )
            properties_by_uid.clear()

    for paper_uid, properties in properties_by_uid.items():
        task = task_by_uid[paper_uid]
        results_by_uid[paper_uid] = PaperResult(
            paper_id=task.paper_id,
            paper_uid=paper_uid,
            properties=properties,
            chunks=chunks_by_uid[paper_uid],
            latency_seconds=time.monotonic() - started_by_uid[paper_uid],
        )

    return [results_by_uid[task.paper_uid] for task in tasks]


def retryable_sqlite_lock_error(exc: sqlite3.Error) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        primary_code = error_code & 0xFF
        if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message


def write_result_batch(
    connection: sqlite3.Connection,
    source: str,
    results: list[PaperResult],
    progress: Progress,
    retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> None:
    progress.attempted += len(results)
    progress.chunks += sum(result.chunks for result in results)
    candidates: list[tuple[PaperResult, str]] = []

    for result in results:
        prefix = f"source={source} paper_uid={result.paper_uid}"
        if result.properties is None:
            progress.inference_errors += 1
            if result.markdown_missing:
                progress.missing_markdown += 1
            LOG.error(
                "%s extraction_failure=%s chunks=%d latency=%.2fs",
                prefix,
                result.error or "unknown error",
                result.chunks,
                result.latency_seconds,
            )
            continue

        payload = json.dumps(
            result.properties,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        candidates.append((result, payload))
        LOG.info(
            "%s properties=%d chunks=%d latency=%.2fs",
            prefix,
            len(result.properties),
            result.chunks,
            result.latency_seconds,
        )

    if not candidates:
        return

    update_sql = f"""
        UPDATE {quote_identifier(TABLE)}
        SET {quote_identifier(RESULT_COLUMN)} = ?
        WHERE paper_id = ?
          AND paper_uid = ?
          AND canonical_source = ?
          AND {quote_identifier(RESULT_COLUMN)} IS NULL
    """
    applied: list[PaperResult] = []
    skipped: list[PaperResult] = []
    for attempt in range(retries + 1):
        applied = []
        skipped = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            for result, payload in candidates:
                cursor = connection.execute(
                    update_sql,
                    (payload, result.paper_id, result.paper_uid, source),
                )
                if cursor.rowcount == 1:
                    applied.append(result)
                else:
                    skipped.append(result)
            connection.commit()
            break
        except sqlite3.Error as exc:
            connection.rollback()
            if retryable_sqlite_lock_error(exc) and attempt < retries:
                exponential_delay = min(
                    retry_max_delay,
                    retry_base_delay * (2**attempt),
                )
                delay = exponential_delay * random.uniform(0.8, 1.2)
                LOG.warning(
                    "SQLite write lock; retry=%d/%d delay=%.2fs error=%s",
                    attempt + 1,
                    retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue
            progress.database_errors += len(candidates)
            LOG.exception(
                "SQLite transaction failed for %d results after %d attempt(s)",
                len(candidates),
                attempt + 1,
            )
            return

    progress.updated += len(applied)
    progress.properties += sum(len(result.properties or []) for result in applied)
    progress.database_errors += len(skipped)
    for result in skipped:
        LOG.error(
            "Database update skipped paper_id=%d paper_uid=%s; row changed "
            "concurrently or no longer has a NULL result",
            result.paper_id,
            result.paper_uid,
        )


def log_progress(progress: Progress, target: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    LOG.info(
        "Progress attempted=%d/%d updated=%d properties=%d chunks=%d "
        "missing_markdown=%d inference_errors=%d database_errors=%d "
        "rate=%.2f papers/s",
        progress.attempted,
        target,
        progress.updated,
        progress.properties,
        progress.chunks,
        progress.missing_markdown,
        progress.inference_errors,
        progress.database_errors,
        progress.attempted / elapsed,
    )


def run(args: argparse.Namespace) -> int:
    source = args.source.strip()
    if not safe_path_component(source):
        LOG.error("Source must be one safe path component: %r", args.source)
        return 1
    if args.max_chunk_tokens > 510:
        LOG.error("--max-chunk-tokens must be at most 510")
        return 1

    db_path = args.db_path.expanduser().resolve()
    fulltext_root = args.fulltext_root.expanduser().resolve()
    if not db_path.is_file():
        LOG.error("Database does not exist: %s", db_path)
        return 1
    if not fulltext_root.is_dir():
        LOG.error("Full-text root does not exist: %s", fulltext_root)
        return 1
    source_root = fulltext_root / source
    if not source_root.is_dir():
        LOG.error("Full-text source directory does not exist: %s", source_root)
        return 1

    connection = sqlite3.connect(
        db_path,
        timeout=args.sqlite_timeout,
        isolation_level=None,
    )
    connection.execute(
        f"PRAGMA busy_timeout = {int(args.sqlite_timeout * 1000)}"
    )
    started = time.monotonic()
    progress = Progress()

    try:
        validate_table(connection)
        tasks, already_completed, source_rows = load_source_tasks(
            connection,
            source,
            fulltext_root,
        )
        progress.source_rows = source_rows
        progress.already_completed = already_completed
        target = len(tasks)
        LOG.info(
            "Source=%s rows=%d already_completed=%d pending=%d database=%s "
            "result_column=%s fulltext_root=%s model=%s workers=%d "
            "batch_size=%d paper_batch_size=%d max_chunk_tokens=%d device=%d",
            source,
            source_rows,
            already_completed,
            target,
            db_path,
            RESULT_COLUMN,
            fulltext_root,
            args.model,
            args.workers,
            args.batch_size,
            args.paper_batch_size,
            args.max_chunk_tokens,
            args.device,
        )
        if not tasks:
            LOG.info("No pending rows for source=%s", source)
            return 0

        task_batches = batched(tasks, args.paper_batch_size)
        initializer_args = (
            args.model,
            args.device,
            args.max_chunk_tokens,
            args.batch_size,
        )

        if args.workers == 1:
            initialize_worker(*initializer_args)
            result_batches = map(process_paper_batch, task_batches)
            for results in result_batches:
                write_result_batch(
                    connection,
                    source,
                    results,
                    progress,
                    args.sqlite_write_retries,
                    args.sqlite_retry_base_delay,
                    args.sqlite_retry_max_delay,
                )
                log_progress(progress, target, started)
        else:
            context = get_context("spawn")
            with context.Pool(
                processes=args.workers,
                initializer=initialize_worker,
                initargs=initializer_args,
            ) as pool:
                for results in pool.imap_unordered(
                    process_paper_batch,
                    task_batches,
                    chunksize=1,
                ):
                    write_result_batch(
                        connection,
                        source,
                        results,
                        progress,
                        args.sqlite_write_retries,
                        args.sqlite_retry_base_delay,
                        args.sqlite_retry_max_delay,
                    )
                    log_progress(progress, target, started)

    except (OSError, RuntimeError, sqlite3.Error):
        LOG.exception("Fatal source processing or database error")
        return 1
    finally:
        connection.close()

    elapsed = time.monotonic() - started
    LOG.info(
        "Finished source=%s rows=%d already_completed=%d attempted=%d "
        "updated=%d properties=%d chunks=%d elapsed=%.1fs rate=%.2f papers/s",
        source,
        progress.source_rows,
        progress.already_completed,
        progress.attempted,
        progress.updated,
        progress.properties,
        progress.chunks,
        elapsed,
        progress.attempted / max(elapsed, 1e-9),
    )
    if progress.inference_errors or progress.database_errors:
        LOG.error(
            "Job incomplete: inference_errors=%d database_errors=%d; failed "
            "rows remain NULL and will be retried on the next run",
            progress.inference_errors,
            progress.database_errors,
        )
        return 2
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        return run(args)
    except KeyboardInterrupt:
        LOG.warning("Interrupted; previously committed results are preserved")
        return 130
    except Exception:
        LOG.exception("Fatal unhandled error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
