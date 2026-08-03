#!/usr/bin/env python3
"""Classify pending papers and store binary labels in the central SQLite database."""

from __future__ import annotations

import argparse
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
    PaperClassifier,
)

LOG = logging.getLogger("paper_classification")
TABLE = "papers"
QWEN_PREFIX = "qwen"


@dataclass
class Progress:
    attempted: int = 0
    updated: int = 0
    true: int = 0
    false: int = 0
    classification_errors: int = 0
    database_errors: int = 0


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify rows whose qwen-prefixed result column is NULL. "
            "Successful True/False labels become 1/0; errors remain NULL."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=REPO_ROOT / "data" / "central_workspace.db",
        help="SQLite database path (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "qwen3.6-27b"),
        help="Model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", "not-required"),
        help="API key; local vLLM normally ignores it",
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
        default=100,
        help="Rows fetched from SQLite per batch (default: %(default)s)",
    )
    parser.add_argument(
        "--log-every",
        type=positive_int,
        default=10,
        help="Progress interval in attempted rows (default: %(default)s)",
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
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file; logs are always written to stderr",
    )
    return parser.parse_args()


def configure_logging(level: str, log_file: Path | None) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        path = log_file.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=handlers,
        force=True,
    )


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def find_qwen_column(connection: sqlite3.Connection) -> str:
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

    matches = [name for name in names if name.lower().startswith(QWEN_PREFIX)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one column beginning with {QWEN_PREFIX!r}; found {matches}"
        )
    return matches[0]


def normalize(value: object) -> str:
    return "" if value is None else str(value)


def log_progress(progress: Progress, target: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    LOG.info(
        "Progress attempted=%d/%d updated=%d true=%d false=%d "
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


def run(args: argparse.Namespace) -> int:
    db_path = args.db_path.expanduser().resolve()
    if not db_path.is_file():
        LOG.error("Database does not exist: %s", db_path)
        return 1

    classifier = PaperClassifier(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    config = classifier.llm_config()
    LOG.info(
        "Classifier model=%s base_url=%s timeout=%ss max_tokens=%s prompt_hash=%s",
        config["model"],
        config["base_url"],
        config["timeout"],
        config["max_tokens"],
        config["prompt_hash"],
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

    connection = sqlite3.connect(db_path, timeout=args.sqlite_timeout)
    connection.execute(f"PRAGMA busy_timeout = {int(args.sqlite_timeout * 1000)}")
    progress = Progress()
    started = time.monotonic()
    target = 0

    try:
        result_column = find_qwen_column(connection)
        table_sql = quote_identifier(TABLE)
        column_sql = quote_identifier(result_column)
        pending = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table_sql} WHERE {column_sql} IS NULL"
            ).fetchone()[0]
        )
        target = pending if args.limit is None else min(pending, args.limit)
        LOG.info(
            "Starting database=%s table=%s result_column=%s pending=%d target=%d",
            db_path,
            TABLE,
            result_column,
            pending,
            target,
        )

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
        last_paper_id = -1

        while progress.attempted < target:
            fetch_size = min(args.batch_size, target - progress.attempted)
            rows = connection.execute(
                select_sql,
                (last_paper_id, fetch_size),
            ).fetchall()
            if not rows:
                break

            for paper_id, paper_uid, title_value, abstract_value in rows:
                last_paper_id = int(paper_id)
                progress.attempted += 1
                title = normalize(title_value)
                abstract = normalize(abstract_value)

                try:
                    result = classifier.classify(title=title, abstract=abstract)
                except Exception:
                    progress.classification_errors += 1
                    LOG.exception(
                        "Unexpected classifier error paper_id=%s paper_uid=%s; "
                        "row was not updated",
                        paper_id,
                        paper_uid,
                    )
                    result = None

                if isinstance(result, ClassificationFailure):
                    progress.classification_errors += 1
                    LOG.error(
                        "Classification failed paper_id=%s paper_uid=%s "
                        "error_type=%s message=%s; row was not updated",
                        paper_id,
                        paper_uid,
                        result.error_type,
                        result.message,
                    )
                elif isinstance(result, ClassificationResult):
                    label = int(result.label)
                    try:
                        cursor = connection.execute(update_sql, (label, paper_id))
                        if cursor.rowcount != 1:
                            connection.rollback()
                            progress.database_errors += 1
                            LOG.error(
                                "Update skipped paper_id=%s paper_uid=%s rowcount=%d; "
                                "the row may have changed concurrently",
                                paper_id,
                                paper_uid,
                                cursor.rowcount,
                            )
                        else:
                            connection.commit()
                            progress.updated += 1
                            progress.true += label
                            progress.false += 1 - label
                            LOG.debug(
                                "Updated paper_id=%s paper_uid=%s label=%d",
                                paper_id,
                                paper_uid,
                                label,
                            )
                    except sqlite3.Error:
                        connection.rollback()
                        progress.database_errors += 1
                        LOG.exception(
                            "Database update failed paper_id=%s paper_uid=%s; "
                            "row was not updated",
                            paper_id,
                            paper_uid,
                        )

                if (
                    progress.attempted % args.log_every == 0
                    or progress.attempted == target
                ):
                    log_progress(progress, target, started)

    except KeyboardInterrupt:
        connection.rollback()
        LOG.warning("Interrupted; previously committed labels are preserved")
        log_progress(progress, target, started)
        return 130
    except (RuntimeError, sqlite3.Error):
        connection.rollback()
        LOG.exception("Fatal database processing error")
        return 1
    finally:
        connection.close()

    log_progress(progress, target, started)
    LOG.info("Finished. Failed rows remain NULL and can be retried.")
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level, args.log_file)
    try:
        return run(args)
    except Exception:
        LOG.exception("Fatal unhandled error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
