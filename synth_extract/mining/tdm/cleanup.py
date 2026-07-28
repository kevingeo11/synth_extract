#!/usr/bin/env python3
"""Audit successful full-text paths and clean failed/exhausted UID folders.

Successful rows are read-only. The script verifies that their recorded file
path, source, UID directory, filename, and format agree and that both the
directory and file exist. Every mismatch is logged individually.

For ``failed`` and ``exhausted`` rows, no local UID directory should remain.
The script checks the relevant source directories beneath ``--fulltext-root``,
logs every unexpected path, and removes it. Use ``--dry-run`` to report what
would be removed without changing the filesystem.

The central database is always opened read-only and is never updated.

Usage
-----
    python -m synth_extract.mining.tdm.cleanup \
        --db data/central_papers.db \
        --fulltext-root data/fulltext \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Optional

DEFAULT_TABLE = "papers"
DEFAULT_FULLTEXT_ROOT = Path("data/fulltext")
DEFAULT_PATH_BASE = Path(".")
FETCH_BATCH_SIZE = 10_000

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")

logger = logging.getLogger("tdm_cleanup")


# --------------------------------------------------------------------------- #
# Logging and helpers
# --------------------------------------------------------------------------- #


def setup_logging(level: str = "INFO") -> None:
    """Configure timestamped logging to stdout for SLURM capture."""
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


def _load_source_list(
    raw: Optional[str],
    paper_uid: str,
    field_name: str,
    stats: "CleanupStats",
) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        stats.metadata_issues += 1
        logger.error(
            "%s | INVALID_%s | Could not parse JSON: %s",
            paper_uid,
            field_name.upper(),
            exc,
        )
        return []
    if not isinstance(value, list):
        stats.metadata_issues += 1
        logger.error(
            "%s | INVALID_%s | Expected a JSON list.",
            paper_uid,
            field_name.upper(),
        )
        return []
    return [
        source.strip()
        for source in value
        if isinstance(source, str) and source.strip()
    ]


def _safe_component(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and _SAFE_COMPONENT_RE.fullmatch(value) is not None
    )


def _resolved_stored_path(raw_path: str, path_base: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = path_base / path
    return path.resolve(strict=False)


def _path_contains_expected_segment(
    raw_path: str,
    source: str,
    paper_uid: str,
) -> bool:
    stored_path = PurePosixPath(raw_path.replace("\\", "/"))
    parts = stored_path.parts
    expected = ("data", "fulltext", source, paper_uid)
    return any(
        tuple(parts[index : index + len(expected)]) == expected
        for index in range(len(parts) - len(expected) + 1)
    )


def _read_rows(
    conn: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
) -> Iterator[sqlite3.Row]:
    cursor = conn.execute(query, params)
    while True:
        rows = cursor.fetchmany(FETCH_BATCH_SIZE)
        if not rows:
            return
        yield from rows


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


@dataclass
class CleanupStats:
    success_checked: int = 0
    success_mismatch_rows: int = 0
    success_issue_counts: Counter[str] = field(
        default_factory=Counter
    )
    non_success_checked: int = 0
    unexpected_paths: int = 0
    removed_paths: int = 0
    would_remove_paths: int = 0
    removal_failures: int = 0
    metadata_issues: int = 0


# --------------------------------------------------------------------------- #
# Successful-row audit
# --------------------------------------------------------------------------- #


def audit_success_row(
    row: sqlite3.Row,
    fulltext_root: Path,
    path_base: Path,
    stats: CleanupStats,
) -> None:
    stats.success_checked += 1
    paper_uid = row["paper_uid"]
    canonical_source = (row["canonical_source"] or "").strip()
    downloaded_from = (row["downloaded_from"] or "").strip()
    raw_path = (row["fulltext_path"] or "").strip()
    fulltext_format = (
        (row["fulltext_format"] or "").strip().lower().lstrip(".")
    )
    issues: list[tuple[str, str]] = []

    if not downloaded_from:
        issues.append(
            (
                "MISSING_DOWNLOADED_FROM",
                "downloaded_from is empty",
            )
        )
    elif canonical_source != downloaded_from:
        issues.append(
            (
                "SOURCE_MISMATCH",
                f"canonical_source={canonical_source!r}, "
                f"downloaded_from={downloaded_from!r}",
            )
        )

    if not raw_path:
        issues.append(("MISSING_PATH", "fulltext_path is empty"))
    else:
        stored_path = _resolved_stored_path(raw_path, path_base)

        if not stored_path.parent.is_dir():
            issues.append(
                (
                    "MISSING_FOLDER",
                    f"recorded folder does not exist: "
                    f"{stored_path.parent}",
                )
            )
        if not stored_path.is_file():
            issues.append(
                (
                    "MISSING_FILE",
                    f"recorded file does not exist: {stored_path}",
                )
            )

        if downloaded_from and not _path_contains_expected_segment(
            raw_path,
            downloaded_from,
            paper_uid,
        ):
            issues.append(
                (
                    "PATH_SOURCE_MISMATCH",
                    "path does not contain expected segment "
                    f"data/fulltext/{downloaded_from}/{paper_uid}",
                )
            )

        if not fulltext_format:
            issues.append(
                ("MISSING_FORMAT", "fulltext_format is empty")
            )
        else:
            expected_filename = f"{paper_uid}.{fulltext_format}"
            if stored_path.name != expected_filename:
                issues.append(
                    (
                        "FILENAME_FORMAT_MISMATCH",
                        f"expected filename {expected_filename!r}, "
                        f"found {stored_path.name!r}",
                    )
                )

        if downloaded_from and _safe_component(downloaded_from):
            expected_path = (
                fulltext_root
                / downloaded_from
                / paper_uid
                / f"{paper_uid}.{fulltext_format}"
            )
            if (
                fulltext_format
                and stored_path != expected_path.resolve(strict=False)
            ):
                issues.append(
                    (
                        "RECORDED_PATH_MISMATCH",
                        f"recorded path resolves to {stored_path}, expected "
                        f"{expected_path.resolve(strict=False)}",
                    )
                )

    if issues:
        stats.success_mismatch_rows += 1
        for issue_code, detail in issues:
            stats.success_issue_counts[issue_code] += 1
        issue_text = " | ".join(
            f"{code}: {detail}" for code, detail in issues
        )
        logger.error(
            "%s | SUCCESS_PATH_MISMATCH | %s",
            paper_uid,
            issue_text,
        )


# --------------------------------------------------------------------------- #
# Failed/exhausted cleanup
# --------------------------------------------------------------------------- #


def _sources_to_check(
    row: sqlite3.Row,
    stats: CleanupStats,
) -> set[str]:
    paper_uid = row["paper_uid"]
    sources = {
        source
        for source in (
            row["canonical_source"],
            row["last_attempted_source"],
        )
        if isinstance(source, str) and source.strip()
    }
    sources.update(
        _load_source_list(
            row["attempted_sources"],
            paper_uid,
            "attempted_sources",
            stats,
        )
    )

    # Exhausted rows should have no local file under any candidate source.
    if row["download_status"] == "exhausted":
        sources.update(
            _load_source_list(
                row["sources"],
                paper_uid,
                "sources",
                stats,
            )
        )
    return {source.strip() for source in sources}


def _remove_unexpected_path(
    path: Path,
    paper_uid: str,
    status: str,
    source: str,
    dry_run: bool,
    stats: CleanupStats,
) -> None:
    if not path.exists() and not path.is_symlink():
        return

    stats.unexpected_paths += 1
    action = "would remove" if dry_run else "removing"
    logger.warning(
        "%s | UNEXPECTED_NON_SUCCESS_PATH | status=%s source=%s "
        "path=%s; %s.",
        paper_uid,
        status,
        source,
        path,
        action,
    )

    if dry_run:
        stats.would_remove_paths += 1
        return

    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        stats.removal_failures += 1
        logger.error(
            "%s | REMOVE_FAILED | status=%s source=%s path=%s "
            "error=%s",
            paper_uid,
            status,
            source,
            path,
            exc,
        )
        return

    stats.removed_paths += 1
    logger.info(
        "%s | REMOVED | status=%s source=%s path=%s",
        paper_uid,
        status,
        source,
        path,
    )


def cleanup_non_success_row(
    row: sqlite3.Row,
    fulltext_root: Path,
    dry_run: bool,
    stats: CleanupStats,
) -> None:
    stats.non_success_checked += 1
    paper_uid = row["paper_uid"]
    status = row["download_status"]

    if not _safe_component(paper_uid):
        stats.metadata_issues += 1
        logger.error(
            "%s | UNSAFE_PAPER_UID | Refusing filesystem cleanup.",
            paper_uid,
        )
        return

    raw_path = (row["fulltext_path"] or "").strip()
    if raw_path:
        stats.metadata_issues += 1
        logger.error(
            "%s | NON_SUCCESS_HAS_RECORDED_PATH | status=%s "
            "fulltext_path=%s",
            paper_uid,
            status,
            raw_path,
        )

    for source in sorted(_sources_to_check(row, stats)):
        if not _safe_component(source):
            stats.metadata_issues += 1
            logger.error(
                "%s | UNSAFE_SOURCE | status=%s source=%r; refusing "
                "filesystem cleanup.",
                paper_uid,
                status,
                source,
            )
            continue

        uid_path = fulltext_root / source / paper_uid
        _remove_unexpected_path(
            uid_path,
            paper_uid,
            status,
            source,
            dry_run,
            stats,
        )


# --------------------------------------------------------------------------- #
# CLI and main
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit successful TDM paths and remove local UID folders for "
            "failed/exhausted rows."
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
        "--fulltext-root",
        type=Path,
        default=DEFAULT_FULLTEXT_ROOT,
        help=(
            "Root containing source/UID full-text directories "
            f"(default: {DEFAULT_FULLTEXT_ROOT})."
        ),
    )
    parser.add_argument(
        "--path-base",
        type=Path,
        default=DEFAULT_PATH_BASE,
        help=(
            "Base used to resolve relative fulltext_path values "
            f"(default: current directory)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log unexpected paths without removing them.",
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

    if not args.db.is_file():
        logger.error("Central database not found at %s", args.db)
        return 2
    if not _TABLE_NAME_RE.fullmatch(args.table):
        logger.error("Unsafe table name: %r", args.table)
        return 2

    fulltext_root = args.fulltext_root.resolve(strict=False)
    path_base = args.path_base.resolve(strict=False)
    if not fulltext_root.is_dir():
        logger.error(
            "Full-text root is not a directory: %s", fulltext_root
        )
        return 2

    stats = CleanupStats()
    uri = f"{args.db.resolve().as_uri()}?mode=ro"

    try:
        with closing(
            sqlite3.connect(
                uri,
                uri=True,
                timeout=30,
                isolation_level=None,
            )
        ) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")

            logger.info("Auditing successful database rows.")
            success_query = f"""
                SELECT paper_uid, canonical_source, downloaded_from,
                       fulltext_path, fulltext_format
                FROM {args.table}
                WHERE download_status = 'success'
                ORDER BY paper_uid
            """
            for row in _read_rows(conn, success_query):
                audit_success_row(
                    row,
                    fulltext_root,
                    path_base,
                    stats,
                )

            logger.info(
                "Checking failed and exhausted rows for unexpected paths.%s",
                " [dry-run]" if args.dry_run else "",
            )
            non_success_query = f"""
                SELECT paper_uid, download_status, canonical_source,
                       last_attempted_source, attempted_sources, sources,
                       fulltext_path
                FROM {args.table}
                WHERE download_status IN ('failed', 'exhausted')
                ORDER BY paper_uid
            """
            for row in _read_rows(conn, non_success_query):
                cleanup_non_success_row(
                    row,
                    fulltext_root,
                    args.dry_run,
                    stats,
                )
    except sqlite3.Error as exc:
        logger.exception("SQLite audit failed: %s", exc)
        return 1

    logger.info(
        "Done | success_checked=%d success_mismatch_rows=%d "
        "non_success_checked=%d unexpected_paths=%d removed=%d "
        "would_remove=%d removal_failures=%d metadata_issues=%d",
        stats.success_checked,
        stats.success_mismatch_rows,
        stats.non_success_checked,
        stats.unexpected_paths,
        stats.removed_paths,
        stats.would_remove_paths,
        stats.removal_failures,
        stats.metadata_issues,
    )

    if stats.success_issue_counts:
        logger.info(
            "Success issue counts | %s",
            " ".join(
                f"{code}={count}"
                for code, count in sorted(
                    stats.success_issue_counts.items()
                )
            ),
        )

    return 1 if (
        stats.success_mismatch_rows
        or stats.removal_failures
        or stats.metadata_issues
    ) else 0


if __name__ == "__main__":
    sys.exit(main())
