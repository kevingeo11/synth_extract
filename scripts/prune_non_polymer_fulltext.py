#!/usr/bin/env python3
"""Remove full-text UID folders absent from ``polymer_material_papers``.

The expected filesystem layout is::

    FULLTEXT_ROOT / SOURCE / PAPER_UID / files

The SQLite database is opened read-only. Immediate child directories beneath
each source directory are compared with ``polymer_material_papers.paper_uid``.
Directories whose names are absent from the table are removal candidates.

The default mode is a dry run. Pass ``--delete`` to permanently remove the
candidate UID directories.

Examples
--------
Preview all removals using the Arrhenius defaults::

    python scripts/prune_non_polymer_fulltext.py

Permanently remove the reported directories::

    python scripts/prune_non_polymer_fulltext.py --delete

Process only selected sources::

    python scripts/prune_non_polymer_fulltext.py --source wiley --source arxiv
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(
    "/nobackup/proj/disk/naiss2024-5-630/personal/george/synth_extract"
)
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "central_papers.db"
DEFAULT_FULLTEXT_ROOT = PROJECT_ROOT / "data" / "fulltext"
TABLE = "polymer_material_papers"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")

LOG = logging.getLogger("prune_non_polymer_fulltext")


@dataclass
class Stats:
    source_directories: int = 0
    uid_directories: int = 0
    retained: int = 0
    candidates: int = 0
    removed: int = 0
    removal_errors: int = 0
    skipped_symlinks: int = 0
    skipped_unsafe_names: int = 0
    candidates_by_source: Counter[str] = field(default_factory=Counter)
    removed_by_source: Counter[str] = field(default_factory=Counter)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove source/UID directories whose UID is absent from "
            f"{TABLE}. Dry-run is the default."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite database path (default: %(default)s)",
    )
    parser.add_argument(
        "--fulltext-root",
        type=Path,
        default=DEFAULT_FULLTEXT_ROOT,
        help="Directory containing source/UID folders (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Only process this source; may be supplied multiple times",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete candidates; without this flag, only report",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: %(default)s)",
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
        level=getattr(logging, level),
        handlers=[handler],
        force=True,
    )


def safe_component(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and SAFE_COMPONENT.fullmatch(value) is not None
    )


def validate_fulltext_root(path: Path) -> Path:
    """Resolve and validate the destructive-operation boundary."""
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Full-text root is not a directory: {root}")
    if root.name != "fulltext" or root.parent.name != "data":
        raise ValueError(
            "Refusing a nonstandard deletion root. Expected a directory "
            f"ending in data/fulltext, received: {root}"
        )
    forbidden = {Path("/").resolve(), Path.home().resolve()}
    if root in forbidden:
        raise ValueError(f"Refusing unsafe full-text root: {root}")
    return root


def load_retained_uids(db_path: Path) -> set[str]:
    """Read and validate all UIDs that must remain on disk."""
    db_path = db_path.expanduser().resolve(strict=True)
    if not db_path.is_file():
        raise ValueError(f"Database is not a file: {db_path}")

    uri = f"{db_path.as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=60)) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 60000")

        columns = connection.execute(
            f'PRAGMA table_info("{TABLE}")'
        ).fetchall()
        if not columns:
            raise ValueError(f"Database has no {TABLE!r} table")
        column_names = {str(row[1]) for row in columns}
        if "paper_uid" not in column_names:
            raise ValueError(f"{TABLE!r} has no 'paper_uid' column")

        row_count, distinct_count, invalid_count = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT paper_uid),
                SUM(paper_uid IS NULL OR TRIM(paper_uid) = '')
            FROM "{TABLE}"
            """
        ).fetchone()
        row_count = int(row_count or 0)
        distinct_count = int(distinct_count or 0)
        invalid_count = int(invalid_count or 0)

        if row_count == 0:
            raise ValueError(
                f"Refusing cleanup because {TABLE!r} contains no rows"
            )
        if invalid_count:
            raise ValueError(
                f"{TABLE!r} contains {invalid_count:,} null or empty UID(s)"
            )
        if row_count != distinct_count:
            raise ValueError(
                f"{TABLE!r} contains {row_count - distinct_count:,} "
                "duplicate UID row(s)"
            )

        retained_uids = {
            str(row[0])
            for row in connection.execute(
                f'SELECT paper_uid FROM "{TABLE}"'
            )
        }

    if len(retained_uids) != row_count:
        raise RuntimeError("UID-set size changed unexpectedly while loading")
    return retained_uids


def source_directories(
    fulltext_root: Path,
    requested_sources: list[str] | None,
) -> list[Path]:
    """Resolve source directories without following symlinks."""
    if requested_sources:
        duplicates = sorted(
            source
            for source, count in Counter(requested_sources).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate --source value(s): {duplicates}")

        directories = []
        for source in requested_sources:
            if not safe_component(source):
                raise ValueError(f"Unsafe source name: {source!r}")
            source_path = fulltext_root / source
            if source_path.is_symlink():
                raise ValueError(f"Refusing symlinked source: {source_path}")
            if not source_path.is_dir():
                raise ValueError(f"Source directory not found: {source_path}")
            directories.append(source_path)
        return sorted(directories, key=lambda path: path.name)

    directories = []
    with os.scandir(fulltext_root) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                LOG.warning("Skipping symlinked source entry: %s", entry.path)
                continue
            if entry.is_dir(follow_symlinks=False):
                if not safe_component(entry.name):
                    LOG.warning("Skipping unsafe source name: %r", entry.name)
                    continue
                directories.append(Path(entry.path))
    return sorted(directories, key=lambda path: path.name)


def safely_remove_uid_directory(uid_path: Path, source_path: Path) -> None:
    """Revalidate the immediate-child relationship and remove one directory."""
    if uid_path.is_symlink():
        raise ValueError(f"Refusing symlinked UID path: {uid_path}")
    if uid_path.parent.resolve(strict=True) != source_path.resolve(strict=True):
        raise ValueError(f"UID path is not an immediate source child: {uid_path}")
    if not uid_path.is_dir():
        raise ValueError(f"UID path is no longer a directory: {uid_path}")
    shutil.rmtree(uid_path)


def prune_source(
    source_path: Path,
    retained_uids: set[str],
    delete: bool,
    stats: Stats,
) -> None:
    source = source_path.name
    stats.source_directories += 1
    LOG.info("Scanning source=%s path=%s", source, source_path)

    with os.scandir(source_path) as entries:
        for entry in entries:
            if entry.is_symlink():
                stats.skipped_symlinks += 1
                LOG.error("Skipping symlinked source child: %s", entry.path)
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue

            paper_uid = entry.name
            uid_path = Path(entry.path)
            stats.uid_directories += 1
            if not safe_component(paper_uid):
                stats.skipped_unsafe_names += 1
                LOG.error("Skipping unsafe UID directory name: %s", uid_path)
                continue
            if paper_uid in retained_uids:
                stats.retained += 1
                continue

            stats.candidates += 1
            stats.candidates_by_source[source] += 1
            if not delete:
                LOG.info("DRY RUN | would remove %s", uid_path)
                continue

            LOG.warning("Removing %s", uid_path)
            try:
                safely_remove_uid_directory(uid_path, source_path)
            except (OSError, ValueError) as exc:
                stats.removal_errors += 1
                LOG.error("Removal failed path=%s error=%s", uid_path, exc)
            else:
                stats.removed += 1
                stats.removed_by_source[source] += 1


def log_summary(stats: Stats, delete: bool) -> None:
    mode = "DELETE" if delete else "DRY RUN"
    LOG.info(
        "Summary mode=%s sources=%d uid_directories=%d retained=%d "
        "candidates=%d removed=%d removal_errors=%d skipped_symlinks=%d "
        "skipped_unsafe_names=%d",
        mode,
        stats.source_directories,
        stats.uid_directories,
        stats.retained,
        stats.candidates,
        stats.removed,
        stats.removal_errors,
        stats.skipped_symlinks,
        stats.skipped_unsafe_names,
    )
    for source in sorted(stats.candidates_by_source):
        LOG.info(
            "Source summary source=%s candidates=%d removed=%d",
            source,
            stats.candidates_by_source[source],
            stats.removed_by_source[source],
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        fulltext_root = validate_fulltext_root(args.fulltext_root)
        retained_uids = load_retained_uids(args.db_path)
        sources = source_directories(fulltext_root, args.source)
    except (OSError, sqlite3.Error, ValueError, RuntimeError):
        LOG.exception("Preflight validation failed; nothing was removed")
        return 2

    if not sources:
        LOG.error("No source directories found under %s", fulltext_root)
        return 2

    LOG.info(
        "Starting mode=%s database=%s table=%s retained_uids=%d "
        "fulltext_root=%s sources=%d",
        "DELETE" if args.delete else "DRY RUN",
        args.db_path.expanduser().resolve(),
        TABLE,
        len(retained_uids),
        fulltext_root,
        len(sources),
    )
    if args.delete:
        LOG.warning(
            "Permanent deletion enabled; directories absent from %s will be "
            "removed",
            TABLE,
        )

    stats = Stats()
    try:
        for source_path in sources:
            prune_source(source_path, retained_uids, args.delete, stats)
    except KeyboardInterrupt:
        LOG.warning("Interrupted; directories removed earlier are not recoverable")
        log_summary(stats, args.delete)
        return 130
    except OSError:
        LOG.exception("Filesystem scan failed")
        log_summary(stats, args.delete)
        return 1

    log_summary(stats, args.delete)
    if stats.removal_errors or stats.skipped_symlinks or stats.skipped_unsafe_names:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
