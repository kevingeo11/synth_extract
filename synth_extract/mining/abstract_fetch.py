from __future__ import annotations

import logging
import sqlite3
import time
from typing import Optional

from pybliometrics.scopus import ScopusSearch

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REQUIRED_COLUMNS = {"eid", "abstract", "abstract_source"}
MAX_BATCH_SIZE = 25  # practical cap for an OR-chained EID() query
NO_ABSTRACT_SOURCE = "scopus_no_abstract"   # Scopus had the doc, but no abstract text
NOT_FOUND_SOURCE = "scopus_not_found"       # Scopus returned nothing for this eid
ERROR_SOURCE = "scopus_error"               # the API call itself failed
SUCCESS_SOURCE = "scopus_search"            # abstract successfully retrieved
SKIP_SOURCES = (NO_ABSTRACT_SOURCE, NOT_FOUND_SOURCE, ERROR_SOURCE)


# --------------------------------------------------------------------------- #
# Safety-check helpers
# --------------------------------------------------------------------------- #

def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone() is not None


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    cur = conn.execute(f'PRAGMA table_info("{table_name}")')
    return {row[1] for row in cur.fetchall()}


def _validate(conn: sqlite3.Connection, table_name: str) -> None:
    if not _table_exists(conn, table_name):
        raise ValueError(f"Table '{table_name}' does not exist in this database.")

    columns = _get_table_columns(conn, table_name)
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(
            f"Table '{table_name}' is missing required column(s): {sorted(missing)}"
        )


# --------------------------------------------------------------------------- #
# DB access helpers
# --------------------------------------------------------------------------- #

def _fetch_missing_batch(
    conn: sqlite3.Connection, table_name: str, batch_size: int
) -> list[str]:
    """Return up to `batch_size` distinct, non-empty eids that still need an abstract."""
    query = f"""
        SELECT eid
        FROM "{table_name}"
        WHERE abstract IS NULL
          AND eid IS NOT NULL
          AND TRIM(eid) != ''
          AND (abstract_source IS NULL OR abstract_source NOT IN (?, ?, ?))
        LIMIT ?
    """
    cur = conn.execute(query, (*SKIP_SOURCES, batch_size))
    return [row[0].strip() for row in cur.fetchall()]


def _update_row(
    conn: sqlite3.Connection,
    table_name: str,
    eid: str,
    abstract: Optional[str],
    source: str,
) -> None:
    conn.execute(
        f'UPDATE "{table_name}" SET abstract = ?, abstract_source = ? WHERE eid = ?',
        (abstract, source, eid),
    )


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def fill_missing_abstracts(
    db_path: str,
    table_name: str,
    batch_size: int = 25,
    quota_threshold: int = 1000,
    delay_seconds: float = 2.0,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Backfill missing abstracts in `table_name` using Scopus Search, in place.

    Parameters
    ----------
    db_path : path to the SQLite database file.
    table_name : name of the table to update.
    batch_size : rows fetched from Scopus per query (1-25).
    quota_threshold : stop once the API key's remaining quota drops below this.
    delay_seconds : pause between successive Scopus queries (politeness / rate limiting).
    max_batches : optional hard cap on number of batches processed this run.

    Returns
    -------
    dict summary of counts: batches, updated, no_abstract, not_found, errors.
    """
    if not (1 <= batch_size <= MAX_BATCH_SIZE):
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE} (got {batch_size})")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be >= 0")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # safer/faster for incremental commits

    stats = {"batches": 0, "updated": 0, "no_abstract": 0, "not_found": 0, "errors": 0}

    try:
        _validate(conn, table_name)
        logger.info(
            "Starting abstract backfill | db=%s table=%s batch_size=%d quota_threshold=%d",
            db_path, table_name, batch_size, quota_threshold,
        )

        while True:
            if max_batches is not None and stats["batches"] >= max_batches:
                logger.info("Reached max_batches=%d limit. Stopping.", max_batches)
                break

            eids = _fetch_missing_batch(conn, table_name, batch_size)
            if not eids:
                logger.info("No rows left with missing abstracts. Done.")
                break

            stats["batches"] += 1
            batch_num = stats["batches"]
            logger.info("Batch %d: querying Scopus for %d EID(s)", batch_num, len(eids))

            query_str = " OR ".join(f"EID({eid})" for eid in eids)

            try:
                s = ScopusSearch(query_str, view="COMPLETE")
            except Exception as exc:
                logger.exception("Batch %d: ScopusSearch request failed: %s", batch_num, exc)
                for eid in eids:
                    _update_row(conn, table_name, eid, None, ERROR_SOURCE)
                conn.commit()
                stats["errors"] += len(eids)
                time.sleep(delay_seconds)
                continue

            results = s.results or []
            returned_eids: set[str] = set()
            batch_updated = 0
            batch_no_abstract = 0

            for doc in results:
                doc_eid = (doc.eid or "").strip()
                if not doc_eid:
                    continue
                returned_eids.add(doc_eid)

                abstract_text = (doc.description or "").strip() or None
                if abstract_text:
                    _update_row(conn, table_name, doc_eid, abstract_text, SUCCESS_SOURCE)
                    stats["updated"] += 1
                    batch_updated += 1
                else:
                    _update_row(conn, table_name, doc_eid, None, NO_ABSTRACT_SOURCE)
                    stats["no_abstract"] += 1
                    batch_no_abstract += 1

            not_found = set(eids) - returned_eids
            for eid in not_found:
                _update_row(conn, table_name, eid, None, NOT_FOUND_SOURCE)
            stats["not_found"] += len(not_found)

            conn.commit()
            logger.info(
                "Batch %d done: %d updated, %d no-abstract, %d not-found",
                batch_num,
                batch_updated,
                batch_no_abstract,
                len(not_found),
            )

            # Check remaining quota (only known after at least one request)
            remaining_quota = None
            try:
                remaining_quota = int(s.get_key_remaining_quota())
                logger.info("Remaining Scopus quota: %d", remaining_quota)
            except Exception as exc:
                logger.warning("Could not read remaining quota: %s", exc)

            if remaining_quota is not None and remaining_quota < quota_threshold:
                logger.warning(
                    "Remaining quota (%d) is below threshold (%d). Stopping.",
                    remaining_quota, quota_threshold,
                )
                break

            time.sleep(delay_seconds)

        logger.info(
            "Finished | batches=%d updated=%d no_abstract=%d not_found=%d errors=%d",
            stats["batches"], stats["updated"], stats["no_abstract"],
            stats["not_found"], stats["errors"],
        )
        return stats

    finally:
        conn.close()
