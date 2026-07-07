#!/usr/bin/env python3
"""Download Europe PMC search results into a local SQLite database.

This script queries the Europe PMC REST API (search endpoint), paginates
through all matching results using cursor-based pagination, and stores the
normalized records in a SQLite database.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
DEFAULT_PAGE_SIZE = 1000
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 2.0
USER_AGENT = "get_europepmc/1.0 (+https://europepmc.org)"

RETRY_TOTAL = 5
RETRY_BACKOFF_FACTOR = 1.0
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

logger = logging.getLogger("get_europepmc")


# --------------------------------------------------------------------------
# HTTP session
# --------------------------------------------------------------------------

def create_session() -> requests.Session:
    """Create a requests.Session configured with retries and connection pooling.

    Retries are applied on common transient HTTP errors (429, 500, 502, 503,
    504) using exponential backoff.

    Returns
    -------
    requests.Session
        A configured session ready to use for GET requests.
    """
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def create_database(db_path: Path) -> sqlite3.Connection:
    """Create (if needed) and connect to the SQLite database.

    Creates a single table named ``papers`` with ``pmcid`` as primary key.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        An open connection to the database.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            doi TEXT,
            pmcid TEXT PRIMARY KEY,
            pmid TEXT,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            affiliation TEXT,
            journal TEXT,
            publication_year INTEGER,
            is_open_access INTEGER,
            in_epmc INTEGER,
            in_pmc INTEGER,
            has_pdf INTEGER
        )
        """
    )
    conn.commit()
    return conn


def count_existing_records(conn: sqlite3.Connection) -> int:
    """Return the number of rows currently stored in the papers table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection.

    Returns
    -------
    int
        Number of existing rows.
    """
    cursor = conn.execute("SELECT COUNT(*) FROM papers")
    (count,) = cursor.fetchone()
    return int(count)


# --------------------------------------------------------------------------
# Europe PMC API
# --------------------------------------------------------------------------

def fetch_page(
    session: requests.Session,
    query: str,
    page_size: int,
    cursor_mark: str,
) -> dict[str, Any]:
    """Fetch a single page of results from the Europe PMC search API.

    Parameters
    ----------
    session : requests.Session
        HTTP session to use for the request.
    query : str
        Europe PMC search query string.
    page_size : int
        Number of results to request per page.
    cursor_mark : str
        Cursor mark for pagination; use ``"*"`` for the first page.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response body.

    Raises
    ------
    requests.HTTPError
        If the HTTP request returns an error status after retries.
    """
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": page_size,
        "cursorMark": cursor_mark,
    }
    response = session.get(
        EUROPEPMC_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------
# Record parsing
# --------------------------------------------------------------------------

def _to_bool_int(value: Optional[str]) -> Optional[int]:
    """Convert an EuropePMC 'Y'/'N' flag to 1/0, preserving None if absent.

    Parameters
    ----------
    value : Optional[str]
        Raw flag value from the API (e.g. "Y", "N", or missing).

    Returns
    -------
    Optional[int]
        1 if "Y", 0 if present but not "Y", None if the value was absent.
    """
    if value is None:
        return None
    return 1 if value == "Y" else 0


def _to_int(value: Any) -> Optional[int]:
    """Safely convert a value to int, returning None on failure.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Optional[int]
        Converted integer, or None if conversion is not possible.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_record(item: dict[str, Any]) -> dict[str, Any]:
    """Convert one Europe PMC result record into a row matching the schema.

    Missing fields are handled gracefully and mapped to ``None``.

    Parameters
    ----------
    item : dict[str, Any]
        A single result record from the Europe PMC API response.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys matching the ``papers`` table columns.
    """
    journal_info = item.get("journalInfo") or {}
    journal = journal_info.get("journal") or {}

    return {
        "doi": item.get("doi"),
        "pmcid": item.get("pmcid"),
        "pmid": item.get("pmid"),
        "title": item.get("title"),
        "abstract": item.get("abstractText"),
        "authors": item.get("authorString"),
        "affiliation": item.get("affiliation"),
        "journal": journal.get("title"),
        "publication_year": _to_int(item.get("pubYear")),
        "is_open_access": _to_bool_int(item.get("isOpenAccess")),
        "in_epmc": _to_bool_int(item.get("inEPMC")),
        "in_pmc": _to_bool_int(item.get("inPMC")),
        "has_pdf": _to_bool_int(item.get("hasPDF")),
    }


# --------------------------------------------------------------------------
# Database inserts
# --------------------------------------------------------------------------

def insert_records(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> None:
    """Bulk insert parsed records into the papers table, ignoring duplicates.

    Commits once after the insert. Records without a ``pmcid`` are skipped
    since ``pmcid`` is the primary key.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection.
    records : list[dict[str, Any]]
        Parsed records to insert.
    """
    columns = (
        "doi",
        "pmcid",
        "pmid",
        "title",
        "abstract",
        "authors",
        "affiliation",
        "journal",
        "publication_year",
        "is_open_access",
        "in_epmc",
        "in_pmc",
        "has_pdf",
    )
    placeholders = ", ".join(f":{col}" for col in columns)
    sql = f"INSERT OR IGNORE INTO papers ({', '.join(columns)}) VALUES ({placeholders})"

    valid_records = [record for record in records if record.get("pmcid")]
    skipped = len(records) - len(valid_records)
    if skipped:
        logger.warning("Skipping %d record(s) with no pmcid", skipped)

    if not valid_records:
        return

    conn.executemany(sql, valid_records)
    conn.commit()


# --------------------------------------------------------------------------
# Main download workflow
# --------------------------------------------------------------------------

def download_query(db_path: Path) -> None:
    """Download all Europe PMC search results for the configured query into SQLite.

    Paginates through the full result set using cursor-based pagination and
    inserts every page into the database as it is downloaded. If the
    database already contains records, downloading simply continues to
    insert with ``INSERT OR IGNORE`` (no cursor-based resume is attempted).
    """
    query = (
        "(TITLE_ABS:polymer* OR TITLE_ABS:copolymer*) AND HAS_ABSTRACT:y "
        "AND (IN_PMC:y OR IN_EPMC:y OR HAS_PDF:y)"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    page_size = DEFAULT_PAGE_SIZE

    session = create_session()
    conn = create_database(db_path)

    try:
        existing_count = count_existing_records(conn)
        logger.info("Database currently has %d record(s)", existing_count)

        cursor_mark = "*"
        page_number = 0
        cumulative_downloaded = 0

        while True:
            page_number += 1
            logger.info("Page %d (cursorMark=%s)", page_number, cursor_mark)

            data = fetch_page(session, query, page_size, cursor_mark)

            
            logger.info("Total hit count: %d", data.get("hitCount", 0))

            results = data.get("resultList", {}).get("result", [])
            next_cursor_mark = data.get("nextCursorMark")
            is_final_page = (
                not results
                or not next_cursor_mark
                or next_cursor_mark == cursor_mark
            )

            if results:
                records = [parse_record(item) for item in results]
                insert_records(conn, records)

                cumulative_downloaded += len(records)
                logger.info("Downloaded %d records", len(records))
                logger.info("Total downloaded %d", cumulative_downloaded)

            if is_final_page:
                logger.info("Reached final page; stopping.")
                break

            time.sleep(REQUEST_DELAY_SECONDS)
            cursor_mark = next_cursor_mark

        final_count = count_existing_records(conn)
        logger.info("Done. Database now has %d record(s)", final_count)

    finally:
        conn.close()


def main() -> None:
    """Entry point: configure logging and run the download."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    start_time = time.monotonic()
    try:
        db_path = Path("data/europepmc.db")
        download_query(db_path)
    except requests.HTTPError as exc:
        logger.error("HTTP error while querying Europe PMC: %s", exc)
        raise SystemExit(1) from exc
    except requests.RequestException as exc:
        logger.error("Network error while querying Europe PMC: %s", exc)
        raise SystemExit(1) from exc
    except sqlite3.Error as exc:
        logger.error("SQLite error: %s", exc)
        raise SystemExit(1) from exc
    finally:
        elapsed = time.monotonic() - start_time
        logger.info("Elapsed time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
