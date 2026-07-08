#!/usr/bin/env python3
"""Download arXiv search results into a local SQLite database.

This script queries the arXiv API (Atom feed endpoint), paginates through
all matching results using start/max_results offset pagination, and stores
the normalized records in a SQLite database.

Notes on the arXiv API
-----------------------
- The API is documented at https://info.arxiv.org/help/api/user-manual.html
- The Terms of Use (https://info.arxiv.org/help/api/tou.html) ask clients to
  wait a few seconds between requests and to avoid large bursts, hence the
  conservative REQUEST_DELAY_SECONDS and DEFAULT_PAGE_SIZE below.
- Offset-based pagination (the `start` parameter) can be mildly unstable at
  very large offsets (results can occasionally shift between pages as the
  index changes). Sorting by submittedDate makes this more deterministic,
  but for very large result sets it's worth spot-checking the final count
  against the `total_results` reported on the first page.
- Author affiliations are almost never populated by arXiv submitters, so
  the `affiliation` column will be NULL for the large majority of records.
  This is a gap in arXiv's own metadata, not a parsing issue.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_PAGE_SIZE = 200
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 3.1  # arXiv's ToU ask for >= 3s between requests
USER_AGENT = "get_arxiv/1.0 (mailto:kevinge@chalmers.se)"

RETRY_TOTAL = 5
RETRY_BACKOFF_FACTOR = 1.0
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS, "opensearch": OPENSEARCH_NS}

ARXIV_ID_VERSION_RE = re.compile(r"^(?P<base_id>.+)v(?P<version>\d+)$")

logger = logging.getLogger("get_arxiv")


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

    Creates a single table named ``papers`` with ``arxiv_id`` (the
    version-independent id, e.g. "2101.12345") as primary key.

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
            arxiv_id TEXT PRIMARY KEY,
            version INTEGER,
            arxiv_url TEXT,
            doi TEXT,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            affiliation TEXT,
            primary_category TEXT,
            categories TEXT,
            published TEXT,
            updated TEXT,
            journal_ref TEXT,
            comment TEXT,
            pdf_url TEXT
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
# arXiv API
# --------------------------------------------------------------------------

def fetch_page(
    session: requests.Session,
    query: str,
    start: int,
    max_results: int,
) -> str:
    """Fetch a single page of results from the arXiv API.

    Parameters
    ----------
    session : requests.Session
        HTTP session to use for the request.
    query : str
        arXiv ``search_query`` string.
    start : int
        Zero-based offset of the first result to return.
    max_results : int
        Number of results to request.

    Returns
    -------
    str
        Raw Atom XML response body.

    Raises
    ------
    requests.HTTPError
        If the HTTP request returns an error status after retries.
    """
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    response = session.get(
        ARXIV_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.text


# --------------------------------------------------------------------------
# Record parsing
# --------------------------------------------------------------------------

def _clean_text(value: Optional[str]) -> Optional[str]:
    """Collapse whitespace/newlines in API text fields and strip.

    Parameters
    ----------
    value : Optional[str]
        Raw text value from the feed.

    Returns
    -------
    Optional[str]
        Cleaned text, or None if the input was None/empty.
    """
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _split_arxiv_id(raw_id_url: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Split an arXiv Atom ``id`` URL into a version-independent id and version.

    Example: "http://arxiv.org/abs/2101.12345v3" -> ("2101.12345", 3)

    Parameters
    ----------
    raw_id_url : Optional[str]
        The raw ``<id>`` element text from an entry.

    Returns
    -------
    tuple[Optional[str], Optional[int]]
        (base_id, version). Both are None if raw_id_url is empty.
    """
    if not raw_id_url:
        return None, None
    last_segment = raw_id_url.rstrip("/").rsplit("/", 1)[-1]
    match = ARXIV_ID_VERSION_RE.match(last_segment)
    if match:
        return match.group("base_id"), int(match.group("version"))
    return last_segment, None


def parse_entry(entry: ET.Element) -> dict[str, Any]:
    """Convert one arXiv Atom ``<entry>`` element into a row matching the schema.

    Missing fields are handled gracefully and mapped to ``None``.

    Parameters
    ----------
    entry : ET.Element
        A single ``<entry>`` element from the arXiv Atom response.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys matching the ``papers`` table columns.
    """
    raw_id = entry.findtext("atom:id", default=None, namespaces=NS)
    arxiv_id, version = _split_arxiv_id(raw_id)

    primary_category_el = entry.find("arxiv:primary_category", NS)
    primary_category = (
        primary_category_el.get("term") if primary_category_el is not None else None
    )

    categories = [
        cat.get("term")
        for cat in entry.findall("atom:category", NS)
        if cat.get("term")
    ]

    author_names: list[str] = []
    affiliations: list[str] = []
    for author_el in entry.findall("atom:author", NS):
        name = author_el.findtext("atom:name", default=None, namespaces=NS)
        if name:
            author_names.append(name.strip())
        affiliation = author_el.findtext(
            "arxiv:affiliation", default=None, namespaces=NS
        )
        if affiliation:
            affiliations.append(affiliation.strip())

    pdf_url = None
    for link_el in entry.findall("atom:link", NS):
        if link_el.get("title") == "pdf" or link_el.get("type") == "application/pdf":
            pdf_url = link_el.get("href")
            break

    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "arxiv_url": raw_id,
        "doi": entry.findtext("arxiv:doi", default=None, namespaces=NS),
        "title": _clean_text(entry.findtext("atom:title", default=None, namespaces=NS)),
        "abstract": _clean_text(
            entry.findtext("atom:summary", default=None, namespaces=NS)
        ),
        "authors": "; ".join(author_names) if author_names else None,
        "affiliation": "; ".join(affiliations) if affiliations else None,
        "primary_category": primary_category,
        "categories": ", ".join(categories) if categories else None,
        "published": entry.findtext("atom:published", default=None, namespaces=NS),
        "updated": entry.findtext("atom:updated", default=None, namespaces=NS),
        "journal_ref": entry.findtext(
            "arxiv:journal_ref", default=None, namespaces=NS
        ),
        "comment": entry.findtext("arxiv:comment", default=None, namespaces=NS),
        "pdf_url": pdf_url,
    }


def parse_feed(xml_text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse an arXiv Atom feed response into records plus pagination metadata.

    Parameters
    ----------
    xml_text : str
        Raw Atom XML response body.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, int]]
        Parsed records, and a metadata dict with ``total_results``,
        ``start_index``, and ``items_per_page``.

    Raises
    ------
    RuntimeError
        If the API returned an in-band error entry (e.g. malformed query).
    """
    root = ET.fromstring(xml_text)

    total_results = int(
        root.findtext("opensearch:totalResults", default="0", namespaces=NS)
    )
    start_index = int(
        root.findtext("opensearch:startIndex", default="0", namespaces=NS)
    )
    items_per_page = int(
        root.findtext("opensearch:itemsPerPage", default="0", namespaces=NS)
    )

    entries = root.findall("atom:entry", NS)

    # arXiv reports malformed-query and other API errors as a single entry
    # whose id starts with "http://arxiv.org/api/errors" rather than as an
    # HTTP error status, so this must be checked explicitly.
    if len(entries) == 1:
        entry_id = entries[0].findtext("atom:id", default="", namespaces=NS)
        if entry_id.startswith("http://arxiv.org/api/errors"):
            error_summary = entries[0].findtext(
                "atom:summary", default="Unknown error", namespaces=NS
            )
            raise RuntimeError(f"arXiv API returned an error: {error_summary.strip()}")

    records = [parse_entry(entry) for entry in entries]
    meta = {
        "total_results": total_results,
        "start_index": start_index,
        "items_per_page": items_per_page,
    }
    return records, meta


# --------------------------------------------------------------------------
# Database inserts
# --------------------------------------------------------------------------

def insert_records(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> None:
    """Bulk insert parsed records into the papers table, ignoring duplicates.

    Commits once after the insert. Records without an ``arxiv_id`` are
    skipped since ``arxiv_id`` is the primary key.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection.
    records : list[dict[str, Any]]
        Parsed records to insert.
    """
    columns = (
        "arxiv_id",
        "version",
        "arxiv_url",
        "doi",
        "title",
        "abstract",
        "authors",
        "affiliation",
        "primary_category",
        "categories",
        "published",
        "updated",
        "journal_ref",
        "comment",
        "pdf_url",
    )
    placeholders = ", ".join(f":{col}" for col in columns)
    sql = f"INSERT OR IGNORE INTO papers ({', '.join(columns)}) VALUES ({placeholders})"

    valid_records = [record for record in records if record.get("arxiv_id")]
    skipped = len(records) - len(valid_records)
    if skipped:
        logger.warning("Skipping %d record(s) with no arxiv_id", skipped)

    if not valid_records:
        return

    conn.executemany(sql, valid_records)
    conn.commit()


# --------------------------------------------------------------------------
# Main download workflow
# --------------------------------------------------------------------------

def download_query(db_path: Path) -> None:
    """Download all arXiv search results for the configured query into SQLite.

    Paginates through the full result set using start/max_results offset
    pagination and inserts every page into the database as it is
    downloaded. If the database already contains records, downloading
    simply continues to insert with ``INSERT OR IGNORE`` (no resume from a
    prior run's offset is attempted, matching the source script's design).
    """
    query = (
        "(ti:polymer OR abs:polymer OR "
        "ti:polymers OR abs:polymers OR "
        "ti:copolymer OR abs:copolymer OR "
        "ti:copolymers OR abs:copolymers OR "
        "ti:polymerization OR abs:polymerization OR "
        "ti:copolymerization OR abs:copolymerization)"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    page_size = DEFAULT_PAGE_SIZE

    session = create_session()
    conn = create_database(db_path)

    try:
        existing_count = count_existing_records(conn)
        logger.info("Database currently has %d record(s)", existing_count)

        start = 0
        page_number = 0
        cumulative_downloaded = 0
        total_results: Optional[int] = None

        while True:
            page_number += 1
            logger.info("Page %d (start=%d)", page_number, start)

            xml_text = fetch_page(session, query, start, page_size)
            records, meta = parse_feed(xml_text)

            if total_results is None:
                total_results = meta["total_results"]
                logger.info("Total hit count: %d", total_results)

            if not records:
                logger.info("No more results; stopping.")
                break

            insert_records(conn, records)

            cumulative_downloaded += len(records)
            logger.info("Downloaded %d records", len(records))
            logger.info(
                "Total downloaded %d / %d", cumulative_downloaded, total_results
            )

            start += len(records)

            if total_results and start >= total_results:
                logger.info("Reached end of result set; stopping.")
                break

            time.sleep(REQUEST_DELAY_SECONDS)

        final_count = count_existing_records(conn)
        logger.info("Done. Database now has %d record(s)", final_count)
        if total_results is not None and cumulative_downloaded < total_results:
            logger.warning(
                "Downloaded %d records but arXiv reported %d total; "
                "consider re-running to fill gaps (arXiv's offset "
                "pagination can occasionally skip results at large offsets).",
                cumulative_downloaded,
                total_results,
            )

    finally:
        conn.close()


def main() -> None:
    """Entry point: configure logging and run the download."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    start_time = time.monotonic()
    try:
        db_path = Path("data/arxiv.db")
        download_query(db_path)
    except requests.HTTPError as exc:
        logger.error("HTTP error while querying arXiv: %s", exc)
        raise SystemExit(1) from exc
    except requests.RequestException as exc:
        logger.error("Network error while querying arXiv: %s", exc)
        raise SystemExit(1) from exc
    except sqlite3.Error as exc:
        logger.error("SQLite error: %s", exc)
        raise SystemExit(1) from exc
    except ET.ParseError as exc:
        logger.error("Failed to parse arXiv response XML: %s", exc)
        raise SystemExit(1) from exc
    finally:
        elapsed = time.monotonic() - start_time
        logger.info("Elapsed time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()