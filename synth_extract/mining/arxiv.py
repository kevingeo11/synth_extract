#!/usr/bin/env python3
"""Download arXiv search results into a local SQLite database.

This script queries the arXiv API (Atom feed endpoint) and stores the
normalized records in a SQLite database.

Why date-bucketed pagination
-----------------------------
arXiv's `start`/`max_results` offset pagination becomes unreliable (500
errors, or silently missing results) once `start` gets into five figures.
This is a long-standing, documented behavior of the underlying search
backend, not a bug in this script -- see e.g.
https://info.arxiv.org/help/api/user-manual.html and the arXiv API mailing
list (groups.google.com/g/arxiv-api), where arXiv's own maintainers'
recommended fix for large result sets is the same one used here: split the
query into smaller pieces using `submittedDate` ranges, and page within
each piece, rather than paging deep into one giant result set.

Concretely: the whole date range (1991-present) is recursively bisected by
submittedDate until every "bucket" has at most MAX_BUCKET_RESULTS hits
(comfortably under the ~10,000 offset wall), and only then does the script
page through each bucket with plain start=0.. pagination.

Resume support
---------------
Because arXiv's API has also been reported (as recently as early 2026) to
intermittently rate-limit (HTTP 429) even compliant clients, this script
persists its bucket list and per-bucket completion status in the database
itself (`download_progress` table). Re-running the script after a crash
skips buckets already marked complete, so you lose at most one bucket's
worth of progress rather than the whole run.

Notes on the arXiv API
-----------------------
- The API is documented at https://info.arxiv.org/help/api/user-manual.html
- The Terms of Use (https://info.arxiv.org/help/api/tou.html) ask clients to
  wait a few seconds between requests and to avoid large bursts, hence the
  conservative REQUEST_DELAY_SECONDS and DEFAULT_PAGE_SIZE below.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_PAGE_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 3.1  # arXiv's ToU ask for >= 3s between requests
USER_AGENT = "get_arxiv/1.0 (mailto:your-email@example.com)"

RETRY_TOTAL = 8
RETRY_BACKOFF_FACTOR = 2.0
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

# Keep each date bucket comfortably under arXiv's ~10,000-result offset wall.
MAX_BUCKET_RESULTS = 8000
# arXiv's earliest submissions are from mid-1991; padding the end date by a
# day avoids edge cases from clock skew / timezone rounding.
GLOBAL_RANGE_START = datetime(1991, 1, 1, tzinfo=timezone.utc)
GLOBAL_RANGE_END_PAD = timedelta(days=1)
MIN_BUCKET_WIDTH = timedelta(minutes=1)  # recursion floor, avoids infinite splits

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS, "opensearch": OPENSEARCH_NS}

ARXIV_ID_VERSION_RE = re.compile(r"^(?P<base_id>.+)v(?P<version>\d+)$")
ARXIV_DATE_FMT = "%Y%m%d%H%M"

logger = logging.getLogger("get_arxiv")


# --------------------------------------------------------------------------
# HTTP session
# --------------------------------------------------------------------------

def create_session() -> requests.Session:
    """Create a requests.Session configured with retries and connection pooling.

    Retries are applied on common transient HTTP errors (429, 500, 502, 503,
    504) using exponential backoff. 429 responses honor the server's
    Retry-After header when present (urllib3's default behavior).

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

    Creates a ``papers`` table (``arxiv_id``, the version-independent id,
    e.g. "2101.12345", as primary key) and a ``download_progress`` table
    used to make downloads resumable across runs.

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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS download_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            range_start TEXT NOT NULL,
            range_end TEXT NOT NULL,
            expected_total INTEGER,
            downloaded INTEGER DEFAULT 0,
            completed INTEGER DEFAULT 0
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


def load_or_create_buckets(
    conn: sqlite3.Connection, session: requests.Session, base_query: str
) -> list[sqlite3.Row]:
    """Load persisted date buckets, or compute and persist them if absent.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection.
    session : requests.Session
        HTTP session used for probe requests if buckets must be computed.
    base_query : str
        The arXiv ``search_query`` string (without any date filter).

    Returns
    -------
    list[sqlite3.Row]
        Rows from ``download_progress``, ordered by ``range_start``.
    """
    conn.row_factory = sqlite3.Row
    existing = conn.execute(
        "SELECT * FROM download_progress ORDER BY range_start"
    ).fetchall()
    if existing:
        logger.info("Resuming from %d previously computed bucket(s)", len(existing))
        return existing

    logger.info("No existing buckets found; computing date buckets...")
    range_end = datetime.now(timezone.utc) + GLOBAL_RANGE_END_PAD
    buckets = compute_date_buckets(
        session, base_query, GLOBAL_RANGE_START, range_end
    )
    logger.info("Computed %d bucket(s)", len(buckets))

    for bucket_start, bucket_end, expected_total in buckets:
        conn.execute(
            "INSERT INTO download_progress "
            "(range_start, range_end, expected_total, downloaded, completed) "
            "VALUES (?, ?, ?, 0, 0)",
            (
                bucket_start.strftime(ARXIV_DATE_FMT),
                bucket_end.strftime(ARXIV_DATE_FMT),
                expected_total,
            ),
        )
    conn.commit()

    return conn.execute(
        "SELECT * FROM download_progress ORDER BY range_start"
    ).fetchall()


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


def get_total_results(session: requests.Session, query: str) -> int:
    """Fetch just the reported total result count for a query.

    Uses ``max_results=1`` to keep the probe request cheap.

    Parameters
    ----------
    session : requests.Session
        HTTP session to use for the request.
    query : str
        arXiv ``search_query`` string.

    Returns
    -------
    int
        The API's reported total number of matches.
    """
    xml_text = fetch_page(session, query, start=0, max_results=1)
    _, meta = parse_feed(xml_text)
    return meta["total_results"]


def build_dated_query(
    base_query: str, range_start: datetime, range_end: datetime
) -> str:
    """Combine a base search_query with a submittedDate range filter.

    Parameters
    ----------
    base_query : str
        The arXiv ``search_query`` string (without any date filter).
    range_start : datetime
        Inclusive lower bound (UTC).
    range_end : datetime
        Exclusive-ish upper bound (UTC); arXiv treats it as inclusive to
        the minute, see the API user manual for exact semantics.

    Returns
    -------
    str
        Combined query string suitable for the ``search_query`` parameter.
    """
    start_fmt = range_start.strftime(ARXIV_DATE_FMT)
    end_fmt = range_end.strftime(ARXIV_DATE_FMT)
    return f"{base_query} AND submittedDate:[{start_fmt} TO {end_fmt}]"


def compute_date_buckets(
    session: requests.Session,
    base_query: str,
    range_start: datetime,
    range_end: datetime,
) -> list[tuple[datetime, datetime, int]]:
    """Recursively bisect a date range into buckets of bounded result count.

    Each returned bucket is guaranteed to contain at most
    ``MAX_BUCKET_RESULTS`` results (or to have hit ``MIN_BUCKET_WIDTH``,
    which acts as a safety-net recursion floor). Buckets with zero results
    are dropped.

    Parameters
    ----------
    session : requests.Session
        HTTP session used for probe requests.
    base_query : str
        The arXiv ``search_query`` string (without any date filter).
    range_start : datetime
        Inclusive lower bound (UTC) of the range to bisect.
    range_end : datetime
        Upper bound (UTC) of the range to bisect.

    Returns
    -------
    list[tuple[datetime, datetime, int]]
        List of (bucket_start, bucket_end, expected_total) tuples, ordered
        chronologically.
    """
    query = build_dated_query(base_query, range_start, range_end)
    total = get_total_results(session, query)
    time.sleep(REQUEST_DELAY_SECONDS)

    logger.info(
        "Probed [%s to %s): %d result(s)",
        range_start.isoformat(),
        range_end.isoformat(),
        total,
    )

    if total == 0:
        return []

    width = range_end - range_start
    if total <= MAX_BUCKET_RESULTS or width <= MIN_BUCKET_WIDTH:
        return [(range_start, range_end, total)]

    midpoint = range_start + width / 2
    left = compute_date_buckets(session, base_query, range_start, midpoint)
    right = compute_date_buckets(session, base_query, midpoint, range_end)
    return left + right


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


def download_bucket(
    conn: sqlite3.Connection,
    session: requests.Session,
    base_query: str,
    bucket_row: sqlite3.Row,
    page_size: int,
) -> None:
    """Download every result within a single date bucket, then mark it complete.

    Pages with plain start=0.. offsets, which is safe here because each
    bucket is bounded to at most MAX_BUCKET_RESULTS results.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection.
    session : requests.Session
        HTTP session to use for requests.
    base_query : str
        The arXiv ``search_query`` string (without any date filter).
    bucket_row : sqlite3.Row
        Row from ``download_progress`` describing this bucket.
    page_size : int
        Number of results to request per page.
    """
    range_start = datetime.strptime(bucket_row["range_start"], ARXIV_DATE_FMT).replace(
        tzinfo=timezone.utc
    )
    range_end = datetime.strptime(bucket_row["range_end"], ARXIV_DATE_FMT).replace(
        tzinfo=timezone.utc
    )
    query = build_dated_query(base_query, range_start, range_end)

    start = 0
    downloaded = 0
    while True:
        xml_text = fetch_page(session, query, start, page_size)
        records, meta = parse_feed(xml_text)

        if not records:
            break

        insert_records(conn, records)
        downloaded += len(records)
        start += len(records)

        logger.info(
            "  bucket [%s to %s]: %d/%d downloaded",
            bucket_row["range_start"],
            bucket_row["range_end"],
            downloaded,
            meta["total_results"] or bucket_row["expected_total"],
        )

        conn.execute(
            "UPDATE download_progress SET downloaded = ? WHERE id = ?",
            (downloaded, bucket_row["id"]),
        )
        conn.commit()

        if meta["total_results"] and start >= meta["total_results"]:
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    conn.execute(
        "UPDATE download_progress SET completed = 1 WHERE id = ?",
        (bucket_row["id"],),
    )
    conn.commit()


# --------------------------------------------------------------------------
# Main download workflow
# --------------------------------------------------------------------------

def download_query(db_path: Path) -> None:
    """Download all arXiv search results for the configured query into SQLite.

    Splits the query into date-bounded buckets (see module docstring),
    then pages through each bucket in turn. Buckets already marked
    complete in ``download_progress`` from a prior run are skipped, making
    the whole process resumable after a crash or rate-limit error.
    """
    base_query = (
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

        buckets = load_or_create_buckets(conn, session, base_query)
        total_expected = sum(b["expected_total"] for b in buckets)
        remaining = [b for b in buckets if not b["completed"]]
        logger.info(
            "%d bucket(s) total (%d already complete), ~%d records expected",
            len(buckets),
            len(buckets) - len(remaining),
            total_expected,
        )

        for i, bucket_row in enumerate(remaining, start=1):
            logger.info(
                "Bucket %d/%d: [%s to %s], ~%d expected",
                i,
                len(remaining),
                bucket_row["range_start"],
                bucket_row["range_end"],
                bucket_row["expected_total"],
            )
            download_bucket(conn, session, base_query, bucket_row, page_size)
            time.sleep(REQUEST_DELAY_SECONDS)

        final_count = count_existing_records(conn)
        logger.info("Done. Database now has %d record(s)", final_count)

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
        logger.error("Re-run the script to resume from the last completed bucket.")
        raise SystemExit(1) from exc
    except requests.RequestException as exc:
        logger.error("Network error while querying arXiv: %s", exc)
        logger.error("Re-run the script to resume from the last completed bucket.")
        raise SystemExit(1) from exc
    except sqlite3.Error as exc:
        logger.error("SQLite error: %s", exc)
        raise SystemExit(1) from exc
    except ET.ParseError as exc:
        logger.error("Failed to parse arXiv response XML: %s", exc)
        logger.error("Re-run the script to resume from the last completed bucket.")
        raise SystemExit(1) from exc
    finally:
        elapsed = time.monotonic() - start_time
        logger.info("Elapsed time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()