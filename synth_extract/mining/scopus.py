import os
import logging
import sqlite3
import time
import requests
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)

SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"

DEFAULT_SCOPUS_FIELDS = (
    "eid,"
    "prism:doi,"
    "dc:title,"
    "prism:publicationName,"
    "dc:publisher,"
    "openaccess,"
    "openaccessFlag,"
    "dc:creator,"
    "affiliation"
)

RATE_LIMIT_HEADER_NAMES = (
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
    "X-ELS-Status",
    "X-ELS-ReqId",
    "Retry-After",
)

SENSITIVE_HEADER_NAMES = {"x-els-apikey", "authorization", "cookie", "set-cookie"}

# Statuses worth retrying: rate limiting + transient server-side failures.
# Anything else (401/403 bad key, 400 bad query, etc.) is a real problem and
# should raise immediately instead of being retried.
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)
RETRY_TOTAL_ATTEMPTS = 5
RETRY_BACKOFF_FACTOR = 1.0

OPENSEARCH_RESULT_KEYS = (
    "opensearch:totalResults",
    "opensearch:startIndex",
    "opensearch:itemsPerPage",
)

LOG_SEPARATOR = "-" * 50


def _log_major_separator(label: str) -> None:
    logger.info("%s %s %s", LOG_SEPARATOR, label, LOG_SEPARATOR)


def _build_scopus_session() -> requests.Session:
    """Build a requests.Session with retries for transient Scopus failures.

    Retries connection errors and the status codes in RETRY_STATUS_FORCELIST
    with exponential backoff, honoring the Retry-After header when present.
    Non-transient errors (auth failures, bad requests) are NOT retried and
    will surface as raised exceptions right away.
    """
    retry = Retry(
        total=RETRY_TOTAL_ATTEMPTS,
        connect=RETRY_TOTAL_ATTEMPTS,
        read=RETRY_TOTAL_ATTEMPTS,
        status=RETRY_TOTAL_ATTEMPTS,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_scopus_session()


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = {
        key: "***REDACTED***" if key.lower() in SENSITIVE_HEADER_NAMES else value
        for key, value in headers.items()
    }
    reset_value = _get_header_case_insensitive(headers, "X-RateLimit-Reset")
    if reset_value is not None:
        redacted["X-RateLimit-Reset-Readable"] = _format_unix_timestamp(reset_value)

    return redacted


def _get_header_case_insensitive(headers: Mapping[str, str], name: str) -> str | None:
    if value := headers.get(name):
        return value

    target_name = name.lower()
    for header_name, value in headers.items():
        if header_name.lower() == target_name:
            return value

    return None


def _get_scopus_api_key(api_key: str | None = None) -> str:
    scopus_api_key = api_key or os.getenv("SCOPUS_API_KEY")

    if not scopus_api_key:
        logger.error("SCOPUS_API_KEY environment variable is not set.")
        raise ValueError("SCOPUS_API_KEY environment variable is not set.")

    return scopus_api_key


def _build_scopus_headers(api_key: str | None = None) -> dict[str, str]:
    return {
        "X-ELS-APIKey": _get_scopus_api_key(api_key),
        "Accept": "application/json",
    }


def _build_scopus_search_params(
    query: str,
    fields: str = DEFAULT_SCOPUS_FIELDS,
    count: int = 25,
    view: str = "STANDARD",
    cursor: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "query": query,
        "field": fields,
        "count": count,
        "view": view,
    }

    if cursor is not None:
        params["cursor"] = cursor

    if extra_params:
        params.update(extra_params)

    return params


def _log_response_details(response: requests.Response, body_preview_chars: int = 1000) -> None:
    request = response.request
    request_headers = _redact_headers(dict(request.headers))
    response_headers = _redact_headers(dict(response.headers))

    logger.debug(
        "Scopus request details | method=%s url=%s request_headers=%s",
        request.method,
        request.url,
        request_headers,
    )
    logger.debug(
        "Scopus response details | status_code=%s elapsed_seconds=%.3f headers=%s",
        response.status_code,
        response.elapsed.total_seconds(),
        response_headers,
    )

    if response.status_code >= 400:
        logger.error(
            "Scopus response error body preview | status_code=%s body=%s",
            response.status_code,
            response.text[:body_preview_chars],
        )


def _raise_for_scopus_error(response: requests.Response) -> None:
    if response.status_code == 200:
        return

    raise RuntimeError(
        f"Request failed with status {response.status_code}. "
        f"ELS Status: {response.headers.get('X-ELS-Status')}. "
        f"Rate limit remaining: {response.headers.get('X-RateLimit-Remaining')}. "
        f"Response: {response.text[:500]}"
    )


def extract_scopus_header_info(
    headers: Mapping[str, str],
) -> dict[str, str | None]:
    """Extract useful Scopus response headers, including rate-limit information."""
    header_info = {
        name: _redact_header_value(name, headers.get(name))
        for name in RATE_LIMIT_HEADER_NAMES
    }

    for name, value in headers.items():
        normalized_name = name.lower()
        if normalized_name.startswith(("x-ratelimit", "x-els")):
            header_info.setdefault(name, _redact_header_value(name, value))

    reset_value = _get_header_case_insensitive(headers, "X-RateLimit-Reset")
    header_info["X-RateLimit-Reset-Readable"] = _format_unix_timestamp(reset_value)

    return header_info


def get_scopus_rate_limit_data(
    response_or_headers: requests.Response | Mapping[str, str],
) -> dict[str, int | str | None]:
    """Extract Scopus API rate-limit data from a response or headers mapping."""
    headers = (
        response_or_headers.headers
        if isinstance(response_or_headers, requests.Response)
        else response_or_headers
    )

    def parse_int_header(name: str) -> int | None:
        value = _get_header_case_insensitive(headers, name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    reset_timestamp = parse_int_header("X-RateLimit-Reset")

    return {
        "limit": parse_int_header("X-RateLimit-Limit"),
        "remaining": parse_int_header("X-RateLimit-Remaining"),
        "reset": reset_timestamp,
        "reset_readable": _format_unix_timestamp(str(reset_timestamp))
        if reset_timestamp is not None
        else None,
        "retry_after": parse_int_header("Retry-After"),
        "els_status": _get_header_case_insensitive(headers, "X-ELS-Status"),
        "els_request_id": _get_header_case_insensitive(headers, "X-ELS-ReqId"),
    }


def _redact_header_value(name: str, value: str | None) -> str | None:
    if value is None:
        return None

    if name.lower() in SENSITIVE_HEADER_NAMES:
        return "***REDACTED***"

    return value


def _format_unix_timestamp(value: str | None) -> str | None:
    if value is None:
        return None

    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def extract_scopus_result_info(search_results: Mapping[str, Any]) -> dict[str, Any]:
    """Extract useful OpenSearch counters from a Scopus search-results object."""
    return {key: search_results.get(key) for key in OPENSEARCH_RESULT_KEYS}


def format_affiliations(affiliations: Any) -> str | None:
    """Combine Scopus affiliation objects into one readable database field."""
    if not affiliations:
        return None

    if isinstance(affiliations, dict):
        affiliation_items = [affiliations]
    elif isinstance(affiliations, list):
        affiliation_items = affiliations
    else:
        return str(affiliations)

    formatted = []
    for affiliation in affiliation_items:
        if not isinstance(affiliation, Mapping):
            formatted.append(str(affiliation))
            continue

        parts = [
            affiliation.get("affilname"),
            affiliation.get("affiliation-city"),
            affiliation.get("affiliation-country"),
        ]
        value = ", ".join(str(part) for part in parts if part)
        if value:
            formatted.append(value)

    return "; ".join(formatted) or None


def _execute_scopus_request(
    params: dict[str, Any],
    api_key: str | None,
    timeout: int,
) -> requests.Response:
    """Single point of contact with the Scopus API.

    Uses the shared retrying session, logs request/response details, and
    raises on any response that isn't a clean 200 (including a response
    that is still failing after all retries have been exhausted).
    """
    response = SESSION.get(
        SCOPUS_SEARCH_URL,
        params=params,
        headers=_build_scopus_headers(api_key),
        timeout=timeout,
    )
    _log_response_details(response)
    _raise_for_scopus_error(response)
    return response


def fetch_scopus_json(
    query: str | None = None,
    params: dict[str, Any] | None = None,
    fields: str = DEFAULT_SCOPUS_FIELDS,
    count: int = 25,
    view: str = "STANDARD",
    cursor: str | None = None,
    api_key: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Send one Scopus search request (with retries) and return the JSON response."""
    if query is None and params is None:
        raise ValueError("Either query or params must be provided.")

    request_params = dict(params or {})
    if query is not None:
        request_params = _build_scopus_search_params(
            query=query,
            fields=fields,
            count=count,
            view=view,
            cursor=cursor,
            extra_params=request_params,
        )

    logger.info("Sending Scopus request | params=%s", request_params)

    response = _execute_scopus_request(request_params, api_key, timeout)

    try:
        data = response.json()
    except ValueError:
        logger.exception(
            "Scopus response was not valid JSON | body=%s",
            response.text[:1000],
        )
        raise

    search_results = data.get("search-results", {})
    entries = search_results.get("entry", [])
    result_info = extract_scopus_result_info(search_results)
    logger.info(
        "Received Scopus JSON | entries=%s total_results=%s start_index=%s "
        "items_per_page=%s next_cursor=%s headers=%s",
        len(entries),
        result_info.get("opensearch:totalResults"),
        result_info.get("opensearch:startIndex"),
        result_info.get("opensearch:itemsPerPage"),
        search_results.get("cursor", {}).get("@next"),
        extract_scopus_header_info(response.headers),
    )

    return data


def get_scopus_header_info(
    query: str,
    params: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: int = 60,
) -> dict[str, str | None]:
    """Send one lightweight Scopus request (with retries) and return response headers."""
    request_params = _build_scopus_search_params(
        query=query,
        count=1,
        extra_params=params,
    )

    logger.info("Sending Scopus header-info request | params=%s", request_params)

    response = _execute_scopus_request(request_params, api_key, timeout)

    header_info = extract_scopus_header_info(response.headers)
    logger.info("Scopus header info | headers=%s", header_info)
    return header_info


def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            eid TEXT PRIMARY KEY,
            doi TEXT,
            title TEXT,
            author TEXT,
            journal TEXT,
            publisher TEXT,
            open_access TEXT,
            open_access_flag TEXT,
            affiliation TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crawl_state (
            publisher TEXT PRIMARY KEY,
            cursor TEXT,
            pages_completed INTEGER DEFAULT 0,
            finished INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    logger.info("Initialized Scopus SQLite database | db_path=%s", db_path)
    return conn


def get_saved_cursor(conn, publisher: str):
    row = conn.execute(
        "SELECT cursor, pages_completed, finished FROM crawl_state WHERE publisher = ?",
        (publisher,)
    ).fetchone()

    if row is None:
        logger.info("No saved Scopus cursor found | publisher=%s", publisher)
        return "*", 0, False

    cursor, pages_completed, finished = row
    logger.info(
        "Loaded saved Scopus cursor | publisher=%s pages_completed=%s finished=%s cursor=%s",
        publisher,
        pages_completed,
        bool(finished),
        cursor,
    )
    return cursor, pages_completed, bool(finished)


def save_state(
    conn,
    publisher: str,
    cursor: str,
    pages_completed: int,
    finished: bool = False,
):
    conn.execute("""
        INSERT INTO crawl_state (publisher, cursor, pages_completed, finished)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(publisher) DO UPDATE SET
            cursor = excluded.cursor,
            pages_completed = excluded.pages_completed,
            finished = excluded.finished
    """, (publisher, cursor, pages_completed, int(finished)))
    conn.commit()
    logger.info(
        "Saved Scopus crawl state | publisher=%s pages_completed=%s finished=%s cursor=%s",
        publisher,
        pages_completed,
        finished,
        cursor,
    )


def save_records(conn, records: list[dict]):
    total_changes_before = conn.total_changes
    conn.executemany("""
        INSERT OR IGNORE INTO papers (
            eid, doi, title, author, journal, publisher, open_access, open_access_flag, affiliation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            r["eid"],
            r["doi"],
            r["title"],
            r.get("author"),
            r["journal"],
            r["publisher"],
            r["open_access"],
            r["open_access_flag"],
            r.get("affiliation"),
        )
        for r in records
        if r.get("eid")
    ])
    conn.commit()
    inserted_count = conn.total_changes - total_changes_before
    total_saved_count = count_saved_records(conn)
    logger.info(
        "Saved Scopus records | attempted=%s inserted=%s ignored_duplicates=%s total_saved=%s",
        len(records),
        inserted_count,
        len(records) - inserted_count,
        total_saved_count,
    )
    return inserted_count


def count_saved_records(conn) -> int:
    row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    return int(row[0])


def fetch_scopus_by_publisher(
    publisher: str,
    db_path: str = "scopus_results.sqlite",
    count: int = 200,
    max_pages: int | None = None,
    api_key: str | None = None,
    timeout: int = 60,
    start: bool = False,
    request_delay: float = 0.2,
):
    _log_major_separator("SCOPUS CRAWL INITIALIZATION")

    if start:
        if Path(db_path).exists():
            raise ValueError(
                f"db_path already exists, refusing to overwrite fresh crawl: {db_path}"
            )

        logger.info(
            "Starting Scopus crawl from scratch | publisher=%s db_path=%s",
            publisher,
            db_path,
        )

    conn = init_db(db_path)

    try:
        if start:
            cursor, pages_completed, finished = "*", 0, False
        else:
            cursor, pages_completed, finished = get_saved_cursor(conn, publisher)

        if finished:
            logger.info("Crawl already finished for publisher | publisher=%s", publisher)
            return

        query = f'PUBLISHER({publisher}) AND TITLE-ABS-KEY(polymer* OR copolymer*)'

        logger.info(
            "Starting Scopus publisher crawl | publisher=%s db_path=%s count=%s "
            "max_pages=%s cursor=%s pages_completed=%s",
            publisher,
            db_path,
            count,
            max_pages,
            cursor,
            pages_completed,
        )

        pages_this_run = 0

        while True:
            _log_major_separator(f"SCOPUS REQUEST PAGE {pages_completed + 1}")

            params = _build_scopus_search_params(
                query=query,
                count=count,
                view="STANDARD",
                cursor=cursor,
            )

            logger.info(
                "Fetching Scopus publisher page | publisher=%s page=%s cursor=%s",
                publisher,
                pages_completed + 1,
                cursor,
            )

            # A serious/non-retryable failure (auth issue, exhausted retries,
            # malformed JSON, etc.) will raise here. Save state first so the
            # crawl can resume from the last good cursor next time.
            try:
                data = fetch_scopus_json(
                    params=params,
                    api_key=api_key,
                    timeout=timeout,
                )
            except Exception:
                save_state(conn, publisher, cursor, pages_completed, finished=False)
                logger.exception(
                    "Scopus publisher crawl failed | publisher=%s pages_completed=%s cursor=%s",
                    publisher,
                    pages_completed,
                    cursor,
                )
                raise

            search_results = data.get("search-results", {})
            entries = search_results.get("entry", [])

            records = []
            for e in entries:
                records.append({
                    "eid": e.get("eid"),
                    "doi": e.get("prism:doi"),
                    "title": e.get("dc:title"),
                    "author": e.get("dc:creator"),
                    "journal": e.get("prism:publicationName"),
                    "publisher": e.get("dc:publisher") or publisher,
                    "open_access": e.get("openaccess"),
                    "open_access_flag": e.get("openaccessFlag"),
                    "affiliation": format_affiliations(e.get("affiliation")),
                })

            inserted_count = save_records(conn, records)

            next_cursor = search_results.get("cursor", {}).get("@next")
            result_info = extract_scopus_result_info(search_results)

            logger.info(
                "Processed Scopus publisher page | publisher=%s page=%s entries=%s "
                "inserted=%s total_results=%s start_index=%s items_per_page=%s "
                "next_cursor=%s",
                publisher,
                pages_completed + 1,
                len(entries),
                inserted_count,
                result_info.get("opensearch:totalResults"),
                result_info.get("opensearch:startIndex"),
                result_info.get("opensearch:itemsPerPage"),
                next_cursor,
            )

            pages_completed += 1
            pages_this_run += 1

            if not entries or not next_cursor or next_cursor == cursor:
                save_state(conn, publisher, cursor, pages_completed, finished=True)
                logger.info(
                    "Finished Scopus publisher crawl | publisher=%s pages=%s",
                    publisher,
                    pages_completed,
                )
                break

            cursor = next_cursor
            save_state(conn, publisher, cursor, pages_completed, finished=False)

            if max_pages is not None and pages_this_run >= max_pages:
                logger.info(
                    "Stopping Scopus publisher crawl because max_pages was reached | "
                    "publisher=%s max_pages=%s",
                    publisher,
                    max_pages,
                )
                break

            if request_delay > 0:
                time.sleep(request_delay)
    finally:
        _log_major_separator("SCOPUS CRAWL SUMMARY")

        try:
            logger.info(
                "Scopus database saved entry count | db_path=%s saved_entries=%s",
                db_path,
                count_saved_records(conn),
            )
        except Exception:
            logger.exception("Could not count saved Scopus records | db_path=%s", db_path)
        conn.close()
        logger.debug("Closed Scopus SQLite database | db_path=%s", db_path)
