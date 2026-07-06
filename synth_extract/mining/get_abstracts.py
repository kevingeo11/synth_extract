from __future__ import annotations

import argparse
import html
import logging
import re
import sqlite3
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)

CROSSREF_WORK_URL = "https://api.crossref.org/works/{doi}"
OPENALEX_WORK_URL = "https://api.openalex.org/works/doi:{doi}"
SOURCE_TABLE = "papers"
DEFAULT_ENRICHED_TABLE = "papers_enriched"

ENRICHMENT_COLUMNS = {
    "abstract": "TEXT",
    "abstract_source": "TEXT",
    "authors_enriched": "TEXT",
    "authors_source": "TEXT",
    "affiliations_enriched": "TEXT",
    "affiliations_source": "TEXT",
}

RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class CrossrefResult:
    abstract: str | None
    authors: str | None
    affiliations: str | None


@dataclass(frozen=True)
class OpenAlexResult:
    abstract: str | None
    authors: str | None
    affiliations: str | None


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
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


def quote_identifier(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise ValueError(f"Invalid SQLite identifier: {identifier!r}")

    return '"' + identifier.replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    raw_columns = conn.execute(
        f"PRAGMA table_info({quote_identifier(table_name)})"
    ).fetchall()
    if not raw_columns:
        raise ValueError(f"Table {table_name!r} does not exist or has no columns.")

    column_names = ("cid", "name", "type", "notnull", "dflt_value", "pk")
    return [
        dict(zip(column_names, column))
        for column in raw_columns
    ]


def column_definition(column: Mapping[str, Any]) -> str:
    column_name = quote_identifier(column["name"])
    column_type = column["type"] or "TEXT"
    return f"{column_name} {column_type}"


def sync_enriched_schema(
    conn: sqlite3.Connection,
    enriched_table: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_columns = get_table_columns(conn, SOURCE_TABLE)
    source_column_names = [column["name"] for column in source_columns]
    source_column_names_set = set(source_column_names)
    enriched_table_exists = table_exists(conn, enriched_table)

    if enriched_table_exists and not resume:
        logger.warning(
            "Enriched table already exists | enriched_table=%s",
            enriched_table,
        )
        raise ValueError(
            f"Enriched table {enriched_table!r} already exists. "
            "Rerun with --resume to continue using it."
        )

    if not enriched_table_exists:
        definitions = [column_definition(column) for column in source_columns]
        for column_name, column_type in ENRICHMENT_COLUMNS.items():
            if column_name not in source_column_names_set:
                definitions.append(f"{quote_identifier(column_name)} {column_type}")

        primary_key_columns = [
            column["name"]
            for column in sorted(source_columns, key=lambda item: item["pk"])
            if column["pk"]
        ]
        if primary_key_columns:
            quoted_pk_columns = ", ".join(
                quote_identifier(column_name) for column_name in primary_key_columns
            )
            definitions.append(f"PRIMARY KEY ({quoted_pk_columns})")

        conn.execute(
            f"""
            CREATE TABLE {quote_identifier(enriched_table)} (
                {", ".join(definitions)}
            )
            """
        )
        conn.commit()
        return source_columns, get_table_columns(conn, enriched_table)

    logger.warning(
        "Resuming with existing enriched table | enriched_table=%s",
        enriched_table,
    )
    enriched_columns_list = get_table_columns(conn, enriched_table)
    validate_resume_schema(source_columns, enriched_columns_list, enriched_table)
    enriched_columns = {column["name"] for column in enriched_columns_list}
    expected_columns: list[tuple[str, str]] = [
        (column["name"], column["type"] or "TEXT")
        for column in source_columns
    ]
    expected_columns.extend(
        (column_name, column_type)
        for column_name, column_type in ENRICHMENT_COLUMNS.items()
        if column_name not in source_column_names_set
    )

    for column_name, column_type in expected_columns:
        if column_name in enriched_columns:
            continue

        logger.info("Adding %s.%s column", enriched_table, column_name)
        conn.execute(
            f"""
            ALTER TABLE {quote_identifier(enriched_table)}
            ADD COLUMN {quote_identifier(column_name)} {column_type or "TEXT"}
            """
        )

    conn.commit()
    return source_columns, get_table_columns(conn, enriched_table)


def primary_key_columns(columns: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        column["name"]
        for column in sorted(columns, key=lambda item: item["pk"])
        if column["pk"]
    ]


def validate_resume_schema(
    source_columns: list[dict[str, Any]],
    enriched_columns: list[dict[str, Any]],
    enriched_table: str,
) -> None:
    source_pk_columns = primary_key_columns(source_columns)
    enriched_pk_columns = primary_key_columns(enriched_columns)

    if source_pk_columns != enriched_pk_columns:
        raise ValueError(
            f"Existing enriched table {enriched_table!r} has primary key "
            f"{enriched_pk_columns or None}, expected {source_pk_columns or None}."
        )

    enriched_columns_by_name = {
        column["name"]: column
        for column in enriched_columns
    }
    for source_column in source_columns:
        enriched_column = enriched_columns_by_name.get(source_column["name"])
        if enriched_column is None:
            continue

        source_type = (source_column["type"] or "TEXT").casefold()
        enriched_type = (enriched_column["type"] or "TEXT").casefold()
        if source_type != enriched_type:
            raise ValueError(
                f"Existing enriched table {enriched_table!r} column "
                f"{source_column['name']!r} has type {enriched_column['type']!r}, "
                f"expected {source_column['type']!r}."
            )


def get_resume_key_columns(
    columns: list[dict[str, Any]],
    table_name: str,
) -> list[str]:
    pk_columns = primary_key_columns(columns)
    if pk_columns:
        return pk_columns

    column_names = {column["name"] for column in columns}
    if "eid" in column_names:
        return ["eid"]

    raise ValueError(
        f"Table {table_name!r} needs a primary key or eid column for resumable copying."
    )


def copy_new_papers(
    conn: sqlite3.Connection,
    enriched_table: str,
    source_columns: list[dict[str, Any]],
    enriched_columns: list[dict[str, Any]],
) -> int:
    source_column_names = [column["name"] for column in source_columns]
    enriched_column_names = {
        column["name"]
        for column in enriched_columns
    }
    copy_columns = [
        column_name
        for column_name in source_column_names
        if column_name in enriched_column_names
    ]
    key_columns = get_resume_key_columns(source_columns, SOURCE_TABLE)

    quoted_copy_columns = ", ".join(quote_identifier(column) for column in copy_columns)
    select_copy_columns = ", ".join(
        f"source.{quote_identifier(column)}" for column in copy_columns
    )
    exists_conditions = " AND ".join(
        f"target.{quote_identifier(column)} = source.{quote_identifier(column)}"
        for column in key_columns
    )

    total_changes_before = conn.total_changes
    conn.execute(
        f"""
        INSERT INTO {quote_identifier(enriched_table)} ({quoted_copy_columns})
        SELECT {select_copy_columns}
        FROM {quote_identifier(SOURCE_TABLE)} AS source
        WHERE NOT EXISTS (
            SELECT 1
            FROM {quote_identifier(enriched_table)} AS target
            WHERE {exists_conditions}
        )
        """
    )
    conn.commit()
    return conn.total_changes - total_changes_before


def prepare_enriched_table(
    conn: sqlite3.Connection,
    enriched_table: str,
    resume: bool,
) -> int:
    if not table_exists(conn, SOURCE_TABLE):
        raise ValueError("Database does not contain a papers table.")

    source_columns, enriched_columns = sync_enriched_schema(conn, enriched_table, resume)
    inserted_count = copy_new_papers(
        conn,
        enriched_table,
        source_columns,
        enriched_columns,
    )
    logger.info(
        "Prepared enriched table | source_table=%s enriched_table=%s copied_new_rows=%s",
        SOURCE_TABLE,
        enriched_table,
        inserted_count,
    )
    return inserted_count


def clean_abstract(raw_abstract: str | None) -> str | None:
    if not raw_abstract:
        return None

    clean = re.sub(
        r"<jats:title[^>]*>.*?</jats:title>",
        " ",
        raw_abstract,
        flags=re.I | re.S,
    )
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean or None


def value_is_missing(value: Any) -> bool:
    if value is None:
        return True

    text = str(value).strip()
    return not text or text.casefold() == "not found"


def normalize_doi(doi: Any) -> str:
    text = str(doi or "").strip()
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    text = re.sub(r"^https?://", "", text, flags=re.I)
    text = re.sub(r"^(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = text.split("#", 1)[0].split("?", 1)[0].strip()
    return text.casefold()


def format_crossref_author(author: Mapping[str, Any]) -> str | None:
    given = str(author.get("given", "")).strip()
    family = str(author.get("family", "")).strip()
    name = " ".join(part for part in (given, family) if part).strip()
    if name:
        return name

    literal_name = str(author.get("name", "")).strip()
    return literal_name or None


def format_crossref_authors(authors: Iterable[Mapping[str, Any]] | None) -> str | None:
    if not authors:
        return None

    names = [
        author_name
        for author in authors
        if isinstance(author, Mapping)
        for author_name in [format_crossref_author(author)]
        if author_name
    ]
    return "; ".join(names) or None


def format_crossref_affiliations(authors: Iterable[Mapping[str, Any]] | None) -> str | None:
    if not authors:
        return None

    institutions: list[str] = []
    seen = set()
    for author in authors:
        if not isinstance(author, Mapping):
            continue

        for affiliation in author.get("affiliation") or []:
            if not isinstance(affiliation, Mapping):
                continue

            institution = str(affiliation.get("name", "")).strip()
            institution_key = institution.casefold()
            if institution and institution_key not in seen:
                seen.add(institution_key)
                institutions.append(institution)

    return "; ".join(institutions) or None


def format_openalex_authors(authorships: Iterable[Mapping[str, Any]] | None) -> str | None:
    if not authorships:
        return None

    names = []
    seen = set()
    for authorship in authorships:
        if not isinstance(authorship, Mapping):
            continue

        author = authorship.get("author") or {}
        if not isinstance(author, Mapping):
            continue

        name = str(author.get("display_name", "")).strip()
        name_key = name.casefold()
        if name and name_key not in seen:
            seen.add(name_key)
            names.append(name)

    return "; ".join(names) or None


def format_openalex_affiliations(
    authorships: Iterable[Mapping[str, Any]] | None,
) -> str | None:
    if not authorships:
        return None

    affiliations = []
    seen = set()
    for authorship in authorships:
        if not isinstance(authorship, Mapping):
            continue

        for institution in authorship.get("institutions") or []:
            if not isinstance(institution, Mapping):
                continue

            name = str(institution.get("display_name", "")).strip()
            name_key = name.casefold()
            if name and name_key not in seen:
                seen.add(name_key)
                affiliations.append(name)

        for raw_affiliation in authorship.get("raw_affiliation_strings") or []:
            name = str(raw_affiliation).strip()
            name_key = name.casefold()
            if name and name_key not in seen:
                seen.add(name_key)
                affiliations.append(name)

    return "; ".join(affiliations) or None


def get_crossref_work(
    session: requests.Session,
    doi: str,
    mailto: str | None,
    timeout: int,
) -> CrossrefResult:
    headers = {"User-Agent": f"abstract-fetcher/0.1 (mailto:{mailto})"} if mailto else {}

    response = session.get(
        CROSSREF_WORK_URL.format(doi=quote(doi, safe="")),
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()

    message = response.json().get("message", {})
    authors = message.get("author") or []
    return CrossrefResult(
        abstract=clean_abstract(message.get("abstract")),
        authors=format_crossref_authors(authors),
        affiliations=format_crossref_affiliations(authors),
    )


def abstract_from_openalex_inverted_index(
    inverted_index: Mapping[str, list[int]] | None,
) -> str | None:
    if not inverted_index:
        return None

    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for position in positions:
            positioned_words.append((position, word))

    if not positioned_words:
        return None

    return " ".join(word for _, word in sorted(positioned_words)).strip() or None


def get_openalex_work(
    session: requests.Session,
    doi: str,
    mailto: str | None,
    timeout: int,
) -> OpenAlexResult:
    params = {"mailto": mailto} if mailto else None
    response = session.get(
        OPENALEX_WORK_URL.format(doi=quote(doi, safe="")),
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    authorships = data.get("authorships") or []
    return OpenAlexResult(
        abstract=abstract_from_openalex_inverted_index(data.get("abstract_inverted_index")),
        authors=format_openalex_authors(authorships),
        affiliations=format_openalex_affiliations(authorships),
    )


def iter_unenriched_papers(
    conn: sqlite3.Connection,
    enriched_table: str,
    limit: int | None = None,
):
    query = f"""
        SELECT
            rowid,
            doi,
            abstract,
            authors_enriched,
            affiliations_enriched
        FROM {quote_identifier(enriched_table)}
        WHERE doi IS NOT NULL
          AND trim(doi) != ''
          AND (
              abstract IS NULL
              OR trim(abstract) = ''
              OR lower(trim(abstract)) = 'not found'
              OR authors_enriched IS NULL
              OR trim(authors_enriched) = ''
              OR lower(trim(authors_enriched)) = 'not found'
              OR affiliations_enriched IS NULL
              OR trim(affiliations_enriched) = ''
              OR lower(trim(affiliations_enriched)) = 'not found'
          )
        ORDER BY rowid
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    return conn.execute(query, params)


def update_missing_field(
    conn: sqlite3.Connection,
    enriched_table: str,
    rowid: int,
    value_column: str,
    source_column: str,
    value: str,
    source: str,
) -> bool:
    cursor = conn.execute(
        f"""
        UPDATE {quote_identifier(enriched_table)}
        SET {quote_identifier(value_column)} = ?,
            {quote_identifier(source_column)} = ?
        WHERE rowid = ?
          AND (
              {quote_identifier(value_column)} IS NULL
              OR trim({quote_identifier(value_column)}) = ''
              OR lower(trim({quote_identifier(value_column)})) = 'not found'
          )
        """,
        (value, source, rowid),
    )
    return cursor.rowcount > 0


def fetch_and_update_abstracts(
    db_path: str,
    enriched_table: str = DEFAULT_ENRICHED_TABLE,
    resume: bool = False,
    mailto: str | None = None,
    limit: int | None = None,
    timeout: int = 30,
    request_delay: float = 0.1,
    commit_every: int = 25,
) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    session = build_session()

    stats = {
        "copied_new_rows": 0,
        "checked": 0,
        "crossref_abstracts": 0,
        "crossref_authors": 0,
        "crossref_affiliations": 0,
        "openalex_abstracts": 0,
        "openalex_authors": 0,
        "openalex_affiliations": 0,
        "failed_crossref": 0,
        "failed_openalex": 0,
        "no_fields_found": 0,
        "skipped_invalid_doi": 0,
    }

    try:
        stats["copied_new_rows"] = prepare_enriched_table(
            conn,
            enriched_table,
            resume,
        )

        for row in iter_unenriched_papers(conn, enriched_table, limit):
            stats["checked"] += 1
            rowid = row["rowid"]
            doi = normalize_doi(row["doi"])
            if not doi:
                stats["skipped_invalid_doi"] += 1
                logger.warning("Skipping row with empty DOI after normalization | rowid=%s", rowid)
                continue

            abstract_missing = value_is_missing(row["abstract"])
            authors_missing = value_is_missing(row["authors_enriched"])
            affiliations_missing = value_is_missing(row["affiliations_enriched"])

            logger.info(
                "Checking DOI | enriched_table=%s rowid=%s doi=%s",
                enriched_table,
                rowid,
                doi,
            )

            crossref_result = None
            try:
                crossref_result = get_crossref_work(
                    session=session,
                    doi=doi,
                    mailto=mailto,
                    timeout=timeout,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    logger.info("Crossref work not found | doi=%s", doi)
                else:
                    logger.warning("Crossref request failed | doi=%s error=%s", doi, exc)
                stats["failed_crossref"] += 1
            except Exception as exc:
                logger.warning("Crossref request failed | doi=%s error=%s", doi, exc)
                stats["failed_crossref"] += 1

            if crossref_result is not None:
                if abstract_missing and crossref_result.abstract:
                    if update_missing_field(
                        conn,
                        enriched_table,
                        rowid,
                        "abstract",
                        "abstract_source",
                        crossref_result.abstract,
                        "crossref",
                    ):
                        stats["crossref_abstracts"] += 1
                        abstract_missing = False
                        logger.info("Stored Crossref abstract | rowid=%s doi=%s", rowid, doi)

                if authors_missing and crossref_result.authors:
                    if update_missing_field(
                        conn,
                        enriched_table,
                        rowid,
                        "authors_enriched",
                        "authors_source",
                        crossref_result.authors,
                        "crossref",
                    ):
                        stats["crossref_authors"] += 1
                        authors_missing = False
                        logger.info("Stored Crossref authors | rowid=%s doi=%s", rowid, doi)

                if affiliations_missing and crossref_result.affiliations:
                    if update_missing_field(
                        conn,
                        enriched_table,
                        rowid,
                        "affiliations_enriched",
                        "affiliations_source",
                        crossref_result.affiliations,
                        "crossref",
                    ):
                        stats["crossref_affiliations"] += 1
                        affiliations_missing = False
                        logger.info(
                            "Stored Crossref affiliations | rowid=%s doi=%s",
                            rowid,
                            doi,
                        )

                conn.commit()

            if not (abstract_missing or authors_missing or affiliations_missing):
                if request_delay > 0:
                    time.sleep(request_delay)
                continue

            openalex_result = None
            try:
                openalex_result = get_openalex_work(
                    session=session,
                    doi=doi,
                    mailto=mailto,
                    timeout=timeout,
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    logger.info("OpenAlex work not found | doi=%s", doi)
                else:
                    logger.warning("OpenAlex request failed | doi=%s error=%s", doi, exc)
                stats["failed_openalex"] += 1
            except Exception as exc:
                logger.warning("OpenAlex request failed | doi=%s error=%s", doi, exc)
                stats["failed_openalex"] += 1

            found_openalex_field = False
            if openalex_result is not None:
                if abstract_missing and openalex_result.abstract:
                    if update_missing_field(
                        conn,
                        enriched_table,
                        rowid,
                        "abstract",
                        "abstract_source",
                        openalex_result.abstract,
                        "openalex",
                    ):
                        stats["openalex_abstracts"] += 1
                        found_openalex_field = True
                        logger.info("Stored OpenAlex abstract | rowid=%s doi=%s", rowid, doi)

                if authors_missing and openalex_result.authors:
                    if update_missing_field(
                        conn,
                        enriched_table,
                        rowid,
                        "authors_enriched",
                        "authors_source",
                        openalex_result.authors,
                        "openalex",
                    ):
                        stats["openalex_authors"] += 1
                        found_openalex_field = True
                        logger.info("Stored OpenAlex authors | rowid=%s doi=%s", rowid, doi)

                if affiliations_missing and openalex_result.affiliations:
                    if update_missing_field(
                        conn,
                        enriched_table,
                        rowid,
                        "affiliations_enriched",
                        "affiliations_source",
                        openalex_result.affiliations,
                        "openalex",
                    ):
                        stats["openalex_affiliations"] += 1
                        found_openalex_field = True
                        logger.info(
                            "Stored OpenAlex affiliations | rowid=%s doi=%s",
                            rowid,
                            doi,
                        )

            if not found_openalex_field:
                stats["no_fields_found"] += 1

            if commit_every > 0 and stats["checked"] % commit_every == 0:
                conn.commit()
                logger.info(
                    "Committed abstract enrichment progress | checked=%s stats=%s",
                    stats["checked"],
                    stats,
                )

            if request_delay > 0:
                time.sleep(request_delay)

        conn.commit()
    finally:
        session.close()
        conn.close()

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy papers into an enriched table and populate abstracts from "
            "Crossref, then OpenAlex as a DOI-only fallback."
        )
    )
    parser.add_argument("--db-path", default="data/scopus.sqlite")
    parser.add_argument("--enriched-table", default=DEFAULT_ENRICHED_TABLE)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing enriched table instead of stopping.",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        help="Optional contact email to send to Crossref/OpenAlex.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--commit-every", type=int, default=25)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    try:
        stats = fetch_and_update_abstracts(
            db_path=args.db_path,
            enriched_table=args.enriched_table,
            resume=args.resume,
            mailto=args.mailto,
            limit=args.limit,
            timeout=args.timeout,
            request_delay=args.request_delay,
            commit_every=args.commit_every,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Abstract update complete | stats=%s", stats)


if __name__ == "__main__":
    main()
