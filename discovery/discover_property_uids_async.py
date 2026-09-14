#!/usr/bin/env python3
"""Discover measured-property spans for UID-manifest papers.

The worker reads UIDs from a text, JSON, or JSONL manifest, selects only rows
whose ``polymer_material_papers.pty_no_res`` value is NULL, and processes their
Markdown full text concurrently. Successful JSON arrays are written to SQLite
in short transactions. Failures and missing Markdown remain NULL, so rerunning
the same manifest retries only unfinished rows.

The property schema and ``PropertyDiscoveryLLM`` live in this script on purpose;
the only shared agent component is the provider-neutral ``LLMBackend``.

Example
-------
python discovery/discover_property_uids_async.py \
    --db-path data/discovery_workspace.db \
    --uid-file data/uid_manifest/uid_manifest_0001.txt \
    --fulltext-root data/fulltext \
    --model qwen3.6-27b \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key none \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synth_extract.agents.llm import LLMBackend  # noqa: E402


LOG = logging.getLogger("property_uid_discovery_async")
TABLE = "polymer_material_papers"
RESULT_COLUMN = "pty_no_res"
PROMPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = PROMPT_DIR / "property_discovery.md"
DEFAULT_USER_PROMPT_PATH = PROMPT_DIR / "user_prompt.md"


class TokenUsage(BaseModel):
    """Token consumption reported by the provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class CompletionMetadata(BaseModel):
    """Useful metadata taken from a raw chat completion."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    created: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    reasoning: str | None = None
    usage: TokenUsage | None = None


class PropertyExtractionResult(BaseModel):
    """Verbatim measured-property names extracted from one paper."""

    model_config = ConfigDict(extra="forbid", strict=True)

    properties: list[str] = Field(
        description=(
            "Property-name spans for which a corresponding property value is "
            "explicitly reported. Duplicate spans are allowed."
        )
    )
    metadata: CompletionMetadata

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, properties: list[str]) -> list[str]:
        """Reject empty spans while deliberately preserving duplicates."""
        if any(not property_name.strip() for property_name in properties):
            raise ValueError("Property spans must be non-empty strings.")
        return properties


class PropertyDiscoveryLLM:
    """Extract measured-property spans using an ``LLMBackend``."""

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
        user_prompt_path: str | Path = DEFAULT_USER_PROMPT_PATH,
    ) -> None:
        self.backend = backend
        self.system_prompt_path = Path(system_prompt_path).expanduser().resolve()
        self.user_prompt_path = Path(user_prompt_path).expanduser().resolve()
        self.reload_prompts()

    @staticmethod
    def _load_prompt(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {path}")
        return path.read_text(encoding="utf-8").strip()

    def reload_prompts(self) -> None:
        """Reload both prompt files from disk."""
        self._system_prompt = self._load_prompt(self.system_prompt_path)
        self._user_prompt = self._load_prompt(self.user_prompt_path)

    @staticmethod
    def response_format() -> dict[str, Any]:
        """Require a bare JSON string array without a uniqueness constraint."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "measured_property_spans",
                "strict": True,
                "schema": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }

    def build_messages(self, fulltext: str) -> list[dict[str, str]]:
        """Build messages for one non-empty Markdown paper."""
        if not isinstance(fulltext, str) or not fulltext.strip():
            raise ValueError("fulltext must be a non-empty string")
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": self._user_prompt.format(fulltext=fulltext.strip()),
            },
        ]

    def render_request(self, fulltext: str) -> str:
        """Render the exact backend request without sending it."""
        return self.backend.render_request(
            messages=self.build_messages(fulltext),
            response_format=self.response_format(),
        )

    @staticmethod
    def _metadata(completion: Any) -> CompletionMetadata:
        choice = completion.choices[0]
        message = choice.message
        usage = getattr(completion, "usage", None)
        token_usage = None
        if usage is not None:
            token_usage = TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )
        return CompletionMetadata(
            model=getattr(completion, "model", None),
            created=getattr(completion, "created", None),
            finish_reason=getattr(choice, "finish_reason", None),
            stop_reason=getattr(choice, "stop_reason", None),
            reasoning=getattr(message, "reasoning", None),
            usage=token_usage,
        )

    @classmethod
    def _parse_completion(cls, completion: Any) -> PropertyExtractionResult:
        if not completion.choices:
            raise ValueError("The provider returned no completion choices.")

        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("The property response reached the token limit.")

        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise ValueError(f"The provider refused the request: {refusal}")

        content = choice.message.content
        if not content:
            raise ValueError("The provider returned an empty response.")

        properties = json.loads(content)
        if not isinstance(properties, list):
            raise ValueError("The response must be a JSON array.")
        if not all(isinstance(item, str) for item in properties):
            raise ValueError("Every property span must be a string.")
        return PropertyExtractionResult(
            properties=properties,
            metadata=cls._metadata(completion),
        )

    async def aextract_raw(self, fulltext: str) -> Any:
        """Return the raw asynchronous completion."""
        return await self.backend.acreate_completion(
            messages=self.build_messages(fulltext),
            response_format=self.response_format(),
        )

    async def aextract(self, fulltext: str) -> PropertyExtractionResult:
        """Extract and validate measured-property spans asynchronously."""
        completion = await self.aextract_raw(fulltext)
        return self._parse_completion(completion)

    def prompt_hash(self) -> str:
        prompt_text = f"{self._system_prompt}\n\n{self._user_prompt}"
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    def llm_config(self) -> dict[str, Any]:
        return {
            **self.backend.config(),
            "system_prompt_path": str(self.system_prompt_path),
            "user_prompt_path": str(self.user_prompt_path),
            "prompt_hash": self.prompt_hash(),
        }

    def health_check(self) -> bool:
        self.backend.list_models()
        return True


@dataclass
class Progress:
    requested: int = 0
    already_completed: int = 0
    attempted: int = 0
    updated: int = 0
    properties_returned: int = 0
    discovery_errors: int = 0
    missing_markdown: int = 0
    database_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class PaperRow:
    paper_id: int
    paper_uid: str
    canonical_source: str


@dataclass(frozen=True)
class DiscoveryItem:
    paper: PaperRow
    markdown_path: Path
    result: PropertyExtractionResult | None
    latency_seconds: float
    local_error: str | None = None
    markdown_missing: bool = False


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


def json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover measured-property spans for pending manifest UIDs and "
            f"update {TABLE}.{RESULT_COLUMN}."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=REPO_ROOT / "data" / "discovery_workspace.db",
        help="SQLite discovery workspace (default: %(default)s)",
    )
    parser.add_argument(
        "--uid-file",
        type=Path,
        required=True,
        help="UID manifest: one UID per line, JSON, or JSONL",
    )
    parser.add_argument(
        "--fulltext-root",
        type=Path,
        default=REPO_ROOT / "data" / "fulltext",
        help="Root containing source/paper_uid/paper_uid.md (default: %(default)s)",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Property-discovery system prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--user-template-path",
        type=Path,
        default=DEFAULT_USER_PROMPT_PATH,
        help="Property-discovery user template (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID exposed by the OpenAI-compatible server",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default="none",
        help="API key or placeholder for servers without authentication",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum pending manifest UIDs to attempt (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=25,
        help="Discoveries per SQLite transaction (default: %(default)s)",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=positive_int,
        default=int(os.getenv("MAX_PARALLEL_REQUESTS", "8")),
        help="Maximum in-flight model requests (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=300.0,
        help="Model request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=16384,
        help="Maximum completion tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--extra-body",
        type=json_object,
        default=None,
        help="Optional JSON object forwarded as OpenAI extra_body",
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


def _uids_from_json_value(value: Any, source: str) -> list[str]:
    if isinstance(value, dict):
        if "uids" not in value:
            raise ValueError(f"{source} must contain a top-level 'uids' array")
        value = value["uids"]
    if not isinstance(value, list):
        raise ValueError(f"{source} must contain a JSON array of UIDs")
    if not all(isinstance(uid, str) for uid in value):
        raise ValueError(f"Every UID in {source} must be a string")
    return value


def load_uid_file(uid_file: Path) -> list[str]:
    """Load and strictly validate a UID manifest."""
    uid_file = uid_file.expanduser().resolve()
    if not uid_file.is_file():
        raise FileNotFoundError(f"UID file does not exist: {uid_file}")

    if uid_file.suffix.lower() == ".json":
        with uid_file.open(encoding="utf-8-sig") as handle:
            uids = _uids_from_json_value(json.load(handle), str(uid_file))
    elif uid_file.suffix.lower() == ".jsonl":
        uids = []
        with uid_file.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, str):
                    uid = value
                elif isinstance(value, dict) and isinstance(
                    value.get("paper_uid"), str
                ):
                    uid = value["paper_uid"]
                else:
                    raise ValueError(
                        f"Invalid JSONL UID at {uid_file}:{line_number}"
                    )
                uids.append(uid)
    else:
        with uid_file.open(encoding="utf-8-sig") as handle:
            uids = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]

    uids = [uid.strip() for uid in uids]
    if not uids:
        raise ValueError(f"UID file is empty: {uid_file}")
    invalid = [uid for uid in uids if not safe_path_component(uid)]
    if invalid:
        raise ValueError(f"Invalid UID path component(s): {invalid[:5]}")
    duplicates = sorted(uid for uid, count in Counter(uids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate UID(s) in manifest: {duplicates[:5]}")
    return uids


def validate_discovery_table(connection: sqlite3.Connection) -> None:
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
    """Validate a completed manifest row before treating it as finished."""
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


def load_manifest_rows(
    connection: sqlite3.Connection,
    uids: list[str],
) -> tuple[list[PaperRow], int]:
    """Resolve manifest UIDs and return rows whose property JSON is NULL."""
    connection.execute(
        "CREATE TEMP TABLE requested_property_uids ("
        "position INTEGER PRIMARY KEY, paper_uid TEXT NOT NULL UNIQUE)"
    )
    connection.executemany(
        "INSERT INTO requested_property_uids(position, paper_uid) VALUES (?, ?)",
        enumerate(uids),
    )
    rows = connection.execute(
        f"""
        SELECT
            requested.position,
            requested.paper_uid,
            papers.paper_id,
            papers.canonical_source,
            papers.{quote_identifier(RESULT_COLUMN)}
        FROM requested_property_uids AS requested
        LEFT JOIN {quote_identifier(TABLE)} AS papers
            ON papers.paper_uid = requested.paper_uid
        ORDER BY requested.position
        """
    ).fetchall()

    missing_uids = [str(row[1]) for row in rows if row[2] is None]
    if missing_uids:
        preview = ", ".join(missing_uids[:10])
        suffix = " ..." if len(missing_uids) > 10 else ""
        raise RuntimeError(
            f"{len(missing_uids)} manifest UID(s) are absent from {TABLE}: "
            f"{preview}{suffix}"
        )

    pending: list[PaperRow] = []
    already_completed = 0
    for _, paper_uid, paper_id, canonical_source, stored_result in rows:
        uid = str(paper_uid)
        source = str(canonical_source)
        if not safe_path_component(source):
            raise RuntimeError(
                f"Unsafe canonical_source for paper_uid={uid}: {source!r}"
            )
        if stored_result is not None:
            validate_stored_property_json(stored_result, uid)
            already_completed += 1
            continue
        pending.append(
            PaperRow(
                paper_id=int(paper_id),
                paper_uid=uid,
                canonical_source=source,
            )
        )
    return pending, already_completed


def chunks(items: list[PaperRow], size: int) -> Iterator[list[PaperRow]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def discover_paper(
    discovery_llm: PropertyDiscoveryLLM,
    semaphore: asyncio.Semaphore,
    fulltext_root: Path,
    paper: PaperRow,
) -> DiscoveryItem:
    markdown_path = (
        fulltext_root
        / paper.canonical_source
        / paper.paper_uid
        / f"{paper.paper_uid}.md"
    )
    if not markdown_path.is_file():
        return DiscoveryItem(
            paper=paper,
            markdown_path=markdown_path,
            result=None,
            latency_seconds=0.0,
            local_error=f"Markdown file not found: {markdown_path}",
            markdown_missing=True,
        )

    started = time.monotonic()
    async with semaphore:
        try:
            full_text = await asyncio.to_thread(
                markdown_path.read_text,
                encoding="utf-8",
            )
            result = await discovery_llm.aextract(full_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = None
            local_error = f"{type(exc).__name__}: {exc}"
        else:
            local_error = None

    return DiscoveryItem(
        paper=paper,
        markdown_path=markdown_path,
        result=result,
        latency_seconds=time.monotonic() - started,
        local_error=local_error,
    )


def token_summary(result: PropertyExtractionResult) -> str:
    usage = result.metadata.usage
    if usage is None:
        return "tokens=unavailable"
    return (
        f"tokens_input={usage.prompt_tokens} "
        f"tokens_output={usage.completion_tokens} "
        f"tokens_total={usage.total_tokens}"
    )


def log_completed_item(item: DiscoveryItem) -> None:
    paper = item.paper
    prefix = f"source={paper.canonical_source} paper_uid={paper.paper_uid}"
    if item.result is not None:
        LOG.info(
            "%s properties=%d %s latency=%.2fs",
            prefix,
            len(item.result.properties),
            token_summary(item.result),
            item.latency_seconds,
        )
    else:
        LOG.error(
            "%s discovery_failure=%s latency=%.2fs",
            prefix,
            item.local_error or "unknown error",
            item.latency_seconds,
        )


def add_usage(progress: Progress, result: PropertyExtractionResult) -> None:
    usage = result.metadata.usage
    if usage is None:
        return
    progress.prompt_tokens += usage.prompt_tokens or 0
    progress.completion_tokens += usage.completion_tokens or 0
    progress.total_tokens += usage.total_tokens or 0


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
    items: list[DiscoveryItem],
    progress: Progress,
    sqlite_write_retries: int,
    sqlite_retry_base_delay: float,
    sqlite_retry_max_delay: float,
) -> None:
    """Commit successful JSON arrays, matching both paper ID and UID."""
    progress.attempted += len(items)
    candidates: list[tuple[DiscoveryItem, str]] = []
    for item in items:
        result = item.result
        if result is not None:
            payload = json.dumps(
                result.properties,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            candidates.append((item, payload))
            add_usage(progress, result)
            continue

        progress.discovery_errors += 1
        if item.markdown_missing:
            progress.missing_markdown += 1

    if not candidates:
        return

    column_sql = quote_identifier(RESULT_COLUMN)
    update_sql = f"""
        UPDATE {quote_identifier(TABLE)}
        SET {column_sql} = ?
        WHERE paper_id = ?
          AND paper_uid = ?
          AND {column_sql} IS NULL
    """
    applied: list[tuple[DiscoveryItem, str]] = []
    skipped: list[DiscoveryItem] = []
    for attempt in range(sqlite_write_retries + 1):
        applied = []
        skipped = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            for item, payload in candidates:
                cursor = connection.execute(
                    update_sql,
                    (payload, item.paper.paper_id, item.paper.paper_uid),
                )
                if cursor.rowcount == 1:
                    applied.append((item, payload))
                else:
                    skipped.append(item)
            connection.commit()
            break
        except sqlite3.Error as exc:
            connection.rollback()
            should_retry = (
                retryable_sqlite_lock_error(exc)
                and attempt < sqlite_write_retries
            )
            if should_retry:
                exponential_delay = min(
                    sqlite_retry_max_delay,
                    sqlite_retry_base_delay * (2**attempt),
                )
                delay = exponential_delay * random.uniform(0.8, 1.2)
                LOG.warning(
                    "SQLite write lock for %d results; retry=%d/%d "
                    "delay=%.2fs error=%s",
                    len(candidates),
                    attempt + 1,
                    sqlite_write_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue

            progress.database_errors += len(candidates)
            LOG.exception(
                "SQLite transaction failed for %d property results after %d "
                "attempt(s); the entire batch was rolled back",
                len(candidates),
                attempt + 1,
            )
            return

    progress.updated += len(applied)
    progress.properties_returned += sum(
        len(item.result.properties)
        for item, _ in applied
        if item.result is not None
    )
    progress.database_errors += len(skipped)
    for item in skipped:
        LOG.error(
            "Database update skipped paper_id=%d paper_uid=%s; the UID or "
            "result column may have changed concurrently",
            item.paper.paper_id,
            item.paper.paper_uid,
        )


def log_progress(progress: Progress, target: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    LOG.info(
        "Progress attempted=%d/%d updated=%d properties=%d "
        "discovery_errors=%d missing_markdown=%d database_errors=%d "
        "tokens_input=%d tokens_output=%d tokens_total=%d rate=%.2f papers/s",
        progress.attempted,
        target,
        progress.updated,
        progress.properties_returned,
        progress.discovery_errors,
        progress.missing_markdown,
        progress.database_errors,
        progress.prompt_tokens,
        progress.completion_tokens,
        progress.total_tokens,
        progress.attempted / elapsed,
    )


async def run_with_discovery_llm(
    args: argparse.Namespace,
    db_path: Path,
    discovery_llm: PropertyDiscoveryLLM,
) -> int:
    fulltext_root = args.fulltext_root.expanduser().resolve()
    if not fulltext_root.is_dir():
        LOG.error("Full-text root does not exist: %s", fulltext_root)
        return 1

    try:
        uids = load_uid_file(args.uid_file)
    except (OSError, ValueError, json.JSONDecodeError):
        LOG.exception("Could not load UID manifest: %s", args.uid_file)
        return 1

    progress = Progress(requested=len(uids))
    connection = sqlite3.connect(
        db_path,
        timeout=args.sqlite_timeout,
        isolation_level=None,
    )
    connection.execute(
        f"PRAGMA busy_timeout = {int(args.sqlite_timeout * 1000)}"
    )
    started = time.monotonic()

    try:
        validate_discovery_table(connection)
        pending, already_completed = load_manifest_rows(connection, uids)
        progress.already_completed = already_completed
        if args.limit is not None:
            pending = pending[: args.limit]
        target = len(pending)
        LOG.info(
            "Manifest requested=%d already_completed=%d pending_target=%d "
            "database=%s table=%s result_column=%s fulltext_root=%s",
            progress.requested,
            progress.already_completed,
            target,
            db_path,
            TABLE,
            RESULT_COLUMN,
            fulltext_root,
        )
        if target == 0:
            LOG.info("No pending manifest UIDs to process")
            return 0

        config = discovery_llm.llm_config()
        LOG.info(
            "Discovery model=%s base_url=%s timeout=%s max_tokens=%s "
            "prompt_hash=%s extra_body=%s max_parallel_requests=%d batch_size=%d",
            config.get("model"),
            config.get("base_url"),
            config.get("timeout"),
            config.get("max_tokens"),
            config.get("prompt_hash"),
            config.get("extra_body"),
            args.max_parallel_requests,
            args.batch_size,
        )
        try:
            discovery_llm.health_check()
        except Exception:
            LOG.exception("Model endpoint health check failed")
            return 1

        semaphore = asyncio.Semaphore(args.max_parallel_requests)
        processed = 0
        for batch_number, batch in enumerate(
            chunks(pending, args.batch_size),
            start=1,
        ):
            LOG.info(
                "Starting batch=%d rows=%d progress=%d/%d",
                batch_number,
                len(batch),
                processed,
                target,
            )
            items = await asyncio.gather(
                *(
                    discover_paper(
                        discovery_llm=discovery_llm,
                        semaphore=semaphore,
                        fulltext_root=fulltext_root,
                        paper=paper,
                    )
                    for paper in batch
                )
            )
            for item in items:
                log_completed_item(item)
            write_result_batch(
                connection=connection,
                items=items,
                progress=progress,
                sqlite_write_retries=args.sqlite_write_retries,
                sqlite_retry_base_delay=args.sqlite_retry_base_delay,
                sqlite_retry_max_delay=args.sqlite_retry_max_delay,
            )
            processed += len(batch)
            log_progress(progress, target, started)

    except asyncio.CancelledError:
        LOG.warning("Cancelled; previously committed results are preserved")
        raise
    except (RuntimeError, sqlite3.Error):
        LOG.exception("Fatal database or manifest validation error")
        return 1
    finally:
        connection.close()

    LOG.info(
        "Finished requested=%d already_completed=%d attempted=%d updated=%d "
        "properties=%d; failed rows remain NULL and can be retried",
        progress.requested,
        progress.already_completed,
        progress.attempted,
        progress.updated,
        progress.properties_returned,
    )
    if progress.discovery_errors or progress.database_errors:
        LOG.error(
            "Job incomplete: discovery_errors=%d database_errors=%d; rerun "
            "the same manifest to retry remaining NULL rows",
            progress.discovery_errors,
            progress.database_errors,
        )
        return 2
    return 0


async def run_async(args: argparse.Namespace) -> int:
    db_path = args.db_path.expanduser().resolve()
    if not db_path.is_file():
        LOG.error("Database does not exist: %s", db_path)
        return 1

    backend = LLMBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        extra_body=args.extra_body,
    )
    try:
        discovery_llm = PropertyDiscoveryLLM(
            backend=backend,
            system_prompt_path=args.system_prompt_path,
            user_prompt_path=args.user_template_path,
        )
        return await run_with_discovery_llm(args, db_path, discovery_llm)
    finally:
        try:
            backend.close()
        finally:
            await backend.aclose()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        return asyncio.run(run_async(args))
    except KeyboardInterrupt:
        LOG.warning("Interrupted; previously committed results are preserved")
        return 130
    except Exception:
        LOG.exception("Fatal unhandled error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
