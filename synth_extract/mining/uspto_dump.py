#!/usr/bin/env python3
"""Download the USPTO PatentsView bulk datasets used by this project.

Files are streamed into ``data/uspto/dump`` by default. Completed files are
skipped, while new downloads are written to ``.part`` files and atomically
renamed only after the response finishes.

The USPTO Open Data Portal requires an API key in ``USPTO_API_KEY``. It may
also be supplied with ``--api-key``.

Usage
-----
    python -m synth_extract.mining.uspto_dump
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

PRODUCT_METADATA_URL = "https://api.uspto.gov/api/v1/datasets/products/{product}"
DEFAULT_OUTPUT_DIR = Path("data/uspto/dump")
CHUNK_SIZE_BYTES = 8 * 1024 * 1024
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 600

logger = logging.getLogger("uspto_dump")


@dataclass(frozen=True)
class DatasetFile:
    product: str
    filename: str
    url: str


FIXED_FILES: dict[str, tuple[str, ...]] = {
    "PVGPATDIS": (
        "g_patent.tsv.zip",
        "g_patent_abstract.tsv.zip",
        "g_application.tsv.zip",
    ),
    "PVPGPUBDIS": (
        "pg_published_application.tsv.zip",
        "pg_published_application_abstract.tsv.zip",
        "pg_granted_pgpubs_crosswalk.tsv.zip",
        "pg_cpc_current.tsv.zip",
    ),
}

ANNUAL_FILE_PATTERNS: dict[str, re.Pattern[str]] = {
    "PVPGPUBTXT": re.compile(
        r"^pg_(?:detail_desc_text|claims|brf_sum_text)_\d{4}\.tsv\.zip$"
    ),
    "PVGPATTXT": re.compile(
        r"^g_(?:detail_desc_text|claims|brf_sum_text)_\d{4}\.tsv\.zip$"
    ),
}


def setup_logging(level: str = "INFO") -> None:
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
    logging.getLogger("urllib3").setLevel(max(logging.WARNING, root.level))


def build_manifest(session: requests.Session) -> list[DatasetFile]:
    """Discover and select requested files from each product's metadata."""
    files: list[DatasetFile] = []
    products = [*FIXED_FILES, *ANNUAL_FILE_PATTERNS]

    for product in products:
        response = session.get(
            PRODUCT_METADATA_URL.format(product=product),
            headers={"Accept": "application/json"},
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        payload = response.json()

        product_bag = payload.get("bulkDataProductBag", [])
        if not product_bag:
            raise ValueError(f"Metadata for {product} contains no product")
        file_bag = product_bag[0].get("productFileBag", {})
        metadata_files = file_bag.get("fileDataBag", [])

        fixed_names = set(FIXED_FILES.get(product, ()))
        annual_pattern = ANNUAL_FILE_PATTERNS.get(product)
        selected_names: set[str] = set()

        for entry in metadata_files:
            filename = entry.get("fileName")
            url = entry.get("fileDownloadURI")
            if not filename or not url:
                continue
            if filename in fixed_names or (
                annual_pattern is not None
                and annual_pattern.fullmatch(filename)
            ):
                files.append(DatasetFile(product, filename, url))
                selected_names.add(filename)

        missing = fixed_names - selected_names
        if missing:
            raise ValueError(
                f"Metadata for {product} is missing requested file(s): "
                f"{', '.join(sorted(missing))}"
            )

    return files


def download_file(
    session: requests.Session,
    dataset_file: DatasetFile,
    output_dir: Path,
) -> str:
    output_path = output_dir / dataset_file.filename
    part_path = output_path.with_name(output_path.name + ".part")

    if output_path.is_file() and output_path.stat().st_size > 0:
        logger.info(
            "SKIP | %s | %s already exists (%s bytes)",
            dataset_file.product,
            output_path,
            output_path.stat().st_size,
        )
        return "skipped"

    try:
        part_path.unlink(missing_ok=True)
        logger.info(
            "START | %s | %s", dataset_file.product, dataset_file.filename
        )

        with session.get(
            dataset_file.url,
            stream=True,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            downloaded = 0

            with part_path.open("wb") as output_file:
                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE_BYTES
                ):
                    if not chunk:
                        continue
                    output_file.write(chunk)
                    downloaded += len(chunk)

        if downloaded == 0:
            raise OSError("USPTO response body was empty")

        os.replace(part_path, output_path)
    except (requests.RequestException, OSError) as exc:
        logger.error(
            "FAILURE | %s | %s | %s",
            dataset_file.product,
            dataset_file.filename,
            exc,
        )
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass
        return "failed"

    logger.info(
        "SUCCESS | %s | %s | %s bytes | %s",
        dataset_file.product,
        dataset_file.filename,
        downloaded,
        output_path,
    )
    return "success"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected USPTO PatentsView bulk datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Download directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="USPTO API key. Defaults to USPTO_API_KEY.",
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
    load_dotenv()

    api_key = args.api_key or os.environ.get("USPTO_API_KEY")
    if not api_key:
        logger.error(
            "No USPTO API key provided "
            "(use --api-key or set USPTO_API_KEY)."
        )
        return 2

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "Could not create output directory %s: %s",
            args.output_dir,
            exc,
        )
        return 2

    counts = {"success": 0, "failed": 0, "skipped": 0}
    with requests.Session() as session:
        session.headers.update(
            {
                "X-API-KEY": api_key,
                "User-Agent": "synth-extract-uspto-dump/1.0",
            }
        )
        try:
            manifest = build_manifest(session)
        except (requests.RequestException, ValueError) as exc:
            logger.error("Could not build manifest from USPTO metadata: %s", exc)
            return 2

        logger.info(
            "Discovered %d requested file(s) for download into %s.",
            len(manifest),
            args.output_dir,
        )

        for index, dataset_file in enumerate(manifest, start=1):
            logger.info(
                "[%d/%d] %s/%s",
                index,
                len(manifest),
                dataset_file.product,
                dataset_file.filename,
            )
            outcome = download_file(
                session, dataset_file, args.output_dir
            )
            counts[outcome] += 1

    logger.info(
        "Done | success=%d skipped=%d failed=%d",
        counts["success"],
        counts["skipped"],
        counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
