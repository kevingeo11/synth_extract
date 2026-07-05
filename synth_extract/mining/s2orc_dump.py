"""
s2orc_dump.py
=============

Downloader for the Semantic Scholar Academic Graph "s2orc_v2" dataset.

Dataset: s2orc_v2
------------------
Full-body paper text parsed from open-access PDFs. Identifies structural
elements such as sentences and paragraphs, and bibliographic references.
~16M records spread across ~30 shard files (~6 GB each, gzip-compressed).

The "s2orc_v2" dataset contains parsed full-body text from selected papers.
It is a replacement for the legacy "s2orc" dataset. The body text is parsed
from PDF documents using Grobid (https://grobid.readthedocs.io). Grobid's
XML output is converted into a "body" string and a "bibliography" string,
each with its own set of annotation spans. See
https://github.com/allenai/s2orc for further details.

Schema
------
- openaccessinfo:
    - externalIds : IDs of this paper in different catalogs
    - license     : License information from Unpaywall, linked by DOI or
                    PubMed Central ID
    - url         : URL to the paper, if available
    - status      : Open-access status of the paper, if available
    - disclaimer  : Human readable description of the copyright status
                    of the paper
- title
- authors
- body:
    - text        : Full body text as a single string
    - annotations : Annotated spans of the full body text
- bibliography:
    - text        : Full bibliography text as a single string
    - annotations : Annotated spans of the full bibliography text

License
-------
This collection is licensed under ODC-BY
(https://opendatacommons.org/licenses/by/1.0/). By downloading this data
you acknowledge that you have read and agreed to all the terms of this
license.

Attribution
-----------
When using this data in a product or service, or including it in a
redistribution, please cite:

    @misc{https://doi.org/10.48550/arxiv.2301.10140,
      title     = {The Semantic Scholar Open Data Platform},
      author    = {Kinney, Rodney and Anastasiades, Chloe and Authur, Russell
                   and Beltagy, Iz and Bragg, Jonathan and Buraczynski, Alexandra
                   and Cachola, Isabel and Candra, Stefan
                   and Chandrasekhar, Yoganand and Cohan, Arman
                   and Crawford, Miles and Downey, Doug and Dunkelberger, Jason
                   and Etzioni, Oren and Evans, Rob and Feldman, Sergey
                   and Gorney, Joseph and Graham, David and Hu, Fangzhou
                   and Huff, Regan and King, Daniel and Kohlmeier, Sebastian
                   and Kuehl, Bailey and Langan, Michael and Lin, Daniel
                   and Liu, Haokun and Lo, Kyle and Lochner, Jaron
                   and MacMillan, Kelsey and Murray, Tyler and Newell, Chris
                   and Rao, Smita and Rohatgi, Shaurya and Sayre, Paul
                   and Shen, Zejiang and Singh, Amanpreet and Soldaini, Luca
                   and Subramanian, Shivashankar and Tanaka, Amber
                   and Wade, Alex D. and Wagner, Linda and Wang, Lucy Lu
                   and Wilhelm, Chris and Wu, Caroline and Yang, Jiangjiang
                   and Zamarron, Angele and Van Zuylen, Madeleine
                   and Weld, Daniel S.},
      publisher = {arXiv},
      year      = {2023},
      doi       = {10.48550/ARXIV.2301.10140},
      url       = {https://arxiv.org/abs/2301.10140},
    }

Usage
-----
    export S2_API_KEY=<your api key>
    python s2orc_dump.py

Behavior
--------
- get_shard_urls() queries the API for the latest release, then the
  pre-signed shard URLs for s2orc_v2 within that release, and writes them
  to data/s2orc/shard_urls.json (a fresh manifest each run, since the S3
  links are pre-signed and expire after a while).
- download_shards() reads that manifest back from disk and downloads each
  shard into data/s2orc, with:
    - skip-if-already-downloaded (verified via gzip integrity check),
    - resume of interrupted downloads via HTTP Range requests (partial
      files kept as "<shard>.gz.part" until confirmed complete),
    - retries with exponential backoff on transient network failures,
    - size verification against the server-reported Content-Length.
- Progress is logged (timestamped) to both stdout and a persistent log
  file in the download directory, so it survives across Slurm job
  resubmissions and is readable from the .out file.

Requires: requests, tqdm
"""

import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

API_KEY = os.getenv("S2_API_KEY")
DATASET_NAME = "s2orc_v2"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "s2orc"
SHARD_URLS_PATH = DATA_DIR / "shard_urls.json"
MAX_RETRIES = 5
CHUNK_SIZE = 1024 * 1024  # 1 MB
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300
PROGRESS_LOG_INTERVAL_PCT = 10  # log every N% of a shard's download

logger = logging.getLogger("s2orc_dump")


def _setup_logging():
    """Log to stdout (captured by Slurm's .out file) and to a persistent
    file in DATA_DIR so history survives across job resubmissions."""
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    fmt = f"%(asctime)s [job={job_id}] %(levelname)s %(message)s"
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(stream_handler)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(DATA_DIR / "download.log", mode="a")
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)


def get_shard_urls(session):
    """Look up the latest release, fetch the pre-signed s2orc_v2 shard
    URLs for it, persist them to SHARD_URLS_PATH, and return the list."""
    resp = session.get(
        "https://api.semanticscholar.org/datasets/v1/release",
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    resp.raise_for_status()
    releases = resp.json()
    if not releases:
        logger.error("No releases returned by the API.")
        sys.exit(1)
    latest = releases[-1]
    logger.info("Latest release: %s", latest)

    resp = session.get(
        f"https://api.semanticscholar.org/datasets/v1/release/{latest}/dataset/{DATASET_NAME}",
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    resp.raise_for_status()
    dataset = resp.json()
    files = dataset.get("files", [])
    if not files:
        logger.error("No files returned for dataset '%s'.", DATASET_NAME)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SHARD_URLS_PATH, "w") as f:
        json.dump({"release": latest, "dataset": DATASET_NAME, "files": files}, f, indent=2)

    logger.info("Found %d shard(s); manifest written to %s", len(files), SHARD_URLS_PATH)
    return files


def download_shards(session):
    """Read the shard URL manifest from SHARD_URLS_PATH and download every
    shard into DATA_DIR, with resume, retry, and integrity verification."""
    if not SHARD_URLS_PATH.exists():
        logger.error("No manifest found at %s; run get_shard_urls() first.", SHARD_URLS_PATH)
        sys.exit(1)

    with open(SHARD_URLS_PATH) as f:
        manifest = json.load(f)
    files = manifest["files"]
    logger.info("Loaded %d shard URL(s) from %s", len(files), SHARD_URLS_PATH)

    for idx, url in enumerate(tqdm(files, desc="Shards", unit="file"), start=1):
        shard_name = Path(urlparse(url).path).name  # pre-signed URLs carry query params, keep only the path
        if not shard_name.endswith(".gz"):
            shard_name += ".gz"
        dest_path = DATA_DIR / shard_name
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        logger.info("[%d/%d] Processing %s", idx, len(files), shard_name)

        if dest_path.exists():
            try:
                with gzip.open(dest_path, "rb") as f:
                    while f.read(CHUNK_SIZE):
                        pass
                logger.info("[%d/%d] %s already downloaded and verified intact; skipping.", idx, len(files), shard_name)
                continue
            except (OSError, EOFError) as exc:
                logger.warning("[%d/%d] %s exists but failed gzip integrity check (%s); re-downloading.", idx, len(files), shard_name, exc)
                dest_path.unlink()

        resume_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
        if resume_bytes:
            logger.info("[%d/%d] Resuming %s from byte %d", idx, len(files), shard_name, resume_bytes)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                req_headers = {}
                mode = "wb"
                if resume_bytes:
                    req_headers["Range"] = f"bytes={resume_bytes}-"
                    mode = "ab"

                with session.get(
                    url,
                    headers=req_headers,
                    stream=True,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                ) as r:
                    if r.status_code == 416:
                        # Range not satisfiable. This *usually* means our .part file
                        # is already complete, but it can also mean the local file is
                        # stale/oversized relative to the remote object. Confirm via
                        # the Content-Range header (format "bytes */<total_size>")
                        # before trusting it.
                        content_range = r.headers.get("Content-Range", "")
                        remote_total = None
                        if "/" in content_range:
                            try:
                                remote_total = int(content_range.rsplit("/", 1)[-1])
                            except ValueError:
                                remote_total = None

                        local_size = tmp_path.stat().st_size
                        if remote_total is not None and local_size == remote_total:
                            tmp_path.rename(dest_path)
                            logger.info("[%d/%d] %s confirmed complete via 416/Content-Range.", idx, len(files), shard_name)
                            break

                        logger.warning(
                            "[%d/%d] %s: got 416 but local size (%s) doesn't match remote size (%s); "
                            "discarding partial file and retrying.",
                            idx, len(files), shard_name, local_size, remote_total,
                        )
                        tmp_path.unlink()
                        resume_bytes = 0
                        raise IOError("stale or mismatched partial file after 416 response")
                    r.raise_for_status()

                    expected_total = int(r.headers.get("Content-Length", 0)) + resume_bytes

                    downloaded = resume_bytes
                    last_logged_pct = -1
                    with open(tmp_path, mode) as out_f, tqdm(
                        total=expected_total or None,
                        initial=resume_bytes,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=shard_name,
                        leave=False,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                out_f.write(chunk)
                                bar.update(len(chunk))
                                downloaded += len(chunk)
                                if expected_total:
                                    pct = int(downloaded * 100 / expected_total)
                                    if pct >= last_logged_pct + PROGRESS_LOG_INTERVAL_PCT:
                                        logger.info(
                                            "[%d/%d] %s: %d%% (%.2f / %.2f GB)",
                                            idx, len(files), shard_name, pct,
                                            downloaded / 1e9, expected_total / 1e9,
                                        )
                                        last_logged_pct = pct

                actual_size = tmp_path.stat().st_size
                if expected_total and actual_size != expected_total:
                    raise IOError(
                        f"size mismatch for {shard_name}: expected {expected_total} bytes, "
                        f"got {actual_size} bytes"
                    )

                tmp_path.rename(dest_path)
                logger.info("[%d/%d] %s downloaded successfully (%.2f GB).", idx, len(files), shard_name, actual_size / 1e9)
                break  # success, move on to the next shard

            except (requests.RequestException, IOError) as exc:
                resume_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
                if attempt == MAX_RETRIES:
                    logger.error("[%d/%d] %s failed after %d attempts: %s", idx, len(files), shard_name, MAX_RETRIES, exc)
                    sys.exit(1)
                wait = min(60, 2 ** attempt)
                logger.warning(
                    "[%d/%d] %s attempt %d/%d failed: %s. Retrying in %ds...",
                    idx, len(files), shard_name, attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

    logger.info("Done. All shards downloaded successfully.")


def main():
    _setup_logging()

    if not API_KEY:
        logger.error("Environment variable S2_API_KEY is not set.")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({"x-api-key": API_KEY})

    logger.info("Starting s2orc_v2 download. Target directory: %s", DATA_DIR)
    get_shard_urls(session)
    download_shards(session)


if __name__ == "__main__":
    main()