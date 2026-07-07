"""
papers_dump.py
===============

Downloader for the Semantic Scholar Academic Graph "papers" dataset.

Dataset: papers
---------------
Core bibliographic metadata for every paper in the Semantic Scholar corpus
(paper id, external ids, title, abstract, venue, year, citation/reference
counts, fields of study, authors, etc). This is metadata only -- it does
not include full body text (see the "s2orc_v2" dataset for that).

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
    python papers_dump.py

Set LATEST_RELEASE below to a specific release id (e.g. "2024-01-01") to
skip the "current latest release" API lookup and pin a specific release.
Leave it as None to auto-fetch whatever the API reports as latest.

Behavior
--------
- download_shards(release, data_dir) resolves the release (using the
  passed-in value, or querying the API for the latest one if None), fetches
  the pre-signed shard URLs for the "papers" dataset for that release,
  writes them to <data_dir>/shard_urls.json, and downloads each shard into
  data_dir, with:
    - skip-if-already-downloaded (verified via gzip integrity check),
    - resume of interrupted downloads via HTTP Range requests (partial
      files kept as "<shard>.gz.part" until confirmed complete),
    - retries with exponential backoff on transient network failures,
    - size verification against the server-reported Content-Length.
- Progress (per-shard, and every PROGRESS_LOG_INTERVAL_PCT% within a shard)
  is logged via plain `logging` to stdout only -- no tqdm progress bars,
  since those render poorly inside a Slurm .out file. Since everything goes
  to stdout, it's already captured in the Slurm .out file across job
  resubmissions; there's no separate log file to keep track of.

Requires: requests
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

API_KEY = os.getenv("S2_API_KEY")
DATASET_NAME = "papers"
MAX_RETRIES = 5
CHUNK_SIZE = 1024 * 1024  # 1 MB
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300
PROGRESS_LOG_INTERVAL_PCT = 10  # log every N% of a shard's download

# Set this to a specific release id (e.g. "2024-01-01") to use that release
# directly. Leave as None to look up whatever the API reports as latest.
LATEST_RELEASE = "2026-06-24"

logger = logging.getLogger("papers_dump")


def _setup_logging():
    """Log to stdout only. Slurm captures stdout into the .out file, so
    this is what survives across job resubmissions -- no separate log
    file to manage."""
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    fmt = f"%(asctime)s [job={job_id}] %(levelname)s %(message)s"
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)


def get_latest_release(session):
    """Query the API for the latest release id."""
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
    return latest


def get_shard_urls(session, release, shard_urls_path):
    """Fetch the pre-signed 'papers' shard URLs for the given release and
    persist them to shard_urls_path."""
    resp = session.get(
        f"https://api.semanticscholar.org/datasets/v1/release/{release}/dataset/{DATASET_NAME}",
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    resp.raise_for_status()
    dataset = resp.json()
    files = dataset.get("files", [])
    if not files:
        logger.error("No files returned for dataset '%s'.", DATASET_NAME)
        sys.exit(1)

    with open(shard_urls_path, "w") as f:
        json.dump({"release": release, "dataset": DATASET_NAME, "files": files}, f, indent=2)

    logger.info("Found %d shard(s); manifest written to %s", len(files), shard_urls_path)
    return files


def download_shards(release, data_dir):
    """Resolve the release (using `release` if given, otherwise looking up
    the latest one), fetch its 'papers' shard manifest, and download every
    shard into `data_dir`, with resume, retry, and integrity verification."""
    if not API_KEY:
        logger.error("Environment variable S2_API_KEY is not set.")
        sys.exit(1)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    shard_urls_path = data_dir / "shard_urls.json"

    session = requests.Session()
    session.headers.update({"x-api-key": API_KEY})
    try:
        if release is None:
            release = get_latest_release(session)
        else:
            logger.info("Using pinned release: %s", release)

        files = get_shard_urls(session, release, shard_urls_path)
        logger.info("Starting download of %d shard(s) into %s", len(files), data_dir)

        for idx, url in enumerate(files, start=1):
            shard_name = Path(urlparse(url).path).name  # pre-signed URLs carry query params, keep only the path
            if not shard_name.endswith(".gz"):
                shard_name += ".gz"
            dest_path = data_dir / shard_name
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
                        with open(tmp_path, mode) as out_f:
                            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                                if chunk:
                                    out_f.write(chunk)
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
    finally:
        session.close()


def main():
    _setup_logging()
    DATA_DIR = "data/s2ag/papers"
    download_shards(LATEST_RELEASE, DATA_DIR)


if __name__ == "__main__":
    main()