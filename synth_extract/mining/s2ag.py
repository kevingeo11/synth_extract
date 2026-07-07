import gzip
import json
import logging
import os
import sqlite3
import sys
import time
from glob import glob


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure stdout logging."""
    logger = logging.getLogger("build_abstracts_db")
    logger.setLevel(log_level)
    logger.propagate = False

    # Avoid duplicate handlers if setup_logging() is called more than once
    # (e.g. interactive use, or re-importing in a notebook).
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized.")

    return logger


def build_abstracts_db(
    dump_dir: str,
    db_path: str,
    batch_size: int = 100_000,
    skip_empty_abstracts: bool = True,
    log_every_n_batches: int = 10,
    force: bool = False,
) -> None:
    """
    Create a SQLite database containing corpusid -> abstract
    mappings from a directory of gzipped JSONL Semantic Scholar abstract
    dump files.

    Parameters
    ----------
    dump_dir : str
        Directory containing the *.gz JSONL files.
    db_path : str
        Path to the SQLite DB file to create.
    batch_size : int
        Number of rows to accumulate before each executemany/commit.
        Larger = faster but more RAM.
    skip_empty_abstracts : bool
        If True, rows with null/empty abstract are not inserted at all
        (saves a lot of space, since a large fraction of S2 records have
        no public abstract).
    log_every_n_batches : int
        Emit a progress log line every N committed batches.
    force : bool
        If True, allow writing to an existing database. If False, raise an
        error when db_path already exists.
    """
    logger = setup_logging()

    if os.path.exists(db_path) and not force:
        raise FileExistsError(
            f"Database already exists: {db_path}. "
            "Set force=True to write into the existing database."
        )

    gz_files = sorted(glob(os.path.join(dump_dir, "*.gz")))
    if not gz_files:
        logger.error(f"No .gz files found in {dump_dir}")
        raise FileNotFoundError(f"No .gz files found in {dump_dir}")

    logger.info(f"Found {len(gz_files)} .gz files in {dump_dir}")
    logger.info(f"Output DB: {db_path}")
    logger.info(
        f"Config: batch_size={batch_size:,} "
        f"skip_empty_abstracts={skip_empty_abstracts} "
        f"log_every_n_batches={log_every_n_batches} "
        f"force={force}"
    )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Pragmas tuned for a big one-shot bulk load. NOT safe against a
    # power-loss mid-write, but fine for building a derived dataset that
    # can simply be rebuilt if interrupted.
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("PRAGMA cache_size = -200000;")  # ~200MB page cache

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS abstracts (
            corpusid INTEGER PRIMARY KEY,
            abstract TEXT
        );
        """
    )
    conn.commit()

    insert_sql = "INSERT OR REPLACE INTO abstracts (corpusid, abstract) VALUES (?, ?)"

    total_rows = 0
    total_skipped = 0
    total_bad_lines = 0
    batch = []
    batch_count = 0
    start_time = time.time()

    try:
        for file_idx, path in enumerate(gz_files, start=1):
            file_start = time.time()
            logger.info(
                f"[{file_idx}/{len(gz_files)}] Processing {os.path.basename(path)}"
            )
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        total_bad_lines += 1
                        continue

                    corpusid = rec.get("corpusid")
                    abstract = rec.get("abstract")

                    if corpusid is None:
                        total_bad_lines += 1
                        continue

                    if skip_empty_abstracts and not abstract:
                        total_skipped += 1
                        continue

                    batch.append((corpusid, abstract))

                    if len(batch) >= batch_size:
                        cur.executemany(insert_sql, batch)
                        conn.commit()
                        total_rows += len(batch)
                        batch.clear()
                        batch_count += 1

                        if batch_count % log_every_n_batches == 0:
                            elapsed = time.time() - start_time
                            rate = total_rows / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"{total_rows:,} rows inserted | "
                                f"{rate:,.0f} rows/sec | "
                                f"{elapsed:,.0f}s elapsed | "
                                f"skipped={total_skipped:,} bad={total_bad_lines:,}"
                            )

            logger.info(
                f"[{file_idx}/{len(gz_files)}] Finished {os.path.basename(path)} "
                f"in {time.time() - file_start:,.0f}s"
            )

        # Flush remaining rows
        if batch:
            cur.executemany(insert_sql, batch)
            conn.commit()
            total_rows += len(batch)

    except Exception:
        logger.exception("Fatal error while building the database")
        conn.close()
        raise

    elapsed = time.time() - start_time
    logger.info(
        f"Insert phase done. Inserted {total_rows:,} rows "
        f"(skipped {total_skipped:,} empty abstracts, "
        f"{total_bad_lines:,} malformed lines) in {elapsed:,.0f}s "
        f"({total_rows / elapsed:,.0f} rows/sec avg)."
    )

    # Restore safer pragmas for normal future use of the DB (read-only
    # queries don't care, but this avoids surprises if something writes
    # to it later).
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    conn.commit()

    logger.info("Running VACUUM...")
    t0 = time.time()
    cur.execute("VACUUM;")
    logger.info(f"VACUUM done in {time.time() - t0:,.0f}s")

    logger.info("Running ANALYZE...")
    t0 = time.time()
    cur.execute("ANALYZE;")
    conn.commit()
    logger.info(f"ANALYZE done in {time.time() - t0:,.0f}s")

    conn.close()
    logger.info(f"Database ready at: {db_path}")


if __name__ == "__main__":
    dump_dir = "data/s2ag/abstracts"
    db_path = "data/s2ag/abstracts.db"

    build_abstracts_db(dump_dir=dump_dir, db_path=db_path)
