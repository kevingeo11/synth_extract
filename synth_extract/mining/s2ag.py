import gzip
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from glob import glob


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure stdout logging."""
    logger = logging.getLogger("build_abstracts_db")
    logger.setLevel(log_level)
    logger.propagate = False

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
    scratch_dir: str | None = None,
    batch_size: int = 200_000,
    skip_empty_abstracts: bool = True,
    log_every_seconds: float = 60.0,
    cache_size_kib: int = 2_000_000,   # ~2GB
    mmap_size_bytes: int = 4_000_000_000,  # ~4GB
) -> None:
    """
    Create a SQLite database containing corpusid -> abstract mappings from a
    directory of gzipped JSONL Semantic Scholar abstract dump files.

    Key differences from the original version:
      * Always a fresh build; raises if db_path already exists.
      * Loads into a plain (unindexed) rowid table first -- pure appends,
        no B-tree lookups on corpusid during the hot loop -- then builds a
        UNIQUE index on corpusid afterwards in one efficient sorted pass.
        This avoids the "random insert into a giant B-tree" slowdown that
        gets worse as the table grows.
      * Optionally builds the DB on fast local scratch storage (e.g. Slurm
        node-local disk / $TMPDIR) and copies the finished file to db_path
        at the end, since random writes over a network filesystem are the
        usual bottleneck on HPC.
      * Logs on a wall-clock heartbeat (every log_every_seconds) in
        addition to per-file boundaries, so you always see progress even
        if a batch takes a long time to fill or a file has few abstracts.

    Parameters
    ----------
    dump_dir : str
        Directory containing the *.gz JSONL files.
    db_path : str
        Final path for the SQLite DB file.
    scratch_dir : str, optional
        If given, the DB is built at os.path.join(scratch_dir, basename(db_path))
        and moved to db_path only at the end. Use node-local scratch
        (e.g. os.environ.get("TMPDIR")) for a big speedup on network filesystems.
    batch_size : int
        Rows accumulated before each executemany/commit.
    skip_empty_abstracts : bool
        If True, rows with null/empty abstract are skipped entirely.
    log_every_seconds : float
        Minimum wall-clock time between progress log lines.
    cache_size_kib : int
        SQLite page cache size in KiB (passed as a negative value to PRAGMA cache_size).
    mmap_size_bytes : int
        SQLite PRAGMA mmap_size, in bytes.
    """
    logger = setup_logging()

    if os.path.exists(db_path):
        raise FileExistsError(
            f"Database already exists: {db_path}. This script always does a fresh build."
        )

    gz_files = sorted(glob(os.path.join(dump_dir, "*.gz")))
    if not gz_files:
        logger.error(f"No .gz files found in {dump_dir}")
        raise FileNotFoundError(f"No .gz files found in {dump_dir}")

    build_path = db_path
    using_scratch = False
    if scratch_dir:
        os.makedirs(scratch_dir, exist_ok=True)
        build_path = os.path.join(scratch_dir, os.path.basename(db_path))
        if os.path.exists(build_path):
            raise FileExistsError(f"Scratch DB already exists: {build_path}")
        using_scratch = True

    logger.info(f"Found {len(gz_files)} .gz files in {dump_dir}")
    logger.info(f"Building at: {build_path}" + (" (scratch)" if using_scratch else ""))
    logger.info(f"Final destination: {db_path}")
    logger.info(
        f"Config: batch_size={batch_size:,} "
        f"skip_empty_abstracts={skip_empty_abstracts} "
        f"log_every_seconds={log_every_seconds} "
        f"cache_size_kib={cache_size_kib:,} mmap_size_bytes={mmap_size_bytes:,}"
    )

    conn = sqlite3.connect(build_path)
    cur = conn.cursor()

    # Pragmas tuned for a big one-shot bulk load.
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute(f"PRAGMA cache_size = -{cache_size_kib};")
    cur.execute(f"PRAGMA mmap_size = {mmap_size_bytes};")
    cur.execute("PRAGMA locking_mode = EXCLUSIVE;")

    # NOTE: no PRIMARY KEY / index on corpusid yet. Plain rowid table so
    # every insert is a pure append -- no B-tree lookups on corpusid.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS abstracts_raw (
            corpusid INTEGER,
            abstract TEXT
        );
        """
    )
    conn.commit()

    insert_sql = "INSERT INTO abstracts_raw (corpusid, abstract) VALUES (?, ?)"

    total_rows = 0
    total_skipped = 0
    total_bad_lines = 0
    total_lines = 0
    batch = []
    start_time = time.time()
    last_log_time = start_time

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
                    total_lines += 1
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

                    now = time.time()
                    if now - last_log_time >= log_every_seconds:
                        elapsed = now - start_time
                        rate = total_rows / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"{total_rows:,} rows inserted | "
                            f"{total_lines:,} lines read | "
                            f"{rate:,.0f} rows/sec | "
                            f"{elapsed:,.0f}s elapsed | "
                            f"skipped={total_skipped:,} bad={total_bad_lines:,} | "
                            f"file [{file_idx}/{len(gz_files)}]"
                        )
                        last_log_time = now

            logger.info(
                f"[{file_idx}/{len(gz_files)}] Finished {os.path.basename(path)} "
                f"in {time.time() - file_start:,.0f}s"
            )

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

    # Now build the real, indexed table in one efficient sorted pass,
    # instead of having maintained the index incrementally during load.
    logger.info("Building final indexed table (corpusid PRIMARY KEY)...")
    t0 = time.time()
    cur.execute(
        """
        CREATE TABLE abstracts (
            corpusid INTEGER PRIMARY KEY,
            abstract TEXT
        );
        """
    )
    cur.execute(
        "INSERT INTO abstracts (corpusid, abstract) "
        "SELECT corpusid, abstract FROM abstracts_raw ORDER BY corpusid;"
    )
    conn.commit()
    cur.execute("DROP TABLE abstracts_raw;")
    conn.commit()
    logger.info(f"Final table built in {time.time() - t0:,.0f}s")

    # Restore safer pragmas for normal future use.
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA locking_mode = NORMAL;")
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

    if using_scratch:
        logger.info(f"Copying finished DB from scratch to final destination: {db_path}")
        t0 = time.time()
        shutil.move(build_path, db_path)
        logger.info(f"Copy done in {time.time() - t0:,.0f}s")

    logger.info(f"Database ready at: {db_path}")


if __name__ == "__main__":
    dump_dir = "data/s2ag/abstracts"
    db_path = "data/s2ag/abstracts.db"
    # Use node-local scratch if Slurm/the cluster provides one.
    scratch_dir = os.environ.get("TMPDIR") or os.environ.get("SNIC_TMP")

    build_abstracts_db(dump_dir=dump_dir, db_path=db_path, scratch_dir=scratch_dir)