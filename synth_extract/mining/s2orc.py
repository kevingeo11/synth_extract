from pathlib import Path
import gzip
import json
import logging
import os
import sqlite3
import sys
import time
from glob import glob

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

logger = logging.getLogger(__name__)



def extract_polymer_s2orc(input_dir, output_file, keyword="polymer"):
    """
    Scan all .gz S2ORC JSONL files in input_dir.
    Write records whose title or body text contains keyword to output_file.
    """

    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        raise FileExistsError(
            f"Output file already exists: {output_file}\n"
            "Delete it or specify a different output file."
        )

    total_files = 0
    total_entries = 0
    total_hits = 0

    keyword = keyword.lower()

    gz_files = sorted(input_dir.glob("*.gz"))

    logging.info(f"Found {len(gz_files)} gz files in {input_dir}")

    with gzip.open(output_file, "wt", encoding="utf-8") as out:
        for gz_file in gz_files:
            file_entries = 0
            file_hits = 0

            logging.info(f"Starting {gz_file.name}")

            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                for line in f:
                    file_entries += 1
                    total_entries += 1

                    paper = json.loads(line)

                    title = (paper.get("title") or "").lower()
                    body = (paper.get("body", {}).get("text") or "").lower()

                    if keyword in title or keyword in body:
                        file_hits += 1
                        total_hits += 1
                        out.write(json.dumps(paper, ensure_ascii=False) + "\n")

            total_files += 1

            logging.info(
                f"Done {gz_file.name} | "
                f"entries={file_entries:,} | "
                f"polymer_hits={file_hits:,}"
            )

    logging.info(
        f"Finished all files | "
        f"files={total_files:,} | "
        f"entries={total_entries:,} | "
        f"polymer_hits={total_hits:,} | "
        f"output={output_file}"
    )


def check_polymer_ids_have_abstracts(
    polymer_file,
    abstracts_dir,
    abstract_output_file,
    keyword="polymer",
):
    """Write S2AG abstract rows for polymer S2ORC IDs whose abstract has keyword."""
    polymer_file = Path(polymer_file)
    abstracts_dir = Path(abstracts_dir)
    abstract_output_file = Path(abstract_output_file)

    if not polymer_file.exists():
        raise FileNotFoundError(f"Polymer file does not exist: {polymer_file}")
    if not abstracts_dir.is_dir():
        raise NotADirectoryError(f"Abstracts directory does not exist: {abstracts_dir}")

    abstract_output_file.parent.mkdir(parents=True, exist_ok=True)

    if abstract_output_file.exists():
        raise FileExistsError(
            f"Output file already exists: {abstract_output_file}\n"
            "Delete it or specify a different output file."
        )

    keyword = keyword.lower().strip()
    if not keyword:
        raise ValueError("keyword must be a non-empty string")

    def parse_corpus_id(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    polymer_ids = set()
    polymer_rows = 0
    duplicate_polymer_ids = 0
    invalid_polymer_rows = 0

    logging.info("Loading polymer corpus IDs from %s", polymer_file)

    with gzip.open(polymer_file, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            polymer_rows += 1
            try:
                paper = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_polymer_rows += 1
                logging.warning(
                    "Skipping invalid JSON in %s at line %s: %s",
                    polymer_file.name,
                    line_number,
                    exc,
                )
                continue

            cid = parse_corpus_id(paper.get("corpusid"))
            if cid is None:
                invalid_polymer_rows += 1
                logging.warning(
                    "Skipping polymer record with missing/invalid corpusid: paperid=%s",
                    paper.get("paperid"),
                )
                continue

            if cid in polymer_ids:
                duplicate_polymer_ids += 1
            polymer_ids.add(cid)

    logging.info(
        "Loaded polymer IDs | rows=%s | unique_ids=%s | duplicates=%s | invalid_rows=%s",
        f"{polymer_rows:,}",
        f"{len(polymer_ids):,}",
        f"{duplicate_polymer_ids:,}",
        f"{invalid_polymer_rows:,}",
    )

    if not polymer_ids:
        raise ValueError("No corpus IDs found in polymer file")

    found_ids = set()
    ids_with_abstract = set()
    keyword_positive_ids = set()

    total_abstract_rows = 0
    total_invalid_json_rows = 0
    total_invalid_cid_rows = 0
    total_matching_cids = 0
    total_empty_abstracts = 0
    total_written = 0

    abstract_files = sorted(abstracts_dir.glob("*.gz"))
    if not abstract_files:
        raise FileNotFoundError(f"No .gz abstract files found in {abstracts_dir}")

    logging.info("Found %s abstract gz files in %s", f"{len(abstract_files):,}", abstracts_dir)
    logging.info("Writing matching polymer abstracts to %s", abstract_output_file)

    with gzip.open(abstract_output_file, "wt", encoding="utf-8") as out:
        for file_number, gz_file in enumerate(abstract_files, start=1):
            file_rows = 0
            file_invalid_json_rows = 0
            file_invalid_cid_rows = 0
            file_matching_cids = 0
            file_empty_abstracts = 0
            file_written = 0

            logging.info(
                "Starting abstract file %s/%s: %s",
                f"{file_number:,}",
                f"{len(abstract_files):,}",
                gz_file.name,
            )

            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    file_rows += 1
                    total_abstract_rows += 1

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        file_invalid_json_rows += 1
                        total_invalid_json_rows += 1
                        logging.warning(
                            "Skipping invalid JSON in %s at line %s: %s",
                            gz_file.name,
                            line_number,
                            exc,
                        )
                        continue

                    cid = parse_corpus_id(record.get("corpusid"))
                    if cid is None:
                        file_invalid_cid_rows += 1
                        total_invalid_cid_rows += 1
                        continue

                    if cid not in polymer_ids:
                        continue

                    found_ids.add(cid)
                    file_matching_cids += 1
                    total_matching_cids += 1

                    abstract = record.get("abstract")
                    if not isinstance(abstract, str) or not abstract.strip():
                        file_empty_abstracts += 1
                        total_empty_abstracts += 1
                        continue

                    ids_with_abstract.add(cid)

                    if keyword in abstract.lower():
                        keyword_positive_ids.add(cid)
                        file_written += 1
                        total_written += 1
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")

            logging.info(
                "Done %s | rows=%s | invalid_json=%s | invalid_corpusid=%s | "
                "polymer_cid_rows=%s | empty_abstracts=%s | keyword_hits=%s | "
                "unique_ids_found=%s/%s | unique_ids_with_abstract=%s",
                gz_file.name,
                f"{file_rows:,}",
                f"{file_invalid_json_rows:,}",
                f"{file_invalid_cid_rows:,}",
                f"{file_matching_cids:,}",
                f"{file_empty_abstracts:,}",
                f"{file_written:,}",
                f"{len(found_ids):,}",
                f"{len(polymer_ids):,}",
                f"{len(ids_with_abstract):,}",
            )

            out.flush()

            if len(ids_with_abstract) == len(polymer_ids):
                logging.info("All polymer IDs have non-empty abstracts. Stopping early.")
                break

    missing_ids = polymer_ids - found_ids
    missing_abstract_ids = polymer_ids - ids_with_abstract
    missing_keyword_ids = polymer_ids - keyword_positive_ids

    logging.info(
        "Finished abstract scan | rows=%s | invalid_json=%s | invalid_corpusid=%s | "
        "polymer_cid_rows=%s | empty_matching_abstracts=%s | written=%s",
        f"{total_abstract_rows:,}",
        f"{total_invalid_json_rows:,}",
        f"{total_invalid_cid_rows:,}",
        f"{total_matching_cids:,}",
        f"{total_empty_abstracts:,}",
        f"{total_written:,}",
    )
    logging.info(
        "Polymer ID coverage | unique_ids=%s | found_in_abstract_rows=%s | "
        "with_non_empty_abstract=%s | missing_from_abstract_rows=%s | "
        "missing_non_empty_abstract=%s",
        f"{len(polymer_ids):,}",
        f"{len(found_ids):,}",
        f"{len(ids_with_abstract):,}",
        f"{len(missing_ids):,}",
        f"{len(missing_abstract_ids):,}",
    )
    logging.info(
        "Keyword coverage | keyword=%r | matching_rows_written=%s | "
        "unique_keyword_positive_ids=%s | ids_without_keyword_positive_abstract=%s",
        keyword,
        f"{total_written:,}",
        f"{len(keyword_positive_ids):,}",
        f"{len(missing_keyword_ids):,}",
    )
    logging.info("Output file: %s", abstract_output_file)


def _chunked(seq, size):
    """Yield successive `size`-sized chunks from `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _fetch_abstracts(conn, corpusids, sql_chunk_size=500):
    """
    Given a list of corpusids, return a dict {corpusid: abstract} for the
    subset that exists in the abstracts DB.

    SQLite has a default limit of ~999 host parameters per query, so the
    lookup is done in chunks regardless of how large the caller's batch is.
    """
    result = {}
    if not corpusids:
        return result

    for chunk in _chunked(list(set(corpusids)), sql_chunk_size):
        placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT corpusid, abstract FROM abstracts WHERE corpusid IN ({placeholders})"
        for cid, abstract in conn.execute(query, chunk):
            result[cid] = abstract
    return result


def filter_s2orc_by_keyword(
    s2orc_dump_dir: str,
    abstract_db_path: str,
    output_path: str,
    keyword: str = "polymer",
    batch_size: int = 10000,
    sql_chunk_size: int = 15000,
    log_every_n_batches: int = 10,
) -> dict:
    """
    Read a directory of gzipped S2ORC JSONL files, join each paper against
    `abstract_db_path` by corpusid, keep papers where `keyword` appears in
    the title and/or the matched abstract, and write the survivors out as
    a single gzipped JSONL file.

    Parameters
    ----------
    s2orc_dump_dir : str
        Directory containing the S2ORC *.gz JSONL files.
    abstract_db_path : str
        Path to the SQLite DB built earlier (table `abstracts(corpusid, abstract)`).
    output_path : str
        Path to write the filtered output to, e.g. "polymer_papers.jsonl.gz".
        Raises FileExistsError if this path already exists, to avoid
        silently overwriting a previous run.
    keyword : str
        Keyword to search for, case-insensitive substring match. Default "polymer".
    batch_size : int
        Number of S2ORC records to buffer in memory before doing a batch
        lookup against the abstracts DB.
    sql_chunk_size : int
        Max number of corpusids per single SQL `IN (...)` query (kept
        comfortably under SQLite's default ~999 host-parameter limit).
    log_every_n_batches : int
        Emit a progress log line every N processed batches.
    """
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Output path already exists, refusing to overwrite: {output_path}"
        )

    gz_files = sorted(glob(os.path.join(s2orc_dump_dir, "*.gz")))
    if not gz_files:
        raise FileNotFoundError(f"No .gz files found in {s2orc_dump_dir}")

    if not os.path.exists(abstract_db_path):
        raise FileNotFoundError(f"Abstracts DB not found: {abstract_db_path}")

    keyword_lower = keyword.lower()

    logger.info(f"Found {len(gz_files)} S2ORC .gz files in {s2orc_dump_dir}")
    logger.info(f"Abstracts DB: {abstract_db_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Keyword (case-insensitive): '{keyword}'")
    logger.info(f"Config: batch_size={batch_size:,} sql_chunk_size={sql_chunk_size}")

    # Open the abstracts DB read-only so this job can never accidentally
    # mutate it, and so it's safe to run concurrently with other readers.
    db_uri = f"file:{os.path.abspath(abstract_db_path)}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    # For speed up
    conn.execute("PRAGMA mmap_size = 68719476736")   # ~64GB, covers the whole 58GB DB comfortably
    conn.execute("PRAGMA cache_size = -2000000")      # ~2GB SQLite-private cache, fine
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA query_only = TRUE")

    # Counters
    total_lines = 0
    bad_json_lines = 0
    total_papers = 0  # valid JSON lines with a corpusid
    discarded_no_abstract = 0
    matched_with_abstract = 0
    passed_filter = 0
    failed_filter = 0

    batch_count = 0
    start_time = time.time()

    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as out_f:
            for file_idx, path in enumerate(gz_files, start=1):
                file_start = time.time()
                logger.info(
                    f"[{file_idx}/{len(gz_files)}] Processing {os.path.basename(path)}"
                )

                batch = []  # list of (corpusid, record)

                def flush_batch():
                    nonlocal matched_with_abstract, discarded_no_abstract
                    nonlocal passed_filter, failed_filter, batch_count

                    if not batch:
                        return

                    corpusids = [cid for cid, _rec in batch]
                    abstract_map = _fetch_abstracts(conn, corpusids, sql_chunk_size)

                    for cid, rec in batch:
                        if cid not in abstract_map:
                            discarded_no_abstract += 1
                            continue

                        matched_with_abstract += 1
                        abstract_text = abstract_map[cid] or ""
                        title_text = rec.get("title") or ""

                        title_has_keyword = keyword_lower in title_text.lower()
                        abstract_has_keyword = keyword_lower in abstract_text.lower()

                        if title_has_keyword or abstract_has_keyword:
                            passed_filter += 1
                            rec_out = dict(rec)
                            rec_out["abstract"] = abstract_text
                            out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                        else:
                            failed_filter += 1

                    batch.clear()
                    batch_count += 1

                    if batch_count % log_every_n_batches == 0:
                        elapsed = time.time() - start_time
                        rate = total_papers / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"progress: {total_papers:,} papers seen | "
                            f"{matched_with_abstract:,} matched abstract | "
                            f"{passed_filter:,} passed filter | "
                            f"{rate:,.0f} papers/sec | {elapsed:,.0f}s elapsed"
                        )

                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        total_lines += 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            bad_json_lines += 1
                            continue

                        cid = rec.get("corpusid")
                        if cid is None:
                            bad_json_lines += 1
                            continue

                        total_papers += 1
                        batch.append((cid, rec))

                        if len(batch) >= batch_size:
                            flush_batch()

                # Flush any leftover records for this file
                flush_batch()

                logger.info(
                    f"[{file_idx}/{len(gz_files)}] Finished {os.path.basename(path)} "
                    f"in {time.time() - file_start:,.0f}s"
                )

    except Exception:
        logger.exception("Fatal error while filtering S2ORC dump")
        conn.close()
        raise
    finally:
        conn.close()

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("DONE. Summary:")
    logger.info(f"  Total lines read:                 {total_lines:,}")
    logger.info(f"  Malformed / missing-corpusid lines:{bad_json_lines:,}")
    logger.info(f"  Total papers (valid, w/ corpusid): {total_papers:,}")
    logger.info(f"  Discarded (no matching abstract):  {discarded_no_abstract:,}")
    logger.info(f"  Matched with an abstract:          {matched_with_abstract:,}")
    logger.info(f"    - Passed filter (kept):          {passed_filter:,}")
    logger.info(f"    - Failed filter (dropped):       {failed_filter:,}")
    avg_rate = total_papers / elapsed if elapsed > 0 else 0
    logger.info(f"  Elapsed: {elapsed:,.0f}s ({avg_rate:,.0f} papers/sec avg)")
    logger.info(f"  Output written to: {output_path}")
    logger.info("=" * 60)


def main():

    # input_dir = "data/s2orc/dump"
    # output_file = "data/s2orc/polymer_s2orc.jsonl.gz"
    # keyword = "polymer"

    # extract_polymer_s2orc(input_dir, output_file, keyword)

    # polymer_file = "data/s2orc/polymer_s2orc.jsonl.gz"
    # abstracts_dir = "data/s2ag/abstracts"
    # abstract_output_file="data/s2orc/polymer_s2ag_abstracts.jsonl.gz"

    # check_polymer_ids_have_abstracts(
    #     polymer_file, 
    #     abstracts_dir, 
    #     abstract_output_file)

    s2orc_dump_dir = "data/s2orc/dump"
    abstract_db_path = "data/s2ag/abstracts.db"
    output_path = "data/s2orc/s2orc_filtered_polymer.jsonl.gz"
    keyword = "polymer"

    filter_s2orc_by_keyword(
        s2orc_dump_dir=s2orc_dump_dir,
        abstract_db_path=abstract_db_path,
        output_path=output_path,
        keyword=keyword,
    )

if __name__ == "__main__":
    main()
