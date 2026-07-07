from pathlib import Path
import gzip
import json
import logging


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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # input_dir = "data/s2orc/dump"
    # output_file = "data/s2orc/polymer_s2orc.jsonl.gz"
    # keyword = "polymer"

    # extract_polymer_s2orc(input_dir, output_file, keyword)

    polymer_file = "data/s2orc/polymer_s2orc.jsonl.gz"
    abstracts_dir = "data/s2ag/abstracts"
    abstract_output_file="data/s2orc/polymer_s2ag_abstracts.jsonl.gz"

    check_polymer_ids_have_abstracts(
        polymer_file, 
        abstracts_dir, 
        abstract_output_file)

if __name__ == "__main__":
    main()
