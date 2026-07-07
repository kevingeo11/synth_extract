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
    polymer_file = Path(polymer_file)
    abstracts_dir = Path(abstracts_dir)
    abstract_output_file = Path(abstract_output_file)

    abstract_output_file.parent.mkdir(parents=True, exist_ok=True)

    if abstract_output_file.exists():
        raise FileExistsError(
            f"Output file already exists: {abstract_output_file}\n"
            "Delete it or specify a different output file."
        )

    keyword = keyword.lower()

    polymer_ids_list = []

    logging.info("Loading polymer corpus IDs from %s", polymer_file)

    with gzip.open(polymer_file, "rt", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            cid = paper.get("corpusid")

            if cid is not None:
                polymer_ids_list.append(int(cid))
            else:
                logging.warning(
                    "Missing corpusid in polymer record: %s",
                    paper.get("paperid"),
                )

    logging.info(
        "Loaded %s polymer corpus IDs including duplicates",
        f"{len(polymer_ids_list):,}",
    )

    polymer_ids = set(polymer_ids_list)

    logging.info(
        "Loaded %s unique polymer corpus IDs",
        f"{len(polymer_ids):,}",
    )

    if not polymer_ids:
        raise ValueError("No corpus IDs found in polymer file")

    found_ids = set()
    abstract_polymer_ids = set()

    total_abstract_rows = 0
    total_matching_cids = 0
    total_written = 0

    abstract_files = sorted(abstracts_dir.glob("*.gz"))

    logging.info("Found %s abstract gz files", f"{len(abstract_files):,}")
    logging.info("Writing matching polymer abstracts to %s", abstract_output_file)

    with gzip.open(abstract_output_file, "wt", encoding="utf-8") as out:
        for gz_file in abstract_files:
            file_rows = 0
            file_matching_cids = 0
            file_written = 0

            logging.info("Starting %s", gz_file.name)

            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                for line in f:
                    file_rows += 1
                    total_abstract_rows += 1

                    record = json.loads(line)
                    cid = record.get("corpusid")

                    if cid is None:
                        continue

                    cid = int(cid)

                    if cid not in polymer_ids:
                        continue

                    found_ids.add(cid)
                    file_matching_cids += 1
                    total_matching_cids += 1

                    abstract = (record.get("abstract") or "").lower()

                    if keyword in abstract:
                        abstract_polymer_ids.add(cid)
                        file_written += 1
                        total_written += 1
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")

            logging.info(
                "Done %s | rows=%s | polymer_cid_matches=%s | "
                "abstract_keyword_hits=%s | total_found_cids=%s / %s",
                gz_file.name,
                f"{file_rows:,}",
                f"{file_matching_cids:,}",
                f"{file_written:,}",
                f"{len(found_ids):,}",
                f"{len(polymer_ids):,}",
            )

            out.flush()

            if len(found_ids) == len(polymer_ids):
                logging.info("All polymer IDs found in abstracts. Stopping early.")
                break

    missing_ids = polymer_ids - found_ids
    missing_keyword_ids = polymer_ids - abstract_polymer_ids

    logging.info("Total abstract records scanned: %s", f"{total_abstract_rows:,}")
    logging.info("Unique polymer IDs: %s", f"{len(polymer_ids):,}")
    logging.info("Found in abstracts: %s", f"{len(found_ids):,}")
    logging.info("Missing from abstracts: %s", f"{len(missing_ids):,}")
    logging.info("Abstracts containing keyword '%s': %s", keyword, f"{total_written:,}")
    logging.info(
        "Unique polymer IDs whose abstract contains keyword '%s': %s",
        keyword,
        f"{len(abstract_polymer_ids):,}",
    )
    logging.info(
        "Polymer IDs found in S2ORC but not keyword-positive in abstracts: %s",
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