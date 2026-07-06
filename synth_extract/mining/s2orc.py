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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
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
    

def main():
    input_dir = "data/s2orc/dump"
    output_file = "data/s2orc/polymer_s2orc.jsonl.gz"
    keyword = "polymer"

    extract_polymer_s2orc(input_dir, output_file, keyword)

if __name__ == "__main__":
    main()