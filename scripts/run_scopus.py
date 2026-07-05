import logging

from synth_extract.mining import scopus


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    scopus.fetch_scopus_by_publisher(
        publisher="Springer Nature",
        db_path="data/scopus_springer_nature.db"
    )


if __name__ == "__main__":
    main()
