import logging

from synth_extract.mining import scopus


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    scopus.fetch_scopus_by_publisher(
        publisher="Elsevier",
        max_pages=5,
    )


if __name__ == "__main__":
    main()
