"""Render Springer Nature JATS XML articles as Markdown."""

from __future__ import annotations

import logging
from pathlib import Path

from lxml import etree

from .europepmc_xml import (
    _figure_to_markdown,
    _formula_to_markdown,
    _list_to_markdown,
    _local_name,
    _paragraph_to_markdown,
    _render_inline_content,
    _section_to_markdown,
    _table_to_markdown,
)

LOG = logging.getLogger(__name__)


def _output_path_for_xml(
    xml_path: Path,
    output_path: str | Path | None,
) -> Path:
    """Resolve a directory or explicit Markdown output path."""
    if output_path is None:
        return xml_path.with_suffix(".md")

    requested_path = Path(output_path)
    if requested_path.suffix.lower() == ".md":
        return requested_path
    return requested_path / f"{xml_path.stem}.md"


def _render_block(element, section_level: int = 2) -> str:
    """Render one block-level JATS element."""
    tag = _local_name(element)
    if tag == "p":
        return _paragraph_to_markdown(element)
    if tag == "sec":
        return _section_to_markdown(element, section_level)
    if tag == "fig":
        return _figure_to_markdown(element)
    if tag == "table-wrap":
        return _table_to_markdown(element)
    if tag == "disp-formula":
        return _formula_to_markdown(element)
    if tag == "list":
        return _list_to_markdown(element)
    LOG.debug("Skipping unsupported Springer JATS block: <%s>", tag)
    return ""


def _primary_abstract(metadata):
    """Select the article abstract rather than a publisher summary."""
    abstracts = metadata.xpath("./*[local-name()='abstract']")
    if not abstracts:
        return None

    for abstract in abstracts:
        if not (abstract.get("abstract-type") or "").strip():
            return abstract

    summary_types = {"highlights", "longsummary", "shortsummary"}
    for abstract in abstracts:
        abstract_type = (abstract.get("abstract-type") or "").casefold()
        if abstract_type not in summary_types:
            return abstract
    return abstracts[0]


def _abstract_section_to_markdown(section) -> str:
    """Render an abstract section, omitting duplicate and graphic-only headings."""
    titles = section.xpath("./*[local-name()='title'][1]")
    title = (
        " ".join("".join(titles[0].itertext()).split()).casefold()
        if titles
        else ""
    )
    if title in {"graphic abstract", "graphical abstract"}:
        return ""
    if title and title != "abstract":
        return _section_to_markdown(section, level=3)

    parts: list[str] = []
    for child in section:
        if _local_name(child) in {"title", "label"}:
            continue
        rendered = _render_block(child, section_level=3)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)


def _abstract_to_markdown(abstract) -> str:
    parts: list[str] = []
    for child in abstract:
        tag = _local_name(child)
        if tag == "title":
            continue
        if tag == "sec":
            rendered = _abstract_section_to_markdown(child)
        else:
            rendered = _render_block(child, section_level=3)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)


def _metadata_for_root(root):
    if _local_name(root) == "article":
        nodes = root.xpath(
            "./*[local-name()='front']/*[local-name()='article-meta'][1]"
        )
    else:
        nodes = root.xpath(".//*[local-name()='book-part-meta'][1]")
    return nodes[0] if nodes else None


def _front_to_markdown(metadata) -> str:
    if metadata is None:
        return ""

    parts: list[str] = []
    titles = metadata.xpath(
        "./*[local-name()='title-group']/*["
        "local-name()='article-title' or local-name()='book-part-title' "
        "or local-name()='title'][1]"
    )
    if not titles:
        titles = metadata.xpath("./*[local-name()='book-part-title'][1]")
    if titles:
        title = _render_inline_content(titles[0]).strip()
        if title:
            parts.append(f"# {title}")

    abstract = _primary_abstract(metadata)
    if abstract is not None:
        rendered = _abstract_to_markdown(abstract)
        if rendered:
            parts.extend(["## Abstract", rendered])
    return "\n\n".join(parts)


def _body_for_root(root):
    if _local_name(root) == "article":
        bodies = root.xpath("./*[local-name()='body'][1]")
    else:
        book_parts = root.xpath(".//*[local-name()='book-part'][1]")
        container = book_parts[0] if book_parts else root
        bodies = container.xpath("./*[local-name()='body'][1]")
        if not bodies:
            bodies = container.xpath(".//*[local-name()='body'][1]")
    return bodies[0] if bodies else None


def _render_springer_xml(xml_path: Path) -> str:
    """Parse one Springer Nature JATS document and return Markdown text."""
    parser = etree.XMLParser(
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        recover=False,
    )
    root = etree.parse(str(xml_path), parser).getroot()
    root_tag = _local_name(root)
    if root_tag not in {"article", "book-part", "book-part-wrapper"}:
        raise ValueError(
            "Expected a Springer JATS <article> or <book-part> root "
            f"in {xml_path}; found <{root_tag}>"
        )

    parts: list[str] = []
    front = _front_to_markdown(_metadata_for_root(root))
    if front:
        parts.append(front)

    body = _body_for_root(root)
    if body is not None:
        for child in body:
            rendered = _render_block(child)
            if rendered:
                parts.append(rendered)

    markdown = "\n\n".join(parts).strip()
    if not markdown:
        raise ValueError(f"No article content could be rendered from {xml_path}")
    return markdown + "\n"


def springer_xml_to_markdown(
    xml_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert Springer Nature JATS XML to a Markdown file.

    ``output_path`` may be an output directory or an explicit ``.md`` file.
    When omitted, the Markdown file is written beside the XML with the same
    stem. The generated Markdown path is returned.
    """
    input_path = Path(xml_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"XML file not found: {input_path}")
    if input_path.suffix.lower() != ".xml":
        raise ValueError(f"Expected an XML file, received: {input_path}")

    markdown_path = _output_path_for_xml(input_path, output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_springer_xml(input_path), encoding="utf-8")
    return markdown_path


xml_to_markdown = springer_xml_to_markdown


__all__ = ["springer_xml_to_markdown", "xml_to_markdown"]
