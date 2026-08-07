"""Render Elsevier full-text XML articles as Markdown."""

from __future__ import annotations

import logging
from pathlib import Path

from lxml import etree

LOG = logging.getLogger(__name__)

NS = {
    "ce": "http://www.elsevier.com/xml/common/dtd",
    "xocs": "http://www.elsevier.com/xml/xocs/dtd",
    "sb": "http://www.elsevier.com/xml/common/struct-bib/dtd",
    "ja": "http://www.elsevier.com/xml/ja/dtd",
    "mml": "http://www.w3.org/1998/Math/MathML",
}


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


def build_float_lookup(root):
    lookup = {}

    for elem in root.xpath(".//ce:floats/*", namespaces=NS):
        elem_id = elem.get("id")

        if elem_id:
            lookup[elem_id] = elem

    return lookup


def table_cell_to_markdown(entry) -> str:
    parts = []

    if entry.text:
        parts.append(clean_fragment(entry.text))

    for child in entry:
        tag = etree.QName(child).localname

        if tag == "inf":
            parts.append(f"<sub>{text_content(child)}</sub>")

        elif tag == "sup":
            parts.append(f"<sup>{text_content(child)}</sup>")

        elif tag == "hsp":
            parts.append(" ")

        elif tag == "br":
            parts.append(" ")

        elif tag == "italic":
            parts.append(f"*{text_content(child)}*")

        else:
            parts.append(text_content(child))

        if child.tail:
            append_tail(parts, child.tail)

    return (
        "".join(parts)
        .strip()
        .replace("|", r"\|")
        .replace("\n", " ")
    )


def table_to_markdown(table) -> str:
    parts = []

    label = table.find("ce:label", namespaces=NS)
    caption = table.find(".//ce:caption", namespaces=NS)

    label_text = text_content(label) if label is not None else ""
    caption_text = text_content(caption) if caption is not None else ""
    table_title = ". ".join(text for text in (label_text, caption_text) if text)
    if table_title:
        parts.append(f"**{table_title}**")

    rows = []

    for row in table.xpath(".//*[local-name()='row']"):
        cells = []

        for entry in row.xpath("./*[local-name()='entry']"):
            cells.append(table_cell_to_markdown(entry))

        if cells:
            rows.append(cells)

    if not rows:
        return "\n\n".join(parts)

    column_count = max(len(row) for row in rows)
    normalized_rows = [
        row + [""] * (column_count - len(row))
        for row in rows
    ]
    header = normalized_rows[0]
    body = normalized_rows[1:]

    parts.append(
        "| " + " | ".join(header) + " |\n"
        + "| " + " | ".join(["---"] * len(header)) + " |\n"
        + "\n".join(
            "| " + " | ".join(row) + " |"
            for row in body
        )
    )

    return "\n\n".join(parts)


def text_content(elem) -> str:
    return " ".join("".join(elem.itertext()).split())


def clean_fragment(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def append_tail(parts, tail):
    tail = clean_fragment(tail)

    if not tail:
        return

    if tail[0] in ".,;:!?)]}":
        parts.append(tail)
    else:
        parts.append(" " + tail)


def mathml_to_latex(elem) -> str:
    tag = etree.QName(elem).localname

    if tag in {"mi", "mn", "mo"}:
        return text_content(elem)

    if tag == "mspace":
        return " "

    if tag == "msub":
        children = list(elem)
        if len(children) == 2:
            base = mathml_to_latex(children[0])
            sub = mathml_to_latex(children[1])
            return f"{base}_{{{sub}}}"

    if tag == "msup":
        children = list(elem)
        if len(children) == 2:
            base = mathml_to_latex(children[0])
            superscript = mathml_to_latex(children[1])
            return f"{base}^{{{superscript}}}"

    if tag == "msubsup":
        children = list(elem)
        if len(children) == 3:
            base = mathml_to_latex(children[0])
            subscript = mathml_to_latex(children[1])
            superscript = mathml_to_latex(children[2])
            return f"{base}_{{{subscript}}}^{{{superscript}}}"

    if tag == "mfrac":
        children = list(elem)
        if len(children) == 2:
            numerator = mathml_to_latex(children[0])
            denominator = mathml_to_latex(children[1])
            return f"\\frac{{{numerator}}}{{{denominator}}}"

    # fallback: recursively render children
    parts = []

    for child in elem:
        parts.append(mathml_to_latex(child))


    return "".join(parts).strip()


def display_to_markdown(display) -> str:
    math = display.find(".//mml:math", namespaces=NS)

    if math is None:
        return text_content(display)

    equation = mathml_to_latex(math)

    return f"\n\n$$\n{equation}\n$$\n\n"


def figure_to_markdown(figure) -> str:
    """Render an Elsevier figure label and caption when no image is available."""
    label = figure.find("ce:label", namespaces=NS)
    caption = figure.find(".//ce:caption", namespaces=NS)
    label_text = text_content(label) if label is not None else "Figure"
    caption_text = text_content(caption) if caption is not None else ""
    title = ". ".join(text for text in (label_text, caption_text) if text)
    return f"**{title}**" if title else ""


def render_inline(elem) -> str:
    localname = etree.QName(elem).localname

    if localname in {"cross-ref", "cross-refs"}:
        return text_content(elem)

    elif localname == "inf":
        return f"<sub>{text_content(elem)}</sub>"

    elif localname == "sup":
        return f"<sup>{text_content(elem)}</sup>"

    elif localname == "italic":
        return f"*{text_content(elem)}*"

    elif localname == "hsp":
        return " "

    elif localname == "bold":
        return f"**{text_content(elem)}**"

    elif localname == "underline":
        return f"<u>{text_content(elem)}</u>"

    elif localname == "monospace":
        return f"`{text_content(elem)}`"

    elif localname == "br":
        return "  \n"

    LOG.debug("Rendering unhandled inline XML tag as text: <%s>", localname)

    parts = []

    if elem.text:
        parts.append(clean_fragment(elem.text))

    for child in elem:
        parts.append(render_inline(child))

        if child.tail:
            append_tail(parts, child.tail)

    return "".join(parts)


def paragraph_to_markdown(para, float_lookup) -> str:
    parts = []

    if para.text:
        parts.append(clean_fragment(para.text))

    for child in para:
        localname = etree.QName(child).localname

        if localname == "float-anchor":
            refid = child.get("refid")

            if refid in float_lookup:
                float_elem = float_lookup[refid]
                float_type = etree.QName(float_elem).localname

                if float_type == "table":
                    parts.append(
                        "\n\n" + table_to_markdown(float_elem) + "\n\n"
                    )
                elif float_type == "figure":
                    figure = figure_to_markdown(float_elem)
                    if figure:
                        parts.append("\n\n" + figure + "\n\n")

        elif localname == "display":
            parts.append(display_to_markdown(child))

        else:
            parts.append(render_inline(child))

        if child.tail:
            append_tail(parts, child.tail)

    return "".join(parts).strip()


def list_to_markdown(list_element, float_lookup, depth=0) -> str:
    """Render an Elsevier list, including nested lists."""
    lines = []
    indent = "  " * depth
    for item in list_element.findall("ce:list-item", namespaces=NS):
        item_parts = []
        nested_lists = []
        for child in item:
            localname = etree.QName(child).localname
            if localname in {"para", "simple-para"}:
                text = paragraph_to_markdown(child, float_lookup)
                if text:
                    item_parts.append(text)
            elif localname == "list":
                nested_lists.append(list_to_markdown(child, float_lookup, depth + 1))

        item_text = " ".join(item_parts).strip()
        if item_text:
            lines.append(f"{indent}- {item_text}")
        lines.extend(text for text in nested_lists if text)
    return "\n".join(lines)


def section_to_markdown(section, float_lookup, level=2):
    parts = []

    title = section.find("ce:section-title", namespaces=NS)
    label = section.find("ce:label", namespaces=NS)

    if title is not None:
        title_text = text_content(title)

        if label is not None:
            title_text = f"{text_content(label)} {title_text}"

        parts.append(f"{'#' * level} {title_text}")

    for child in section:
        localname = etree.QName(child).localname

        if localname == "para":
            text = paragraph_to_markdown(
                child,
                float_lookup,
            )

            if text:
                parts.append(text)

        elif localname == "list":
            text = list_to_markdown(child, float_lookup)
            if text:
                parts.append(text)

        elif localname == "display":
            text = display_to_markdown(child).strip()
            if text:
                parts.append(text)

        elif localname == "section":
            parts.append(
                section_to_markdown(
                    child,
                    float_lookup,
                    level=min(level + 1, 6),
                )
            )

    return "\n\n".join(part for part in parts if part)


def head_to_markdown(root) -> str:
    parts = []

    title = root.find(
        ".//ja:head/ce:title",
        namespaces=NS,
    )

    if title is not None:
        parts.append(f"# {text_content(title)}")

    abstract = root.find(
        ".//ja:head/ce:abstract[@class='author']",
        namespaces=NS,
    )

    if abstract is not None:
        abstract_paragraphs = abstract.findall(
            ".//ce:simple-para",
            namespaces=NS,
        )

        if abstract_paragraphs:
            parts.append("## Abstract")
            for abstract_para in abstract_paragraphs:
                paragraph = paragraph_to_markdown(
                    abstract_para,
                    float_lookup={},
                )
                if paragraph:
                    parts.append(paragraph)

    return "\n\n".join(parts)


def _render_elsevier_xml(xml_path: Path) -> str:
    """Parse one Elsevier XML document and return its Markdown text."""
    parser = etree.XMLParser(
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        recover=False,
    )
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    float_lookup = build_float_lookup(root)

    parts = []

    head_md = head_to_markdown(root)

    if head_md:
        parts.append(head_md)

    top_sections = root.xpath(
        ".//ce:sections/ce:section",
        namespaces=NS,
    )

    for section in top_sections:
        parts.append(
            section_to_markdown(
                section,
                float_lookup=float_lookup,
            )
        )

    markdown = "\n\n".join(part for part in parts if part).strip()
    if not markdown:
        raise ValueError(f"No article content could be rendered from {xml_path}")
    return markdown + "\n"


def elsevier_xml_to_markdown(
    xml_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert an Elsevier full-text XML document to a Markdown file.

    ``output_path`` may be an output directory or an explicit ``.md`` file.
    When omitted, the Markdown file is written beside the XML with the same
    stem. The path to the generated Markdown file is returned.
    """
    input_path = Path(xml_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"XML file not found: {input_path}")
    if input_path.suffix.lower() != ".xml":
        raise ValueError(f"Expected an XML file, received: {input_path}")

    markdown_path = _output_path_for_xml(input_path, output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_elsevier_xml(input_path), encoding="utf-8")
    return markdown_path


# Short generic name for callers that already know they are processing
# Elsevier XML.
xml_to_markdown = elsevier_xml_to_markdown


__all__ = ["elsevier_xml_to_markdown", "xml_to_markdown"]
