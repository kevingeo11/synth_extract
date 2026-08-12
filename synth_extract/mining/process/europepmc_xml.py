"""Render Europe PMC JATS XML articles as Markdown."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lxml import etree

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


def _local_name(element) -> str:
    """Return an element's namespace-independent tag name."""
    return etree.QName(element).localname if isinstance(element.tag, str) else ""


def _text_content(element) -> str:
    """Return normalized descendant text."""
    return " ".join("".join(element.itertext()).split())


def _append_text(parts: list[str], text: str | None) -> None:
    """Append normalized XML text while preserving meaningful boundaries."""
    if not text:
        return
    cleaned = " ".join(text.split())
    if not cleaned:
        return
    if parts and text[0].isspace() and not parts[-1].endswith((" ", "\n")):
        parts.append(" ")
    parts.append(cleaned)
    if text[-1].isspace() and not parts[-1].endswith((" ", "\n")):
        parts.append(" ")


def _tex_content(element) -> str | None:
    """Extract TeX from a JATS formula element."""
    matches = element.xpath(".//*[local-name()='tex-math']")
    if not matches:
        return None

    tex = "".join(matches[0].itertext()).strip()
    delimiter_patterns = (
        r"\$\$(.*?)\$\$",
        r"\\\[(.*?)\\\]",
        r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)",
    )
    for pattern in delimiter_patterns:
        match = re.search(pattern, tex, flags=re.DOTALL)
        if match:
            return match.group(1).strip()

    document = re.search(
        r"\\begin\{document\}(.*?)\\end\{document\}",
        tex,
        flags=re.DOTALL,
    )
    if document:
        return document.group(1).strip()
    return tex or None


def _mathml_to_latex(element) -> str:
    """Translate common presentation MathML constructs to LaTeX."""
    tag = _local_name(element)
    children = [child for child in element if isinstance(child.tag, str)]

    if tag == "semantics":
        presentation = next(
            (child for child in children if _local_name(child) != "annotation"),
            None,
        )
        return _mathml_to_latex(presentation) if presentation is not None else ""
    if tag in {"annotation", "annotation-xml"}:
        return ""

    text = "".join(element.itertext()).strip()
    if tag in {"mi", "mn", "mtext"}:
        return text
    if tag == "mo":
        operators = {
            "−": "-",
            "×": r"\times ",
            "·": r"\cdot ",
            "≤": r"\le ",
            "≥": r"\ge ",
            "≠": r"\ne ",
            "≈": r"\approx ",
            "±": r"\pm ",
            "∞": r"\infty ",
            "∑": r"\sum ",
            "∏": r"\prod ",
            "∫": r"\int ",
            "→": r"\to ",
        }
        return operators.get(text, text)
    if tag == "mspace":
        return " "

    rendered = [_mathml_to_latex(child) for child in children]
    if tag in {"math", "mrow", "mstyle", "mpadded", "mphantom"}:
        return "".join(rendered)
    if tag == "msub" and len(rendered) == 2:
        return f"{rendered[0]}_{{{rendered[1]}}}"
    if tag == "msup" and len(rendered) == 2:
        return f"{rendered[0]}^{{{rendered[1]}}}"
    if tag == "msubsup" and len(rendered) == 3:
        return f"{rendered[0]}_{{{rendered[1]}}}^{{{rendered[2]}}}"
    if tag == "mfrac" and len(rendered) == 2:
        return f"\\frac{{{rendered[0]}}}{{{rendered[1]}}}"
    if tag == "msqrt":
        return f"\\sqrt{{{''.join(rendered)}}}"
    if tag == "mroot" and len(rendered) == 2:
        return f"\\sqrt[{rendered[1]}]{{{rendered[0]}}}"
    if tag == "mfenced":
        opening = element.get("open", "(")
        closing = element.get("close", ")")
        separator = element.get("separators", ",")[:1] or ","
        return f"{opening}{separator.join(rendered)}{closing}"
    if tag == "mover" and len(rendered) == 2:
        return f"\\overset{{{rendered[1]}}}{{{rendered[0]}}}"
    if tag == "munder" and len(rendered) == 2:
        return f"\\underset{{{rendered[1]}}}{{{rendered[0]}}}"
    if tag == "munderover" and len(rendered) == 3:
        return f"\\overset{{{rendered[2]}}}{{\\underset{{{rendered[1]}}}{{{rendered[0]}}}}}"
    if tag == "mtable":
        rows = [value for value in rendered if value]
        return r"\begin{matrix}" + r" \\ ".join(rows) + r"\end{matrix}"
    if tag == "mtr":
        return " & ".join(value for value in rendered if value)
    if tag == "mtd":
        return "".join(rendered)

    LOG.debug("Flattening unsupported MathML tag: <%s>", tag)
    return "".join(rendered) or text


def _mathml_content(element) -> str | None:
    """Extract an annotation or translate the first MathML expression."""
    math_nodes = element.xpath(".//*[local-name()='math']")
    if not math_nodes:
        return None

    annotations = math_nodes[0].xpath(".//*[local-name()='annotation']")
    for annotation in annotations:
        encoding = (annotation.get("encoding") or "").casefold()
        if "tex" in encoding:
            tex = "".join(annotation.itertext()).strip()
            if tex:
                return tex

    latex = _mathml_to_latex(math_nodes[0]).strip()
    return latex or None


def _render_inline_content(element) -> str:
    parts: list[str] = []
    _append_text(parts, element.text)
    for child in element:
        if isinstance(child.tag, str):
            parts.append(_render_inline(child))
        _append_text(parts, child.tail)
    return "".join(parts)


def _render_inline(element) -> str:
    """Render one inline JATS element."""
    tag = _local_name(element)
    if not tag or tag in {"supplementary-material", "graphic", "inline-graphic"}:
        return ""
    if tag == "glyph-data":
        return ""
    if tag == "private-char":
        name = (element.get("name") or element.get("description") or "").upper()
        for marker, replacement in (
            ("TRIPLE BOND", "≡"),
            ("DOUBLE BOND", "="),
            ("SINGLE BOND", "–"),
        ):
            if marker in name:
                return replacement
        LOG.warning("Skipping unsupported JATS private character: %s", name or "unknown")
        return ""

    content = _render_inline_content(element)
    if tag == "italic":
        return f"*{content}*"
    if tag == "bold":
        return f"**{content}**"
    if tag == "sup":
        return f"<sup>{content}</sup>"
    if tag == "sub":
        return f"<sub>{content}</sub>"
    if tag in {"monospace", "code"}:
        return f"`{content}`"
    if tag == "underline":
        return f"<u>{content}</u>"
    if tag in {"break", "hr"}:
        return "  \n"
    if tag == "inline-formula":
        tex = _tex_content(element)
        if tex:
            return f"${tex}$"
        mathml = _mathml_content(element)
        if mathml:
            return f"${mathml}$"
        return content

    # xref, ext-link, named-content, email, uri, and other semantic inline
    # elements retain their displayed text without publisher-specific markup.
    LOG.debug("Rendering JATS inline tag as text: <%s>", tag)
    return content


def _formula_to_markdown(formula) -> str:
    tex = _tex_content(formula)
    if tex:
        return f"$$\n{tex}\n$$"

    mathml = _mathml_content(formula)
    if mathml:
        return f"$$\n{mathml}\n$$"

    fallback = _render_inline_content(formula).strip() or _text_content(formula)
    if fallback:
        return fallback
    LOG.warning("Could not extract a formula from a <disp-formula> element")
    return ""


def _caption_to_markdown(caption) -> str:
    """Render caption paragraphs without leaking raw TeX alternatives."""
    paragraphs = caption.xpath(".//*[local-name()='p']")
    if paragraphs:
        return " ".join(
            text
            for paragraph in paragraphs
            if (text := _paragraph_to_markdown(paragraph))
        )
    return _render_inline_content(caption).strip()


def _figure_to_markdown(figure) -> str:
    label = figure.xpath("./*[local-name()='label'][1]")
    caption = figure.xpath("./*[local-name()='caption'][1]")
    label_text = _text_content(label[0]) if label else "Figure"
    caption_text = _caption_to_markdown(caption[0]) if caption else ""
    title = ". ".join(text for text in (label_text, caption_text) if text)
    return f"**{title}**" if title else ""


def _table_cell_to_markdown(cell) -> str:
    """Render table-cell paragraphs as distinct lines."""
    paragraphs = cell.xpath("./*[local-name()='p']")
    if paragraphs:
        return "<br>".join(
            text
            for paragraph in paragraphs
            if (text := _paragraph_to_markdown(paragraph))
        )
    return _render_inline_content(cell).strip()


def _table_to_markdown(table_wrap) -> str:
    parts: list[str] = []
    labels = table_wrap.xpath("./*[local-name()='label'][1]")
    captions = table_wrap.xpath("./*[local-name()='caption'][1]")
    label_text = _text_content(labels[0]) if labels else ""
    caption_text = _caption_to_markdown(captions[0]) if captions else ""
    title = ". ".join(text for text in (label_text, caption_text) if text)
    if title:
        parts.append(f"**{title}**")

    tables = table_wrap.xpath("./*[local-name()='table'][1]")
    if not tables:
        LOG.warning("A <table-wrap> element contains no <table>")
        return "\n\n".join(parts)

    logical_rows: list[tuple[list[str], bool]] = []
    active_rowspans: dict[int, tuple[int, str]] = {}
    for row in tables[0].xpath(".//*[local-name()='tr']"):
        grid: dict[int, str] = {}
        for column, (remaining, value) in list(active_rowspans.items()):
            grid[column] = value
            if remaining <= 1:
                del active_rowspans[column]
            else:
                active_rowspans[column] = (remaining - 1, value)

        cursor = 0
        for cell in row:
            if _local_name(cell) not in {"th", "td"}:
                continue
            while cursor in grid:
                cursor += 1
            cell_text = _table_cell_to_markdown(cell).replace("|", r"\|")
            cell_text = cell_text.replace("\n", " ")
            try:
                colspan = max(1, int(cell.get("colspan", "1")))
                rowspan = max(1, int(cell.get("rowspan", "1")))
            except ValueError:
                colspan = rowspan = 1

            occupied_columns = []
            for _ in range(colspan):
                while cursor in grid:
                    cursor += 1
                grid[cursor] = cell_text
                occupied_columns.append(cursor)
                cursor += 1
            if rowspan > 1:
                for column in occupied_columns:
                    active_rowspans[column] = (rowspan - 1, cell_text)

        if grid:
            width = max(grid) + 1
            values = [grid.get(column, "") for column in range(width)]
            is_header = any(_local_name(parent) == "thead" for parent in row.iterancestors())
            logical_rows.append((values, is_header))

    if not logical_rows:
        return "\n\n".join(parts)

    column_count = max(len(row) for row, _ in logical_rows)
    rows = [
        (row + [""] * (column_count - len(row)), is_header)
        for row, is_header in logical_rows
    ]
    header_rows = [row for row, is_header in rows if is_header]
    body = [row for row, is_header in rows if not is_header]
    if header_rows:
        header = []
        for column in range(column_count):
            values = []
            for row in header_rows:
                value = row[column]
                if value and value not in values:
                    values.append(value)
            header.append("<br>".join(values))
    else:
        header, *body = [row for row, _ in rows]

    markdown_rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    parts.append("\n".join(markdown_rows))
    return "\n\n".join(parts)


def _paragraph_to_markdown(paragraph) -> str:
    parts: list[str] = []
    _append_text(parts, paragraph.text)
    for child in paragraph:
        tag = _local_name(child)
        if tag == "disp-formula":
            rendered = _formula_to_markdown(child)
            if rendered:
                parts.append(f"\n\n{rendered}\n\n")
        elif tag == "fig":
            rendered = _figure_to_markdown(child)
            if rendered:
                parts.append(f"\n\n{rendered}\n\n")
        elif tag == "table-wrap":
            rendered = _table_to_markdown(child)
            if rendered:
                parts.append(f"\n\n{rendered}\n\n")
        elif tag == "list":
            rendered = _list_to_markdown(child)
            if rendered:
                parts.append(f"\n\n{rendered}\n\n")
        else:
            parts.append(_render_inline(child))
        _append_text(parts, child.tail)
    return "".join(parts).strip()


def _list_to_markdown(list_element, depth: int = 0) -> str:
    ordered = list_element.get("list-type") in {"order", "ordered", "alpha-lower"}
    marker = "1." if ordered else "-"
    indent = "  " * depth
    lines: list[str] = []

    for item in list_element.xpath("./*[local-name()='list-item']"):
        item_parts: list[str] = []
        nested: list[str] = []
        for child in item:
            tag = _local_name(child)
            if tag == "p":
                text = _paragraph_to_markdown(child)
                if text:
                    item_parts.append(text)
            elif tag == "list":
                rendered = _list_to_markdown(child, depth + 1)
                if rendered:
                    nested.append(rendered)
        if item_parts:
            lines.append(f"{indent}{marker} {' '.join(item_parts)}")
        lines.extend(nested)
    return "\n".join(lines)


def _section_to_markdown(section, level: int = 2) -> str:
    if section.get("sec-type") == "supplementary-material":
        return ""

    parts: list[str] = []
    labels = section.xpath("./*[local-name()='label'][1]")
    titles = section.xpath("./*[local-name()='title'][1]")
    if titles:
        title = _text_content(titles[0])
        if labels:
            title = f"{_text_content(labels[0])} {title}".strip()
        if title:
            parts.append(f"{'#' * min(level, 6)} {title}")

    for child in section:
        tag = _local_name(child)
        if tag in {"title", "label"}:
            continue
        if tag == "p":
            rendered = _paragraph_to_markdown(child)
        elif tag == "sec":
            rendered = _section_to_markdown(child, level + 1)
        elif tag == "fig":
            rendered = _figure_to_markdown(child)
        elif tag == "table-wrap":
            rendered = _table_to_markdown(child)
        elif tag == "disp-formula":
            rendered = _formula_to_markdown(child)
        elif tag == "list":
            rendered = _list_to_markdown(child)
        else:
            LOG.debug("Skipping unsupported JATS section block: <%s>", tag)
            rendered = ""
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)


def _front_to_markdown(root) -> str:
    parts: list[str] = []
    titles = root.xpath(
        "./*[local-name()='front']/*[local-name()='article-meta']"
        "/*[local-name()='title-group']/*[local-name()='article-title'][1]"
    )
    if titles:
        title = _text_content(titles[0])
        if title:
            parts.append(f"# {title}")

    abstracts = root.xpath(
        "./*[local-name()='front']/*[local-name()='article-meta']"
        "/*[local-name()='abstract'][1]"
    )
    if abstracts:
        paragraphs = abstracts[0].xpath(".//*[local-name()='p']")
        rendered_paragraphs = [
            text
            for paragraph in paragraphs
            if (text := _paragraph_to_markdown(paragraph))
        ]
        if rendered_paragraphs:
            parts.extend(["## Abstract", *rendered_paragraphs])
    return "\n\n".join(parts)


def _render_europepmc_xml(xml_path: Path) -> str:
    """Parse one Europe PMC JATS document and return Markdown text."""
    parser = etree.XMLParser(
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        recover=False,
    )
    root = etree.parse(str(xml_path), parser).getroot()
    if _local_name(root) != "article":
        raise ValueError(f"Expected a JATS <article> root in {xml_path}")

    parts: list[str] = []
    front = _front_to_markdown(root)
    if front:
        parts.append(front)

    bodies = root.xpath("./*[local-name()='body'][1]")
    if bodies:
        body = bodies[0]
        for child in body:
            tag = _local_name(child)
            if tag == "sec":
                rendered = _section_to_markdown(child)
            elif tag == "p":
                rendered = _paragraph_to_markdown(child)
            elif tag == "fig":
                rendered = _figure_to_markdown(child)
            elif tag == "table-wrap":
                rendered = _table_to_markdown(child)
            elif tag == "disp-formula":
                rendered = _formula_to_markdown(child)
            elif tag == "list":
                rendered = _list_to_markdown(child)
            else:
                LOG.debug("Skipping unsupported JATS body block: <%s>", tag)
                rendered = ""
            if rendered:
                parts.append(rendered)

    markdown = "\n\n".join(parts).strip()
    if not markdown:
        raise ValueError(f"No article content could be rendered from {xml_path}")
    return markdown + "\n"


def europepmc_xml_to_markdown(
    xml_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a Europe PMC JATS XML document to a Markdown file.

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
    markdown_path.write_text(_render_europepmc_xml(input_path), encoding="utf-8")
    return markdown_path


xml_to_markdown = europepmc_xml_to_markdown


__all__ = ["europepmc_xml_to_markdown", "xml_to_markdown"]
