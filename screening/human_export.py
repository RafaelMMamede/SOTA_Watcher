"""Export active, human-included title/abstract records to a Word document.

A human metadata "include" means advance to full-text screening, not final
eligibility. This export never substitutes an LLM assessment for a human vote.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


def _text(value, fallback="Not available"):
    if value is None or value == "":
        return fallback
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip() or fallback


def _list_field(value):
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return [value.strip()] if value.strip() else []
        value = parsed
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if value else []


def _authors(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, dict)):
                value = parsed
        except (TypeError, ValueError):
            return value.strip()
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (list, tuple)):
        names = []
        for item in value:
            if isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("display_name")
                    or item.get("author")
                    or item.get("full_name")
                )
                if isinstance(name, dict):
                    name = name.get("display_name") or name.get("name")
            else:
                name = item
            if name:
                names.append(str(name).strip())
        return ", ".join(names)
    return _text(value, "")


def _year(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def load_human_included(store):
    """Return only human-included records from the active candidate selection."""
    active_run = store.latest_candidate_selection_run()
    if active_run is None:
        raise ValueError("No active candidate-selection run. Apply a selection first.")

    counts = {
        (row["decision"] or "pending"): row["n"]
        for row in store.conn.execute(
            """
            SELECT h.decision AS decision, COUNT(*) AS n
              FROM candidate_selection c
              LEFT JOIN human_metadata_screening h USING (corpus_id)
             WHERE c.status = 'selected'
             GROUP BY h.decision
            """
        )
    }
    ids = [
        row["corpus_id"]
        for row in store.conn.execute(
            """
            SELECT c.corpus_id
              FROM candidate_selection c
              JOIN human_metadata_screening h USING (corpus_id)
             WHERE c.status = 'selected' AND h.decision = 'include'
             ORDER BY c.corpus_id
            """
        )
    ]
    papers = [store.get_paper(corpus_id) for corpus_id in ids]
    papers = [paper for paper in papers if paper is not None]
    papers.sort(
        key=lambda p: (
            0 if p.get("candidate_selection_core_protected") else 1,
            -(_year(p.get("year")) or 0),
            _text(p.get("title"), "").casefold(),
            p["corpus_id"],
        )
    )
    return papers, counts, active_run


def _set_styles(document):
    section = document.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9)
    normal.paragraph_format.space_after = Pt(5)

    for style_name, size in (
        ("Title", 20), ("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 10.5),
    ):
        style = document.styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(25, 53, 84)
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run(
        "SOTA_Watcher  |  Human title/abstract inclusions  |  Full-text review pending"
    )
    run.font.size = Pt(7)
    run.font.color.rgb = RGBColor(104, 111, 119)


def _label_line(document, label, value):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(2)
    lead = paragraph.add_run(f"{label}: ")
    lead.bold = True
    paragraph.add_run(_text(value))


def export_human_included_docx(store, path, *, include_abstracts=True):
    """Write an auditable Word listing of active, human-included papers."""
    papers, counts, active_run = load_human_included(store)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    _set_styles(document)

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Human-Selected Research Papers")
    subtitle = document.add_paragraph(
        "Included at title/abstract screening — provisional for full-text review"
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    _label_line(
        document, "Exported",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    _label_line(document, "Active candidate selection", active_run.get("run_id", ""))
    selected_total = sum(counts.values())
    _label_line(document, "Selected candidate pool", str(selected_total))
    _label_line(document, "Human included", str(len(papers)))
    _label_line(document, "Human excluded", str(counts.get("exclude", 0)))
    _label_line(document, "Human uncertain", str(counts.get("uncertain", 0)))
    _label_line(document, "Not yet human reviewed", str(counts.get("pending", 0)))

    document.add_paragraph(
        "Scope note: Only records in the current active candidate-selection pool "
        "with a human title/abstract decision of 'include' are listed. "
        "These decisions advance papers to full-text screening; they are not "
        "final full-text inclusion decisions. Excluded papers and papers "
        "outside the active selection are intentionally omitted."
    )

    distribution = Counter(
        (
            _year(p.get("year")) or "Unknown",
            "Core" if p.get("candidate_selection_core_protected") else "Supporting",
        )
        for p in papers
    )
    if papers:
        document.add_heading("Included papers by year", level=2)
        table = document.add_table(rows=1, cols=4)
        table.style = "Light Shading Accent 1"
        for i, label in enumerate(("Year", "Core", "Supporting", "Total")):
            table.rows[0].cells[i].text = label
        years = sorted({year for year, _ in distribution}, key=lambda v: str(v), reverse=True)
        for year in years:
            core = distribution.get((year, "Core"), 0)
            support = distribution.get((year, "Supporting"), 0)
            cells = table.add_row().cells
            for cell, value in zip(cells, (year, core, support, core + support)):
                cell.text = str(value)

        document.add_page_break()
        index = 0
        current_category = None
        current_year = None
        for paper in papers:
            category = (
                "Core literature"
                if paper.get("candidate_selection_core_protected")
                else "Supporting literature"
            )
            year = _year(paper.get("year")) or "Unknown"
            if category != current_category:
                document.add_heading(category, level=1)
                current_category = category
                current_year = None
            if year != current_year:
                document.add_heading(str(year), level=2)
                current_year = year

            index += 1
            document.add_heading(
                f"{index}. {_text(paper.get('title'), 'Untitled paper')}",
                level=3,
            )
            authors = _authors(paper.get("authors"))
            if authors:
                _label_line(document, "Authors", authors)

            info = [
                f"Year: {year}",
                f"Venue: {_text(paper.get('venue'))}",
                f"Citations: {_text(paper.get('citation_count'))}",
                f"Stream: {_text(paper.get('candidate_selection_stream'))}",
            ]
            document.add_paragraph("  |  ".join(info))
            _label_line(document, "DOI", paper.get("doi"))
            _label_line(document, "URL", paper.get("url"))
            _label_line(document, "Selection reason", paper.get("candidate_selection_reason"))
            sources = _list_field(paper.get("sources")) or _list_field(paper.get("source"))
            if sources:
                _label_line(document, "Discovery sources", ", ".join(sources))
            _label_line(document, "Corpus ID", paper.get("corpus_id"))
            if include_abstracts:
                _label_line(document, "Abstract", paper.get("abstract"))
            notes = _text(paper.get("human_metadata_screening_notes"), "")
            if notes:
                _label_line(document, "Human notes", notes)
            document.add_paragraph()

    document.save(output)
    return {
        "path": str(output),
        "selected": selected_total,
        "human_included": len(papers),
        "human_excluded": counts.get("exclude", 0),
        "human_uncertain": counts.get("uncertain", 0),
        "pending": counts.get("pending", 0),
        "abstracts_included": include_abstracts,
        "active_run": active_run.get("run_id", ""),
    }


def export_human_included_markdown(store, path, *, include_abstracts=True):
    """Export human-included records as plain-text Markdown for LLM review."""
    papers, counts, active_run = load_human_included(store)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected_total = sum(counts.values())
    lines = [
        "# SOTA_Watcher — Human-Selected Papers",
        "",
        "**Screening stage:** Human title/abstract inclusion (not final full-text eligibility)",
        f"**Exported:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Candidate selection run:** {_text(active_run.get('run_id'))}",
        f"**Selected candidate pool:** {selected_total}",
        f"**Human included:** {len(papers)}",
        f"**Human excluded:** {counts.get('exclude', 0)}",
        f"**Human uncertain:** {counts.get('uncertain', 0)}",
        f"**Not yet human reviewed:** {counts.get('pending', 0)}",
        "",
        "This document contains ONLY the currently selected candidates marked",
        "`include` by the human title/abstract reviewer. It does not contain",
        "human-excluded candidates or final full-text eligibility decisions.",
        "",
        "## Distribution by year and selection category",
        "",
        "| Year | Core | Supporting | Total |",
        "|---|---:|---:|---:|",
    ]
    distribution = Counter(
        (
            _year(p.get("year")) or "Unknown",
            "Core" if p.get("candidate_selection_core_protected") else "Supporting",
        )
        for p in papers
    )
    years = sorted(
        {year for year, _ in distribution},
        key=lambda v: str(v),
        reverse=True,
    )
    for year in years:
        core = distribution.get((year, "Core"), 0)
        support = distribution.get((year, "Supporting"), 0)
        lines.append(f"| {year} | {core} | {support} | {core + support} |")
    lines.extend(["", "## Included papers", ""])

    category = None
    current_year = None
    for index, paper in enumerate(papers, start=1):
        new_category = (
            "Core literature"
            if paper.get("candidate_selection_core_protected")
            else "Supporting literature"
        )
        year = _year(paper.get("year")) or "Unknown"
        if new_category != category:
            category = new_category
            current_year = None
            lines.extend([f"### {category}", ""])
        if year != current_year:
            current_year = year
            lines.extend([f"#### {year}", ""])

        title = _text(paper.get("title"), "Untitled paper").replace("\n", " ")
        lines.extend([
            f"##### {index}. {title}",
            "",
            f"- **Corpus ID:** {_text(paper.get('corpus_id'))}",
            f"- **Year:** {year}",
            f"- **Venue:** {_text(paper.get('venue'))}",
            f"- **Citations:** {_text(paper.get('citation_count'))}",
            f"- **Stream:** {_text(paper.get('candidate_selection_stream'))}",
            f"- **Selection reason:** {_text(paper.get('candidate_selection_reason'))}",
            f"- **DOI:** {_text(paper.get('doi'))}",
            f"- **URL:** {_text(paper.get('url'))}",
        ])
        authors = _authors(paper.get("authors"))
        if authors:
            lines.append(f"- **Authors:** {authors}")
        sources = _list_field(paper.get("sources")) or _list_field(paper.get("source"))
        if sources:
            lines.append(f"- **Discovery sources:** {', '.join(sources)}")
        if include_abstracts:
            abstract = _text(paper.get("abstract")).replace("\n", " ").strip()
            lines.extend(["", f"**Abstract:** {abstract}"])
        notes = _text(paper.get("human_metadata_screening_notes"), "")
        if notes:
            lines.extend(["", f"**Human notes:** {notes}"])
        lines.append("")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": str(output),
        "selected": selected_total,
        "human_included": len(papers),
        "human_excluded": counts.get("exclude", 0),
        "human_uncertain": counts.get("uncertain", 0),
        "pending": counts.get("pending", 0),
        "abstracts_included": include_abstracts,
        "active_run": active_run.get("run_id", ""),
        "format": "markdown",
    }
