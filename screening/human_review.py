"""Interactive, resumable human title/abstract screening."""
from __future__ import annotations

from dataclasses import dataclass
import textwrap

from utils.deduplication import decode, present


@dataclass
class HumanScreenSummary:
    eligible_pool: int = 0
    already_reviewed: int = 0
    remaining: int = 0
    session_reviewed: int = 0
    included: int = 0
    excluded: int = 0
    skipped: int = 0
    quit: bool = False


def _values(paper: dict, plural: str, singular: str) -> list[str]:
    values = decode(paper.get(plural), [])
    if not values and present(paper.get(singular)):
        values = [paper.get(singular)]
    return sorted({
        str(value).strip()
        for value in values
        if present(value)
    })


def _text(value, fallback="—"):
    value = str(value or "").strip()
    return value if value else fallback


def _wrap(value, width=100, indent=""):
    text = _text(value, "")
    if not text:
        return "—"
    return textwrap.fill(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        replace_whitespace=True,
    )


def format_human_screen_paper(
    paper: dict,
    *,
    index: int,
    total: int,
    reviewed: int,
) -> str:
    """Render one candidate using only persisted metadata/screening state."""
    sources = ", ".join(_values(paper, "sources", "source")) or "—"
    topics = ", ".join(
        _values(paper, "search_topics", "search_topic")
    ) or "—"

    publication_type = (
        paper.get("publication_type")
        or paper.get("openalex_type")
        or paper.get("openalex_crossref_type")
        or paper.get("venue_type")
        or "—"
    )

    rank = paper.get("candidate_selection_citation_rank")
    quota = paper.get("candidate_selection_citation_quota")
    citation_rank = "—"
    if rank not in (None, ""):
        citation_rank = str(rank)
        if quota not in (None, ""):
            citation_rank += f" / top {quota}"

    model_status = paper.get("metadata_screening_status") or "not_screened"
    model_decision = paper.get("metadata_screening_decision") or "pending"
    model_reason = paper.get("metadata_screening_reason") or ""

    lines = [
        "",
        "=" * 100,
        f"Human screening  {index}/{total}   |   already reviewed: {reviewed}",
        "=" * 100,
        f"Corpus ID:       {_text(paper.get('corpus_id'))}",
        f"Year:            {_text(paper.get('year'))}",
        f"Citations:       {_text(paper.get('citation_count'))}",
        f"Citation rank:   {citation_rank}",
        f"Stream:          {_text(paper.get('candidate_selection_stream'))}",
        f"Selection:       {_text(paper.get('candidate_selection_reason'))}",
        f"Core protected:  {bool(paper.get('candidate_selection_core_protected'))}",
        f"Search topics:   {topics}",
        f"Sources:         {sources}",
        f"Type:            {_text(publication_type)}",
        f"Venue:           {_text(paper.get('venue'))}",
        f"DOI:             {_text(paper.get('doi'))}",
        f"URL:             {_text(paper.get('url'))}",
        "",
        "MODEL METADATA SCREEN",
        f"Status:          {model_status}",
        f"Decision:        {model_decision}",
    ]
    if model_reason:
        lines.extend([
            "Reason:",
            _wrap(model_reason),
        ])

    lines.extend([
        "",
        "TITLE",
        _wrap(paper.get("title")),
        "",
        "ABSTRACT",
        _wrap(paper.get("abstract")),
        "",
        "[y] include   [n] exclude   [s] skip for this session   [q] quit",
    ])
    return "\n".join(lines)


def select_for_human_screening(
    store,
    *,
    selected_only: bool = True,
    revisit: bool = False,
) -> tuple[list[dict], HumanScreenSummary]:
    papers = store.all_papers()

    if selected_only and store.latest_candidate_selection_run():
        papers = [
            paper
            for paper in papers
            if paper.get("candidate_selection_status") == "selected"
        ]

    reviewed = [
        paper
        for paper in papers
        if paper.get("human_metadata_screening_decision")
        in {"include", "exclude", "uncertain"}
    ]
    pending = (
        list(papers)
        if revisit
        else [
            paper
            for paper in papers
            if paper.get("human_metadata_screening_decision")
            not in {"include", "exclude", "uncertain"}
        ]
    )

    # Stable, auditable order. Core first, then historical citation selections,
    # then recent context; corpus_id breaks ties deterministically.
    reason_priority = {
        "core_protected": 0,
        "citation_top_n": 1,
        "recent_unfiltered": 2,
    }
    pending.sort(
        key=lambda paper: (
            reason_priority.get(
                paper.get("candidate_selection_reason"),
                9,
            ),
            int(float(paper.get("year") or 9999)),
            paper.get("candidate_selection_citation_rank")
            if paper.get("candidate_selection_citation_rank") not in (None, "")
            else 10**9,
            str(paper.get("corpus_id") or ""),
        )
    )

    summary = HumanScreenSummary(
        eligible_pool=len(papers),
        already_reviewed=len(reviewed),
        remaining=len(pending),
    )
    return pending, summary


def run_human_screening(
    store,
    *,
    limit: int | None = None,
    selected_only: bool = True,
    revisit: bool = False,
    input_fn=input,
    output_fn=print,
) -> HumanScreenSummary:
    """Interactively review candidates, committing every decision immediately."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("human screening limit must be a positive integer or null.")

    pending, summary = select_for_human_screening(
        store,
        selected_only=selected_only,
        revisit=revisit,
    )
    for session_index, paper in enumerate(pending, start=1):
        if limit is not None and summary.session_reviewed >= limit:
            break

        # Refresh just before display so concurrent metadata-model progress is
        # visible without rebuilding the human queue.
        paper = store.get_paper(paper["corpus_id"]) or paper

        output_fn(
            format_human_screen_paper(
                paper,
                index=session_index,
                total=len(pending),
                reviewed=summary.already_reviewed + summary.session_reviewed,
            )
        )

        while True:
            try:
                answer = input_fn("Decision [y/n/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                output_fn("\nStopping. All previous decisions are already saved.")
                summary.quit = True
                summary.remaining = max(
                    0,
                    len(pending) - summary.session_reviewed - summary.skipped,
                )
                return summary

            if answer in {"y", "yes"}:
                store.set_human_metadata_screening(
                    paper["corpus_id"],
                    decision="include",
                    reason="Human title/abstract screening: include.",
                    notes=paper.get("human_metadata_screening_notes", "") or "",
                )
                summary.session_reviewed += 1
                summary.included += 1
                break

            if answer in {"n", "no"}:
                store.set_human_metadata_screening(
                    paper["corpus_id"],
                    decision="exclude",
                    reason="Human title/abstract screening: exclude.",
                    notes=paper.get("human_metadata_screening_notes", "") or "",
                )
                summary.session_reviewed += 1
                summary.excluded += 1
                break

            if answer in {"s", "skip"}:
                summary.skipped += 1
                break

            if answer in {"q", "quit"}:
                summary.quit = True
                summary.remaining = max(
                    0,
                    len(pending) - summary.session_reviewed - summary.skipped,
                )
                return summary

            output_fn("Please enter y, n, s, or q.")

    summary.remaining = max(
        0,
        len(pending) - summary.session_reviewed,
    )
    return summary
