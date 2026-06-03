# summarize_recommended.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def priority_rank(priority: str) -> int:
    priority = str(priority).lower().strip()

    if priority == "read":
        return 0
    if priority == "skim":
        return 1
    return 2


def make_paper_block(row: pd.Series, index: int) -> str:
    title = clean_cell(row.get("title", ""))
    authors = clean_cell(row.get("authors", ""))
    year = clean_cell(row.get("year", ""))
    venue = clean_cell(row.get("venue", ""))
    published_date = clean_cell(row.get("published_date", ""))
    doi = clean_cell(row.get("doi", ""))
    url = clean_cell(row.get("url", ""))

    topic_label = clean_cell(row.get("topic_label", ""))
    triage_score = clean_cell(row.get("triage_score", ""))
    priority = clean_cell(row.get("llm_priority", ""))
    relevance = clean_cell(row.get("llm_relevance", ""))
    confidence = clean_cell(row.get("llm_confidence", ""))

    # Fast LLM triage fields
    summary = clean_cell(row.get("llm_summary", ""))
    main_contribution = clean_cell(row.get("llm_main_contribution", ""))
    method = clean_cell(row.get("llm_method", ""))
    datasets = clean_cell(row.get("llm_datasets_or_benchmarks", ""))
    reason = clean_cell(row.get("llm_reason", ""))
    key_terms = clean_cell(row.get("llm_key_terms", ""))
    relevance_to_review = clean_cell(row.get("llm_relevance_to_review", ""))

    # Deep reasoning-model fields
    deep_summary = clean_cell(row.get("deep_summary", ""))
    deep_problem = clean_cell(row.get("deep_problem", ""))
    deep_method = clean_cell(row.get("deep_method", ""))
    deep_datasets = clean_cell(row.get("deep_datasets", ""))
    deep_results = clean_cell(row.get("deep_results", ""))
    deep_limitations = clean_cell(row.get("deep_limitations", ""))
    deep_relevance = clean_cell(row.get("deep_relevance_to_review", ""))
    deep_reading_notes = clean_cell(row.get("deep_reading_notes", ""))
    deep_confidence = clean_cell(row.get("deep_confidence", ""))

    parts = []

    parts.append(f"### {index}. {title}")

    meta = []
    if authors:
        meta.append(f"**Authors:** {authors}")
    if year:
        meta.append(f"**Year:** {year}")
    if venue:
        meta.append(f"**Venue:** {venue}")
    if published_date:
        meta.append(f"**Published:** {published_date}")

    if meta:
        parts.append("\n".join(meta))

    parts.append(
        f"**Recommendation:** `{priority}`  \n"
        f"**Relevance:** `{relevance}`  \n"
        f"**Topic:** {topic_label}  \n"
        f"**Triage score:** {triage_score}  \n"
        f"**Fast LLM confidence:** {confidence}"
    )

    if summary:
        parts.append(f"**Fast summary:** {summary}")

    if main_contribution:
        parts.append(f"**Main contribution:** {main_contribution}")

    if method:
        parts.append(f"**Method:** {method}")

    if datasets:
        parts.append(f"**Datasets / benchmarks:** {datasets}")

    if key_terms:
        parts.append(f"**Key terms:** {key_terms}")

    if relevance_to_review:
        parts.append(f"**Fast relevance to review:** {relevance_to_review}")

    if reason:
        parts.append(f"**Why it was recommended:** {reason}")

    has_deep_analysis = any(
        [
            deep_summary,
            deep_problem,
            deep_method,
            deep_datasets,
            deep_results,
            deep_limitations,
            deep_relevance,
            deep_reading_notes,
        ]
    )

    if has_deep_analysis:
        deep_parts = ["#### Deep analysis"]

        if deep_confidence:
            deep_parts.append(f"**Deep-analysis confidence:** {deep_confidence}")

        if deep_summary:
            deep_parts.append(f"**Deep summary:** {deep_summary}")

        if deep_problem:
            deep_parts.append(f"**Problem / gap:** {deep_problem}")

        if deep_method:
            deep_parts.append(f"**Deep method analysis:** {deep_method}")

        if deep_datasets:
            deep_parts.append(f"**Deep datasets / benchmarks:** {deep_datasets}")

        if deep_results:
            deep_parts.append(f"**Reported results:** {deep_results}")

        if deep_limitations:
            deep_parts.append(f"**Limitations / caveats:** {deep_limitations}")

        if deep_relevance:
            deep_parts.append(f"**Deep relevance to review:** {deep_relevance}")

        if deep_reading_notes:
            deep_parts.append(f"**Reading notes:** {deep_reading_notes}")

        parts.append("\n\n".join(deep_parts))

    if doi:
        parts.append(f"**DOI:** {doi}")
    elif url:
        parts.append(f"**URL:** {url}")

    return "\n\n".join(parts)


def create_recommended_summary(
    input_path: str,
    output_path: str,
    include_priorities: list[str] | None = None,
) -> None:
    if include_priorities is None:
        include_priorities = ["read", "skim"]

    df = pd.read_excel(input_path)

    if "llm_priority" not in df.columns:
        raise ValueError("Column 'llm_priority' not found. Run Ollama triage first.")

    df["llm_priority_clean"] = (
        df["llm_priority"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    recommended = df[df["llm_priority_clean"].isin(include_priorities)].copy()

    if recommended.empty:
        print("No read/skim papers found.")
        return

    recommended["priority_rank"] = recommended["llm_priority_clean"].apply(priority_rank)

    sort_cols = []
    ascending = []

    if "priority_rank" in recommended.columns:
        sort_cols.append("priority_rank")
        ascending.append(True)

    if "triage_score" in recommended.columns:
        recommended["triage_score_numeric"] = pd.to_numeric(
            recommended["triage_score"],
            errors="coerce",
        ).fillna(0)
        sort_cols.append("triage_score_numeric")
        ascending.append(False)

    if "published_date" in recommended.columns:
        sort_cols.append("published_date")
        ascending.append(False)

    recommended = recommended.sort_values(by=sort_cols, ascending=ascending)

    n_read = (recommended["llm_priority_clean"] == "read").sum()
    n_skim = (recommended["llm_priority_clean"] == "skim").sum()

    has_deep_summary = (
        "deep_summary" in recommended.columns
        and recommended["deep_summary"].fillna("").astype(str).str.strip().ne("").any()
    )

    lines = []

    lines.append("# Recommended Papers Summary")
    lines.append("")
    lines.append(f"Total recommended papers: **{len(recommended)}**")
    lines.append(f"- Read: **{n_read}**")
    lines.append(f"- Skim: **{n_skim}**")
    lines.append(f"- Contains deep-analysis notes: **{'yes' if has_deep_summary else 'no'}**")
    lines.append("")

    if "topic_label" in recommended.columns:
        lines.append("## Counts by topic")
        lines.append("")
        topic_counts = recommended["topic_label"].fillna("Unknown").value_counts()

        for topic, count in topic_counts.items():
            lines.append(f"- {topic}: **{count}**")

        lines.append("")

    if "llm_relevance" in recommended.columns:
        lines.append("## Counts by LLM relevance")
        lines.append("")
        relevance_counts = recommended["llm_relevance"].fillna("Unknown").value_counts()

        for relevance, count in relevance_counts.items():
            lines.append(f"- {relevance}: **{count}**")

        lines.append("")

    lines.append("## Papers")
    lines.append("")

    for i, (_, row) in enumerate(recommended.iterrows(), start=1):
        lines.append(make_paper_block(row, i))
        lines.append("\n---\n")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved recommended summary to: {output_path}")
    print(f"Recommended papers: {len(recommended)}")
    print(f"Read: {n_read}")
    print(f"Skim: {n_skim}")
    print(f"Contains deep-analysis notes: {'yes' if has_deep_summary else 'no'}")


def main() -> None:
    config = load_config("config.yaml")

    input_path = config["sota_table_path"]
    output_path = config.get(
        "recommended_summary_path",
        "output/recommended_papers_summary.md",
    )

    create_recommended_summary(
        input_path=input_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()