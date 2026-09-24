from __future__ import annotations

import pandas as pd

from sources.discovery import fetch_papers, get_queries_from_search_terms
from utils.config import load_config, load_search_terms
from utils.io import make_output_dirs, load_existing_table, save_table
from utils.scoring import score_paper
from utils.deduplication import deduplicate_papers, merge_with_existing
from classify_with_ollama import classify_papers_with_ollama
from deep_analyze_with_ollama import deep_analyze_recommended_papers_with_ollama

def fetch_papers_from_openalex(config: dict, search_terms: dict) -> list[dict]:
    """Compatibility wrapper for callers that explicitly request OpenAlex."""
    return fetch_papers({**config, "sources": ["openalex"]}, search_terms)


def add_scores_and_manual_fields(
    papers: list[dict],
    config: dict,
    search_terms: dict,
) -> list[dict]:
    scoring_config = config.get("scoring", {})

    for paper in papers:
        scores = score_paper(
            paper=paper,
            search_terms=search_terms,
            scoring_config=scoring_config,
        )

        paper.update(scores)

        # Keep backward-compatible column if you want.
        paper["relevance_score"] = paper["triage_score"]

        paper.setdefault("manual_decision", "")
        paper.setdefault("notes", "")

    return papers


def filter_papers(papers: list[dict], config: dict) -> list[dict]:
    min_score = config.get("min_triage_score", 0)

    include_repositories = config.get("include_repositories", True)
    include_datasets = config.get("include_datasets", True)
    include_preprints = config.get("include_preprints", True)

    filtered = []

    for paper in papers:
        score = paper.get("triage_score", 0)

        if score < min_score:
            continue

        is_repository = bool(paper.get("is_repository", False))
        openalex_type = str(paper.get("openalex_type", "")).lower()

        if is_repository and not include_repositories:
            if include_datasets and openalex_type == "dataset":
                pass
            elif include_preprints and openalex_type == "preprint":
                pass
            else:
                continue

        filtered.append(paper)

    print(
        f"\nFiltering: kept {len(filtered)} / {len(papers)} papers "
        f"with triage_score >= {min_score}"
    )

    return filtered


def drop_low_score_existing_rows(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if not config.get("drop_existing_below_min_score", False):
        return df

    min_score = config.get("min_triage_score", 0)

    score_col = "triage_score" if "triage_score" in df.columns else "relevance_score"

    if score_col not in df.columns:
        return df

    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

    return df[scores >= min_score].copy()


def main() -> None:
    config = load_config("config.yaml")
    search_terms = load_search_terms(config.get("search_terms_path", "search_terms.yaml"))

    make_output_dirs(config)

    all_papers = fetch_papers(config, search_terms)

    unique_papers = deduplicate_papers(all_papers)

    print(f"\nRaw papers found: {len(all_papers)}")
    print(f"Unique papers after deduplication: {len(unique_papers)}")

    if not unique_papers:
        print("No papers found. Skipping table save.")
        return

    unique_papers = add_scores_and_manual_fields(
        unique_papers,
        config,
        search_terms,
    )

    unique_papers = filter_papers(unique_papers, config)

    if not unique_papers:
        print("No papers passed the filter. Skipping table save.")
        return

    unique_papers = classify_papers_with_ollama(unique_papers, config)
    unique_papers = deep_analyze_recommended_papers_with_ollama(unique_papers, config)

    existing_df = load_existing_table(config["sota_table_path"])
    final_df = merge_with_existing(existing_df, unique_papers)

    final_df = drop_low_score_existing_rows(final_df, config)

    save_table(final_df, config["sota_table_path"])

    print(f"\nSaved table to: {config['sota_table_path']}")
    print(f"Total rows in table: {len(final_df)}")


if __name__ == "__main__":
    main()
