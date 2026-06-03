# utils/deduplication.py

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from utils.text import normalize_key


def get_dedup_key(paper: Dict) -> str:
    for field in [
        "doi",
        "arxiv_id",
        "semantic_scholar_id",
        "openalex_id",
        "url",
        "title",
    ]:
        key = normalize_key(paper.get(field, ""))
        if key:
            return f"{field}:{key}"

    return ""


def deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for paper in papers:
        key = get_dedup_key(paper)

        if not key or key in seen:
            continue

        seen.add(key)
        unique.append(paper)

    return unique


def merge_with_existing(existing_df: pd.DataFrame, new_papers: List[Dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_papers)

    if existing_df.empty:
        combined = new_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    if combined.empty:
        return combined

    combined["_dedup_key"] = combined.apply(
        lambda row: get_dedup_key(row.to_dict()),
        axis=1,
    )

    combined = combined.drop_duplicates("_dedup_key", keep="first")
    combined = combined.drop(columns=["_dedup_key"])

    return combined