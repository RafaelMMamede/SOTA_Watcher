# sources/openalex_source.py

from __future__ import annotations

import time
from datetime import date
from typing import Dict, List

import requests

from utils.text import clean_text


OPENALEX_API_URL = "https://api.openalex.org/works"


def _invert_abstract_index(abstract_inverted_index: dict | None) -> str:
    """
    OpenAlex stores abstracts as an inverted index:
        {
            "word": [position_1, position_2, ...]
        }

    This reconstructs the original abstract text.
    """
    if not abstract_inverted_index:
        return ""

    positions = []

    for word, word_positions in abstract_inverted_index.items():
        for pos in word_positions:
            positions.append((pos, word))

    positions.sort(key=lambda x: x[0])

    return " ".join(word for _, word in positions)


def _get_external_id(ids: dict | None, key: str) -> str:
    if not ids:
        return ""
    return ids.get(key, "") or ""


def _extract_authors(item: dict) -> str:
    authorships = item.get("authorships", []) or []

    authors = [
        authorship.get("author", {}).get("display_name", "")
        for authorship in authorships
    ]

    authors = [author for author in authors if author]

    return ", ".join(authors)


def _extract_best_venue(item: dict) -> tuple[str, str]:
    """
    Returns:
        venue_name, venue_type

    Prefer journals/conferences over repositories when possible.
    This avoids putting HAL/ORBi/etc. as the venue when a better source exists.
    """
    locations = item.get("locations", []) or []

    preferred_types = {"journal", "conference"}

    for location in locations:
        source = location.get("source", {}) or {}
        source_type = source.get("type", "") or ""
        source_name = source.get("display_name", "") or ""

        if source_type in preferred_types and source_name:
            return source_name, source_type

    primary_location = item.get("primary_location", {}) or {}
    source = primary_location.get("source", {}) or {}

    venue_name = source.get("display_name", "") or ""
    venue_type = source.get("type", "") or ""

    return venue_name, venue_type


def _build_date_filter(
    from_publication_date: str | None = None,
    to_publication_date: str | None = None,
) -> str:
    filters = []

    if from_publication_date:
        filters.append(f"from_publication_date:{from_publication_date}")

    # Avoid future-looking OpenAlex records unless the user explicitly sets a date.
    if to_publication_date:
        filters.append(f"to_publication_date:{to_publication_date}")
    else:
        filters.append(f"to_publication_date:{date.today().isoformat()}")

    return ",".join(filters)


def search_openalex(
    query: str,
    max_results: int = 25,
    mailto: str | None = None,
    from_publication_date: str | None = None,
    to_publication_date: str | None = None,
    sleep_seconds: float = 1.0,
) -> List[Dict]:
    filters = _build_date_filter(
        from_publication_date=from_publication_date,
        to_publication_date=to_publication_date,
    )

    params = {
        "search": query,
        "per-page": max_results,
        "sort": "publication_date:desc",
        "filter": filters,
    }

    if mailto:
        params["mailto"] = mailto

    response = requests.get(
        OPENALEX_API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()
    results = data.get("results", []) or []

    papers = []

    for item in results:
        ids = item.get("ids", {}) or {}

        doi = _get_external_id(ids, "doi")
        openalex_id = item.get("id", "") or ""

        title = clean_text(item.get("title", ""))
        abstract = clean_text(
            _invert_abstract_index(item.get("abstract_inverted_index"))
        )

        authors = _extract_authors(item)

        venue, venue_type = _extract_best_venue(item)

        paper = {
            "paper_id": f"openalex:{openalex_id}",
            "source": "openalex",
            "title": title,
            "authors": authors,
            "year": item.get("publication_year", None),
            "published_date": item.get("publication_date", ""),
            "updated_date": item.get("updated_date", ""),
            "venue": venue,
            "venue_type": venue_type,
            "abstract": abstract,
            "has_abstract": bool(abstract),
            "url": doi or openalex_id,
            "doi": doi,
            "arxiv_id": "",
            "semantic_scholar_id": "",
            "openalex_id": openalex_id,
            "query": query,
            "citation_count": item.get("cited_by_count", 0),
            "is_repository": venue_type == "repository",
            "openalex_type": item.get("type", ""),
            "openalex_crossref_type": item.get("type_crossref", ""),
        }

        papers.append(paper)

    print(f"OpenAlex query: {query}")
    print(f"Date filter: {filters}")
    print(f"Entries found: {len(papers)}")

    time.sleep(sleep_seconds)

    return papers