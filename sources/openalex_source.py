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


def _normalise(item, query):
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

    return paper


def iter_openalex_pages(query, max_results=None, mailto=None,
                        from_publication_date=None, to_publication_date=None,
                        sleep_seconds=1.0, page_size=100, timeout=30, max_retries=3,
                        session=None):
    import os
    from sources._api_common import get_json, validate, batch
    validate(query, max_results, page_size, 100, None, None, sleep_seconds, timeout, max_retries)
    filters = _build_date_filter(from_publication_date, to_publication_date)
    params = {'search': query, 'per-page': page_size, 'cursor': '*',
              'sort': 'publication_date:desc', 'filter': filters}
    if mailto:
        params['mailto'] = mailto
    client = session or requests.Session()
    seen_ids, seen_cursors = set(), {'*'}
    retrieved, total = 0, None
    try:
        while True:
            params['per-page'] = page_size if max_results is None else min(page_size, max_results - retrieved)
            request_params = dict(params)
            if os.getenv('OPENALEX_API_KEY'):
                request_params['api_key'] = os.environ['OPENALEX_API_KEY']
            data = get_json(client, OPENALEX_API_URL, params=request_params,
                            headers={'Accept':'application/json'}, timeout=timeout,
                            max_retries=max_retries, source='OpenAlex')
            reported = data.get('meta', {}).get('count')
            if not isinstance(reported, int) or reported < 0:
                raise RuntimeError('OpenAlex: missing result count.')
            if total is not None and reported != total:
                raise RuntimeError('OpenAlex: result count changed; rerun the search.')
            total = reported
            entries = data.get('results')
            if not isinstance(entries, list) or len(entries) > params['per-page']:
                raise RuntimeError('OpenAlex: invalid result page.')
            papers = [_normalise(item, query) for item in entries]
            for paper in papers:
                if not paper['openalex_id'] or paper['paper_id'] in seen_ids:
                    raise RuntimeError('OpenAlex: missing/repeated identifier.')
                seen_ids.add(paper['paper_id'])
            retrieved += len(papers)
            page = batch('openalex', query, params, data, papers, total, retrieved, max_results)
            cursor = data.get('meta', {}).get('next_cursor')
            if page['stop_reason'] == 'more_pages' and (not entries or not cursor or cursor in seen_cursors):
                raise RuntimeError('OpenAlex: pagination ended early or repeated its cursor.')
            yield page
            if page['stop_reason'] != 'more_pages':
                return
            seen_cursors.add(cursor)
            params['cursor'] = cursor
            time.sleep(sleep_seconds)
    finally:
        if session is None:
            client.close()


def search_openalex(query, max_results=None, **kwargs):
    import warnings
    papers = []
    for page in iter_openalex_pages(query, max_results=max_results, **kwargs):
        papers.extend(page['papers'])
        if not page['complete'] and page['stop_reason'] != 'more_pages':
            warnings.warn('OpenAlex search capped; retrieval is incomplete.', stacklevel=2)
    return papers
