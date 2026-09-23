"""Scopus Search adapter. Requires SCOPUS_API_KEY or api_key=.

Optional SCOPUS_INSTTOKEN supports institution-issued remote-access tokens.
Pass native queries (e.g. TITLE-ABS-KEY(deepfake AND adversarial)). STANDARD
may omit abstracts/full author lists; COMPLETE needs appropriate entitlement.
No silent view downgrade, abstract-retrieval calls or PDF downloads are made.
"""
from __future__ import annotations

import os
import time
import warnings
from collections.abc import Iterator

import requests

from sources._api_common import api_key as get_key
from sources._api_common import base_paper, batch, doi_url, get_json, number, validate
from utils.text import clean_text

SCOPUS_API_URL = "https://api.elsevier.com/content/search/scopus"


def _normalise(item: dict, query: str) -> dict:
    identifier = clean_text(item.get("dc:identifier")).removeprefix("SCOPUS_ID:")
    eid = clean_text(item.get("eid"))
    if not identifier and not eid:
        raise RuntimeError("Scopus: record has no identifier; search incomplete.")
    paper = base_paper("scopus", identifier or eid, item.get("dc:title"), item.get("dc:description"), query)
    authors = item.get("author") or []
    if isinstance(authors, dict):
        authors = [authors]
    names = [clean_text(a.get("authname") or a.get("ce:indexed-name")) for a in authors if isinstance(a, dict)]
    date = clean_text(item.get("prism:coverDate"))
    doi = doi_url(item.get("prism:doi"))
    links = item.get("link") or []
    if isinstance(links, dict):
        links = [links]
    url = next((a.get("@href") for a in links if a.get("@ref") == "scopus" and a.get("@href")), "")
    aggregation = clean_text(item.get("prism:aggregationType"))
    oa = item.get("openaccess")
    paper.update({
        "scopus_id": identifier, "eid": eid,
        "authors": ", ".join(n for n in names if n) or clean_text(item.get("dc:creator")),
        "year": number(date[:4], None), "published_date": date,
        "venue": clean_text(item.get("prism:publicationName")),
        "venue_type": {"Journal": "journal", "Conference Proceeding": "conference", "Book": "book", "Book Series": "book"}.get(aggregation, ""),
        "publication_type": clean_text(item.get("subtypeDescription")) or aggregation,
        "doi": doi, "url": url or doi or clean_text(item.get("prism:url")),
        "pdf_url": "", "is_open_access": None if oa is None else str(oa).lower() in {"1", "true"},
        "citation_count": number(item.get("citedby-count")),
    })
    return paper


def iter_scopus_pages(
    query: str, *, api_key: str | None = None, insttoken: str | None = None,
    max_results: int | None = None, page_size: int = 25, view: str = "STANDARD",
    start_year: int | None = None, end_year: int | None = None,
    sleep_seconds: float = 1.0, timeout: float = 30, max_retries: int = 3,
    session: requests.Session | None = None,
) -> Iterator[dict]:
    """Yield raw/normalised pages with explicit completion status.

    Uses documented offset paging (maximum 5,000 accessible results per query).
    Uncapped queries exceeding that limit fail before returning a partial corpus:
    narrow the query/year range or implement entitlement-dependent cursor paging.
    Filters use inclusive publication years. Query text is not auto-translated.
    """
    view = view.upper()
    if view not in {"STANDARD", "COMPLETE"}:
        raise ValueError("view must be STANDARD or COMPLETE.")
    validate(query, max_results, page_size, 25 if view == "COMPLETE" else 200,
             start_year, end_year, sleep_seconds, timeout, max_retries)
    headers = {"Accept": "application/json", "X-ELS-APIKey": get_key(api_key, "SCOPUS_API_KEY")}
    token = insttoken or os.getenv("SCOPUS_INSTTOKEN")
    if token:
        headers["X-ELS-Insttoken"] = token
    effective_query = query
    if start_year is not None or end_year is not None:
        effective_query = f"({query})"
        if start_year is not None:
            effective_query += f" AND PUBYEAR > {start_year - 1}"
        if end_year is not None:
            effective_query += f" AND PUBYEAR < {end_year + 1}"
    params = {"query": effective_query, "view": view, "sort": "-coverDate"}
    client = session if session is not None else requests.Session()
    retrieved, total = 0, None
    seen = set()
    try:
        while True:
            params["start"] = retrieved
            params["count"] = page_size if max_results is None else min(page_size, max_results - retrieved)
            params["count"] = min(params["count"], 5000 - retrieved)
            if total is not None:
                params["count"] = min(params["count"], total - retrieved)
            data = get_json(client, SCOPUS_API_URL, params=params, headers=headers,
                            timeout=timeout, max_retries=max_retries, source="Scopus")
            results = data.get("search-results")
            if not isinstance(results, dict):
                raise RuntimeError("Scopus: missing search-results or API error; search incomplete.")
            reported = number(results.get("opensearch:totalResults"), -1)
            if reported < 0:
                raise RuntimeError("Scopus: missing result total; search incomplete.")
            if total is not None and total != reported:
                raise RuntimeError("Scopus: result total changed during pagination; rerun search.")
            total = reported
            if min(total, max_results if max_results is not None else total) > 5000:
                raise RuntimeError("Scopus: offset search exceeds 5,000 records. Narrow query/year range; no complete result can be returned.")
            entries = results.get("entry") or []
            if isinstance(entries, dict):
                entries = [entries]
            # Scopus represents an empty result as an error entry, with total=0.
            if total == 0:
                entries = []
            if not isinstance(entries, list) or len(entries) > params["count"]:
                raise RuntimeError("Scopus: invalid result page; search incomplete.")
            if not entries and retrieved < total:
                raise RuntimeError("Scopus: premature empty page; search incomplete.")
            if any("error" in item for item in entries):
                raise RuntimeError("Scopus: API error entry; search incomplete.")
            papers = [_normalise(item, query) for item in entries]
            for paper in papers:
                if paper["paper_id"] in seen:
                    raise RuntimeError("Scopus: repeated record during pagination; search incomplete.")
                seen.add(paper["paper_id"])
            retrieved += len(papers)
            page = batch("scopus", query, params, data, papers, total, retrieved, max_results)
            yield page
            if page["stop_reason"] != "more_pages":
                return
            time.sleep(sleep_seconds)
    finally:
        if session is None:
            client.close()


def search_scopus(query: str, max_results: int | None = None, **kwargs) -> list[dict]:
    """Return pipeline-compatible records; warns if max_results truncates matches."""
    papers = []
    for page in iter_scopus_pages(query, max_results=max_results, **kwargs):
        papers.extend(page["papers"])
        if page["stop_reason"] == "max_results":
            warnings.warn(f"Scopus search capped at {len(papers)} of {page['total_results']} matches.", RuntimeWarning, stacklevel=2)
    return papers
