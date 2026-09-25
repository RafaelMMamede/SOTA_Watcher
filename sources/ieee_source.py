"""IEEE Xplore metadata discovery. Requires IEEE_API_KEY or api_key=.

Queries use IEEE API syntax, not Scopus syntax. No PDF download is performed.
Use iter_ieee_pages for raw responses/provenance, search_ieee for list[dict].
Publication filters are inclusive YEARS, not insertion dates.
"""
from __future__ import annotations

import re
import time
import warnings
from collections.abc import Iterator

import requests

from sources._api_common import api_key as get_key
from sources._api_common import base_paper, batch, doi_url, get_json, number, validate
from utils.text import clean_text

IEEE_API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"


def _normalise(item: dict, query: str) -> dict:
    identifier = clean_text(item.get("article_number"))
    if not identifier:
        raise RuntimeError("IEEE: record has no article_number; search incomplete.")
    paper = base_paper("ieee", identifier, item.get("title"), item.get("abstract"), query)
    authors = item.get("authors") or {}
    authors = authors.get("authors", []) if isinstance(authors, dict) else authors
    if isinstance(authors, dict):
        authors = [authors]
    content_type = clean_text(item.get("content_type"))
    year = re.search(r"\b\d{4}\b", clean_text(item.get("publication_year")))
    doi = doi_url(item.get("doi"))
    paper.update({
        "ieee_id": identifier,
        "authors": ", ".join(clean_text(a.get("full_name")) for a in authors if isinstance(a, dict) and a.get("full_name")),
        "year": int(year.group()) if year else None,
        "published_date": clean_text(item.get("publication_date")),
        "updated_date": clean_text(item.get("insert_date")),
        "venue": clean_text(item.get("publication_title")),
        "venue_type": {"Conferences": "conference", "Journals": "journal", "Magazines": "journal", "Books": "book"}.get(content_type, ""),
        "publication_type": content_type,
        "doi": doi,
        "url": clean_text(item.get("abstract_url")) or doi or f"https://ieeexplore.ieee.org/document/{identifier}",
        "pdf_url": clean_text(item.get("pdf_url")),
        "html_url": clean_text(item.get("html_url")),
        "access_type": clean_text(item.get("accessType")),
        "is_open_access": clean_text(item.get("accessType")).lower() == "open access",
        "citation_count": number(item.get("citing_paper_count")),
    })
    return paper


def iter_ieee_pages(
    query: str, *, api_key: str | None = None, max_results: int | None = None,
    page_size: int = 200, start_year: int | None = None, end_year: int | None = None,
    sleep_seconds: float = 1.0, timeout: float = 30, max_retries: int = 3,
    session: requests.Session | None = None, resume: dict | None = None,
) -> Iterator[dict]:
    """Yield auditable pages. None retrieves all matches; a cap is marked incomplete.

    Raises on failed/malformed/repeated pages; never silently returns a partial
    successful search. Previously yielded pages can be persisted by the caller.
    request_params excludes the API key. Session ownership stays with the caller.
    """
    validate(query, max_results, page_size, 200, start_year, end_year, sleep_seconds, timeout, max_retries)
    wildcard_words = re.findall(r"[^\s()\"]*\*[^\s()\"]*", query)
    if len(wildcard_words) > 2 or any(len(w.split("*", 1)[0]) < 3 for w in wildcard_words):
        raise ValueError("IEEE permits at most two wildcard words, each with at least three characters before '*'. Expand terms or split the query.")
    resume = resume or {}
    if not isinstance(resume, dict):
        raise ValueError("IEEE resume state must be a mapping.")
    key = get_key(api_key, "IEEE_API_KEY")
    client = session if session is not None else requests.Session()
    params = {"querytext": query, "format": "json", "sort_field": "article_number", "sort_order": "asc"}
    if start_year is not None:
        params["start_year"] = start_year
    if end_year is not None:
        params["end_year"] = end_year
    retrieved = int(resume.get("retrieved", 0))
    total = resume.get("total")
    seen = set(resume.get("seen_ids", []))
    try:
        while True:
            params["start_record"] = retrieved + 1
            params["max_records"] = page_size if max_results is None else min(page_size, max_results - retrieved)
            data = get_json(client, IEEE_API_URL, params={**params, "apikey": key}, headers={"Accept": "application/json"}, timeout=timeout, max_retries=max_retries, source="IEEE")
            reported = number(data.get("total_records", data.get("totalfound")), -1)
            if reported < 0:
                raise RuntimeError("IEEE: missing result total or API error; search incomplete.")
            if total is not None and total != reported:
                raise RuntimeError("IEEE: result total changed during pagination; rerun search.")
            total = reported
            entries = data.get("articles") or []
            if not isinstance(entries, list) or len(entries) > params["max_records"]:
                raise RuntimeError("IEEE: invalid article page; search incomplete.")
            if not entries and retrieved < total:
                raise RuntimeError("IEEE: premature empty page; search incomplete.")
            papers = [_normalise(item, query) for item in entries]
            for paper in papers:
                if paper["paper_id"] in seen:
                    raise RuntimeError("IEEE: repeated record during pagination; search incomplete.")
                seen.add(paper["paper_id"])
            retrieved += len(papers)
            page = batch("ieee", query, params, data, papers, total, retrieved, max_results)
            page["checkpoint"] = {
                "retrieved": retrieved,
                "total": total,
                "seen_ids": sorted(seen),
            }
            yield page
            if page["stop_reason"] != "more_pages":
                return
            time.sleep(sleep_seconds)
    finally:
        if session is None:
            client.close()


def search_ieee(query: str, max_results: int | None = None, **kwargs) -> list[dict]:
    """Return pipeline-compatible records; warns if max_results truncates matches."""
    papers = []
    for page in iter_ieee_pages(query, max_results=max_results, **kwargs):
        papers.extend(page["papers"])
        if page["stop_reason"] == "max_results":
            warnings.warn(f"IEEE search capped at {len(papers)} of {page['total_results']} matches.", RuntimeWarning, stacklevel=2)
    return papers
