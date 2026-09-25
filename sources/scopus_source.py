"""Scopus Search adapter. Requires SCOPUS_API_KEY or api_key=.

Optional SCOPUS_INSTTOKEN supports institution-issued remote-access tokens.
Pass native queries (e.g. TITLE-ABS-KEY(deepfake AND adversarial)). STANDARD
may omit abstracts/full author lists; COMPLETE needs appropriate entitlement.
No silent view downgrade, abstract-retrieval calls or PDF downloads are made.
"""
from __future__ import annotations

from datetime import date
import os
import time
import warnings
from collections.abc import Iterator
from urllib.parse import parse_qs, urlsplit

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
    paper = base_paper(
        "scopus",
        identifier or eid,
        item.get("dc:title"),
        item.get("dc:description"),
        query,
    )
    authors = item.get("author") or []
    if isinstance(authors, dict):
        authors = [authors]
    names = [
        clean_text(a.get("authname") or a.get("ce:indexed-name"))
        for a in authors
        if isinstance(a, dict)
    ]
    publication_date = clean_text(item.get("prism:coverDate"))
    doi = doi_url(item.get("prism:doi"))
    links = item.get("link") or []
    if isinstance(links, dict):
        links = [links]
    url = next(
        (
            a.get("@href")
            for a in links
            if a.get("@ref") == "scopus" and a.get("@href")
        ),
        "",
    )
    aggregation = clean_text(item.get("prism:aggregationType"))
    oa = item.get("openaccess")
    paper.update(
        {
            "scopus_id": identifier,
            "eid": eid,
            "authors": ", ".join(n for n in names if n)
            or clean_text(item.get("dc:creator")),
            "year": number(publication_date[:4], None),
            "published_date": publication_date,
            "venue": clean_text(item.get("prism:publicationName")),
            "venue_type": {
                "Journal": "journal",
                "Conference Proceeding": "conference",
                "Book": "book",
                "Book Series": "book",
            }.get(aggregation, ""),
            "publication_type": clean_text(item.get("subtypeDescription"))
            or aggregation,
            "doi": doi,
            "url": url or doi or clean_text(item.get("prism:url")),
            "pdf_url": "",
            "is_open_access": (
                None
                if oa is None
                else str(oa).lower() in {"1", "true"}
            ),
            "citation_count": number(item.get("citedby-count")),
        }
    )
    return paper


def _parse_bound(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format.") from exc


def _publication_date_in_window(
    paper: dict,
    lower: date | None,
    upper: date | None,
) -> bool:
    if lower is None and upper is None:
        return True

    value = clean_text(paper.get("published_date"))
    try:
        publication_date = date.fromisoformat(value[:10])
    except ValueError as exc:
        raise RuntimeError(
            "Scopus: cannot enforce exact publication-date bounds because a "
            "record has a missing or non-ISO prism:coverDate."
        ) from exc

    if lower and publication_date < lower:
        return False
    if upper and publication_date > upper:
        return False
    return True


def _next_cursor(results: dict) -> str | None:
    """Read the Scopus deep-pagination cursor from either response form."""
    cursor = results.get("cursor")
    if isinstance(cursor, dict):
        value = cursor.get("@next")
        if value:
            return str(value)

    links = results.get("link") or []
    if isinstance(links, dict):
        links = [links]
    for link in links:
        if not isinstance(link, dict) or link.get("@ref") != "next":
            continue
        href = link.get("@href")
        if not href:
            continue
        values = parse_qs(urlsplit(href).query).get("cursor")
        if values and values[0]:
            return values[0]

    return None


def iter_scopus_pages(
    query: str,
    *,
    api_key: str | None = None,
    insttoken: str | None = None,
    max_results: int | None = None,
    page_size: int = 25,
    view: str = "STANDARD",
    start_year: int | None = None,
    end_year: int | None = None,
    from_publication_date: str | None = None,
    to_publication_date: str | None = None,
    pagination_mode: str = "auto",
    offset_page_size: int = 25,
    sleep_seconds: float = 1.0,
    timeout: float = 30,
    max_retries: int = 3,
    session: requests.Session | None = None,
    resume: dict | None = None,
) -> Iterator[dict]:
    """Yield auditable Scopus pages with exact local date cutoffs.

    Scopus Search exposes publication-year filtering in the query. The adapter
    uses those year bounds as an API prefilter, then applies exact inclusive
    YYYY-MM-DD bounds locally using prism:coverDate.

    Cursor pagination is preferred for stable forward deep pagination.
    pagination_mode='auto' tries cursor mode first and falls back to offset mode
    only if the first cursor request is rejected. Offset fallback uses the
    conservative offset_page_size rather than assuming the account accepts the
    documented STANDARD maximum. It retains the 5,000-source-record limit and
    fails if repeated records make completeness uncertain.
    """
    view = view.upper()
    if view not in {"STANDARD", "COMPLETE"}:
        raise ValueError("view must be STANDARD or COMPLETE.")

    pagination_mode = str(pagination_mode).lower()
    if pagination_mode not in {"auto", "cursor", "offset"}:
        raise ValueError("pagination_mode must be auto, cursor or offset.")
    validate(
        query,
        max_results,
        page_size,
        25 if view == "COMPLETE" else 200,
        start_year,
        end_year,
        sleep_seconds,
        timeout,
        max_retries,
    )
    if (
        isinstance(offset_page_size, bool)
        or not isinstance(offset_page_size, int)
        or offset_page_size < 1
        or offset_page_size > (25 if view == "COMPLETE" else 200)
    ):
        raise ValueError("Invalid offset_page_size.")

    lower = _parse_bound(from_publication_date, "from_publication_date")
    upper = _parse_bound(to_publication_date, "to_publication_date")
    if lower and upper and lower > upper:
        raise ValueError(
            "from_publication_date must not exceed to_publication_date."
        )

    # Keep the provider-side query as narrow as Scopus supports.
    if lower and start_year is None:
        start_year = lower.year
    if upper and end_year is None:
        end_year = upper.year

    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": get_key(api_key, "SCOPUS_API_KEY"),
    }
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

    base_params = {
        "query": effective_query,
        "view": view,
        # Scopus supports up to three sort keys. Secondary keys reduce
        # ambiguous page boundaries when many records share a cover date.
        "sort": "-coverDate,+artnum,+creator",
    }
    client = session if session is not None else requests.Session()
    resume = resume or {}
    if not isinstance(resume, dict):
        raise ValueError("Scopus resume state must be a mapping.")

    mode = resume.get(
        "mode",
        "cursor" if pagination_mode in {"auto", "cursor"} else "offset",
    )
    if mode not in {"cursor", "offset"}:
        raise ValueError("Scopus resume mode must be cursor or offset.")
    cursor = resume.get("cursor", "*")
    seen_cursors = set(resume.get("seen_cursors", []))
    if mode == "cursor" and cursor:
        seen_cursors.add(cursor)
    source_retrieved = int(resume.get("source_retrieved", 0))
    exact_retrieved = int(resume.get("exact_retrieved", 0))
    exact_seen = int(resume.get("exact_seen", 0))
    locally_excluded = int(resume.get("locally_excluded", 0))
    cap_trimmed = int(resume.get("cap_trimmed", 0))
    total = resume.get("total")
    seen_ids: set[str] = set(resume.get("seen_ids", []))
    fallback_used = bool(resume.get("fallback_used", False))

    try:
        while True:
            remaining = (
                None
                if max_results is None
                else max_results - exact_retrieved
            )
            if remaining is not None and remaining <= 0:
                return

            params = dict(base_params)

            if mode == "cursor":
                params["cursor"] = cursor
                params["count"] = (
                    page_size
                    if remaining is None
                    else min(page_size, remaining)
                )
            else:
                if source_retrieved >= 5000 and (
                    total is None or source_retrieved < total
                ):
                    raise RuntimeError(
                        "Scopus: offset fallback reached 5,000 source records "
                        "before the exact-date search completed. Use cursor "
                        "pagination with subscriber entitlement or narrow the "
                        "query/year range."
                    )
                params["start"] = source_retrieved
                params["count"] = min(
                    offset_page_size,
                    5000 - source_retrieved,
                    remaining if remaining is not None else offset_page_size,
                )
                if total is not None:
                    params["count"] = min(
                        params["count"],
                        total - source_retrieved,
                    )

            try:
                data = get_json(
                    client,
                    SCOPUS_API_URL,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    max_retries=max_retries,
                    source="Scopus",
                )
            except RuntimeError:
                if (
                    pagination_mode == "auto"
                    and mode == "cursor"
                    and source_retrieved == 0
                ):
                    warnings.warn(
                        "Scopus cursor pagination was rejected; falling back "
                        "to offset pagination (limited to 5,000 source records).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    mode = "offset"
                    fallback_used = True
                    continue
                raise

            results = data.get("search-results")
            if not isinstance(results, dict):
                raise RuntimeError(
                    "Scopus: missing search-results or API error; "
                    "search incomplete."
                )

            reported = number(results.get("opensearch:totalResults"), -1)
            if reported < 0:
                raise RuntimeError(
                    "Scopus: missing result total; search incomplete."
                )
            if total is not None and total != reported:
                raise RuntimeError(
                    "Scopus: result total changed during pagination; "
                    "rerun search."
                )
            total = reported

            if mode == "offset" and max_results is None and total > 5000:
                raise RuntimeError(
                    "Scopus: offset fallback exceeds 5,000 source records. "
                    "Use cursor pagination with subscriber entitlement or "
                    "narrow the query/year range."
                )

            entries = results.get("entry") or []
            if isinstance(entries, dict):
                entries = [entries]
            if total == 0:
                entries = []

            if (
                not isinstance(entries, list)
                or len(entries) > params["count"]
            ):
                raise RuntimeError(
                    "Scopus: invalid result page; search incomplete."
                )
            if not entries and source_retrieved < total:
                raise RuntimeError(
                    "Scopus: premature empty page; search incomplete."
                )
            if any(
                isinstance(item, dict) and "error" in item
                for item in entries
            ):
                raise RuntimeError(
                    "Scopus: API error entry; search incomplete."
                )

            source_papers = [_normalise(item, query) for item in entries]

            duplicate_ids = [
                paper["paper_id"]
                for paper in source_papers
                if paper["paper_id"] in seen_ids
            ]
            if duplicate_ids:
                raise RuntimeError(
                    f"Scopus: repeated record during {mode} pagination; "
                    "search completeness is uncertain."
                )
            seen_ids.update(paper["paper_id"] for paper in source_papers)
            source_retrieved += len(source_papers)

            valid = [
                paper
                for paper in source_papers
                if _publication_date_in_window(paper, lower, upper)
            ]
            locally_excluded += len(source_papers) - len(valid)
            exact_seen += len(valid)

            if remaining is None:
                papers = valid
            else:
                papers = valid[:remaining]
                cap_trimmed += max(0, len(valid) - len(papers))

            exact_retrieved += len(papers)

            provider_complete = source_retrieved >= total
            cap_reached = (
                max_results is not None
                and exact_retrieved >= max_results
                and (not provider_complete or cap_trimmed > 0)
            )

            complete = provider_complete and not cap_reached
            stop_reason = (
                "exhausted"
                if complete
                else "max_results"
                if cap_reached
                else "more_pages"
            )

            next_cursor = (
                _next_cursor(results)
                if mode == "cursor" and stop_reason == "more_pages"
                else None
            )
            if mode == "cursor" and stop_reason == "more_pages":
                if not next_cursor or next_cursor in seen_cursors:
                    raise RuntimeError(
                        "Scopus: cursor pagination ended early or repeated "
                        "its cursor; search incomplete."
                    )

            checkpoint_seen_cursors = set(seen_cursors)
            if next_cursor:
                checkpoint_seen_cursors.add(next_cursor)

            page = {
                "source": "scopus",
                "query": query,
                "request_params": dict(params),
                "raw_response": data,
                "papers": papers,
                "total_results": total,
                "source_retrieved_count": source_retrieved,
                "retrieved_count": exact_retrieved,
                "exact_records_seen": exact_seen,
                "locally_excluded_count": locally_excluded,
                "cap_trimmed_count": cap_trimmed,
                "complete": complete,
                "stop_reason": stop_reason,
                "pagination_mode": mode,
                "cursor_fallback_used": fallback_used,
                "exact_date_bounds": {
                    "from": lower.isoformat() if lower else None,
                    "to": upper.isoformat() if upper else None,
                },
                "exact_date_total": exact_seen if provider_complete else None,
                "checkpoint": {
                    "mode": mode,
                    "cursor": (
                        next_cursor
                        if mode == "cursor" and stop_reason == "more_pages"
                        else None
                    ),
                    "seen_cursors": sorted(checkpoint_seen_cursors),
                    "source_retrieved": source_retrieved,
                    "exact_retrieved": exact_retrieved,
                    "exact_seen": exact_seen,
                    "locally_excluded": locally_excluded,
                    "cap_trimmed": cap_trimmed,
                    "total": total,
                    "seen_ids": sorted(seen_ids),
                    "fallback_used": fallback_used,
                },
            }

            yield page

            if stop_reason != "more_pages":
                return

            if mode == "cursor":
                seen_cursors.add(next_cursor)
                cursor = next_cursor

            time.sleep(sleep_seconds)
    finally:
        if session is None:
            client.close()


def search_scopus(
    query: str,
    max_results: int | None = None,
    **kwargs,
) -> list[dict]:
    """Return pipeline-compatible records; warn when an explicit cap truncates."""
    papers = []
    for page in iter_scopus_pages(
        query,
        max_results=max_results,
        **kwargs,
    ):
        papers.extend(page["papers"])
        if page["stop_reason"] == "max_results":
            warnings.warn(
                "Scopus search capped before the exact-date result set was "
                "exhausted.",
                RuntimeWarning,
                stacklevel=2,
            )
    return papers
