"""Shared transport and metadata helpers for the IEEE/Scopus adapters."""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from utils.text import clean_text


def api_key(value: str | None, variable: str) -> str:
    key = (value or os.getenv(variable, "")).strip()
    if not key:
        raise ValueError(f"Provide api_key or set {variable}.")
    return key


def validate(query, max_results, page_size, page_limit, start_year, end_year,
             sleep_seconds, timeout, max_retries):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty source-native query.")
    for name, value, low, high in (
        ("page_size", page_size, 1, page_limit),
        ("max_retries", max_retries, 0, 20),
        ("max_results", max_results, 1, None),
        ("start_year", start_year, 1, 9999),
        ("end_year", end_year, 1, 9999),
    ):
        if value is None and name in {"max_results", "start_year", "end_year"}:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < low or (high and value > high):
            raise ValueError(f"Invalid {name}.")
    if start_year and end_year and start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    if sleep_seconds < 0 or timeout <= 0:
        raise ValueError("sleep_seconds must be nonnegative and timeout positive.")


def get_json(session, url, *, params, headers, timeout, max_retries, source):
    """Bounded retries; never expose request URLs/headers containing API keys."""
    for attempt in range(max_retries + 1):
        retry_after = None
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException:
            if attempt == max_retries:
                raise RuntimeError(f"{source}: network request failed; search incomplete.") from None
        else:
            status = response.status_code
            retry_after = response.headers.get("Retry-After")
            if status == 200:
                try:
                    data = response.json()
                except ValueError:
                    raise RuntimeError(f"{source}: invalid JSON response; search incomplete.") from None
                if not isinstance(data, dict):
                    raise RuntimeError(f"{source}: expected a JSON object; search incomplete.")
                return data
            if status in (401, 403):
                raise RuntimeError(
                    f"{source}: HTTP {status}; check API key, institutional access and requested view."
                )
            if status not in (429, 500, 502, 503, 504) or attempt == max_retries:
                raise RuntimeError(f"{source}: HTTP {status}; search incomplete.")
        delay = min(2 ** attempt, 60)
        if retry_after:
            try:
                delay = max(0, float(retry_after))
            except ValueError:
                try:
                    delay = max(0, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
            if delay > 60:
                raise RuntimeError(
                    f"{source}: server requests a retry after {delay:.0f}s; rerun later. Search incomplete."
                )
        time.sleep(delay)


def number(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def doi_url(value):
    doi = clean_text(value)
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.I)
    return f"https://doi.org/{doi.lower()}" if doi else ""


def base_paper(source, identifier, title, abstract, query):
    abstract = clean_text(abstract)
    return {
        "paper_id": f"{source}:{identifier}", "source": source,
        "title": clean_text(title), "abstract": abstract, "has_abstract": bool(abstract),
        "query": query, "arxiv_id": "", "openalex_id": "", "semantic_scholar_id": "",
        "scopus_id": "", "ieee_id": "", "updated_date": "", "is_repository": False,
        "openalex_type": "", "openalex_crossref_type": "",
    }


def batch(source, query, params, data, papers, total, retrieved, max_results):
    complete = retrieved >= total
    capped = max_results is not None and retrieved >= max_results and not complete
    return {
        "source": source, "query": query, "request_params": dict(params),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "raw_response": data, "papers": papers, "total_results": total,
        "retrieved_count": retrieved, "complete": complete,
        "stop_reason": "exhausted" if complete else "max_results" if capped else "more_pages",
    }
