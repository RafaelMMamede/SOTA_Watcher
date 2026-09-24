# sources/arxiv_source.py

from __future__ import annotations

import re
from datetime import date
import time
import urllib.parse
from typing import Dict, List

import feedparser
import requests

from utils.text import clean_text


ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _extract_arxiv_id(entry_id: str) -> str:
    # Example: http://arxiv.org/abs/2401.12345v2
    return entry_id.rstrip("/").split("/")[-1]


def _query_to_arxiv_syntax(query: str, field: str = "all") -> str:
    """
    Converts a human-readable query like:
        '"deepfake detection" adversarial'

    into:
        all:"deepfake detection" AND all:adversarial

    And:
        'deepfake detection'

    into:
        all:deepfake AND all:detection
    """

    phrases = re.findall(r'"([^"]+)"', query)
    query_without_phrases = re.sub(r'"[^"]+"', " ", query)

    terms = query_without_phrases.split()

    parts = []

    for phrase in phrases:
        phrase = phrase.strip()
        if phrase:
            parts.append(f'{field}:"{phrase}"')

    for term in terms:
        term = term.strip()
        if term:
            parts.append(f"{field}:{term}")

    if not parts:
        return f"{field}:{query}"

    return " AND ".join(parts)


def _fetch_arxiv_feed(
    url: str,
    max_retries: int = 5,
    timeout: int = 60,
) -> feedparser.FeedParserDict:
    headers = {
        "User-Agent": "sota-watcher/0.1 (mailto:rafaelcmpmamede@gmail.com)"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)

            print("Feed status:", response.status_code)

            if response.status_code == 200:
                return feedparser.parse(response.text)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after else 60 * (attempt + 1)

                print(
                    f"Rate limited by arXiv. "
                    f"Waiting {wait_seconds} seconds before retrying..."
                )
                time.sleep(wait_seconds)
                continue

            if 400 <= response.status_code < 500:
                raise RuntimeError(
                    f"arXiv: HTTP {response.status_code}; request rejected. "
                    "Not retrying this response automatically. "
                    "Check the exact query with a direct request; this status alone "
                    "does not establish whether the query or the service caused the rejection."
                )

            response.raise_for_status()

        except requests.exceptions.Timeout:
            wait_seconds = 60 * (attempt + 1)
            print(
                f"Request timed out. "
                f"Waiting {wait_seconds} seconds before retrying..."
            )
            time.sleep(wait_seconds)

        except requests.exceptions.RequestException as exc:
            wait_seconds = 60 * (attempt + 1)
            print(
                f"Request failed: {exc}. "
                f"Waiting {wait_seconds} seconds before retrying..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        "arXiv request failed after retries. "
        "This is likely temporary rate limiting or connectivity trouble."
    )

def search_arxiv(
    query: str,
    max_results: int = 25,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    sleep_seconds: float = 3.0,
    native_query: bool = False,
    from_publication_date: str | None = None,
    to_publication_date: str | None = None,
) -> List[Dict]:
    arxiv_query = query if native_query else _query_to_arxiv_syntax(query)
    if from_publication_date or to_publication_date:
        lower = date.fromisoformat(str(from_publication_date)) if from_publication_date else date(1991, 1, 1)
        upper = date.fromisoformat(str(to_publication_date)) if to_publication_date else date.today()
        if lower > upper:
            raise ValueError("from_publication_date must not exceed to_publication_date.")
        arxiv_query = f"({arxiv_query}) AND submittedDate:[{lower:%Y%m%d}0000 TO {upper:%Y%m%d}2359]"


    params = {
        "search_query": arxiv_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    print("\nOriginal query:", query)
    print("arXiv query:", arxiv_query)
    print("ARXIV URL:", url)

    feed = _fetch_arxiv_feed(url)

    print("Feed bozo:", getattr(feed, "bozo", None))
    print("Entries found:", len(feed.entries))

    papers = []

    for entry in feed.entries:
        arxiv_id = _extract_arxiv_id(entry.id)

        authors = ", ".join(author.name for author in entry.get("authors", []))

        doi = ""
        for link in entry.get("links", []):
            if link.get("title") == "doi":
                doi = link.get("href", "")

        paper = {
            "paper_id": f"arxiv:{arxiv_id}",
            "source": "arxiv",
            "title": clean_text(entry.get("title", "")),
            "authors": authors,
            "year": int(entry.published[:4]) if entry.get("published") else None,
            "published_date": entry.get("published", ""),
            "updated_date": entry.get("updated", ""),
            "venue": "arXiv",
            "venue_type": "repository",
            "is_repository": True,
            "openalex_type": "preprint",  # Compatibility with existing scoring/filtering.
            "has_abstract": bool(clean_text(entry.get("summary", ""))),
            "abstract": clean_text(entry.get("summary", "")),
            "url": entry.get("id", ""),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "query": query,
        }

        papers.append(paper)

    time.sleep(sleep_seconds)

    return papers