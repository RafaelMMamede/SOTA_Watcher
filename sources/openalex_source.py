# sources/openalex_source.py

from __future__ import annotations

import json
import re
import time
from datetime import date
from typing import Dict, List

import requests

from utils.text import clean_text


OPENALEX_API_URL = "https://api.openalex.org/"

_QUERY_TOKEN_RE = re.compile(
    r'"[^"]*"|\(|\)|\bNOT\b|\bAND\b|\bOR\b|[^\s()]+',
    flags=re.IGNORECASE,
)
_OPERATORS = {"AND", "OR", "NOT"}
_PRECEDENCE = {"OR": 1, "AND": 2, "NOT": 3}


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


def _parse_iso_date(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format.") from exc


def _tokenize_query(query: str) -> list[str]:
    tokens = [token.strip() for token in _QUERY_TOKEN_RE.findall(query) if token.strip()]
    if not tokens:
        raise ValueError("OpenAlex query must not be empty.")

    result: list[str] = []
    previous_kind: str | None = None

    for token in tokens:
        upper = token.upper()
        if upper in _OPERATORS:
            kind = "not" if upper == "NOT" else "operator"
        elif token == "(":
            kind = "lparen"
        elif token == ")":
            kind = "rparen"
        else:
            kind = "term"

        # Classic OpenAlex search treats adjacent words as AND. Make that
        # relationship explicit before translating into OQO.
        if previous_kind in {"term", "rparen"} and kind in {"term", "lparen", "not"}:
            result.append("AND")

        result.append(upper if upper in _OPERATORS else token)
        previous_kind = kind

    return result


def _to_postfix(tokens: list[str]) -> list[str]:
    output: list[str] = []
    operators: list[str] = []

    for token in tokens:
        if token == "(":
            operators.append(token)
            continue

        if token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if not operators:
                raise ValueError("Unmatched ')' in OpenAlex query.")
            operators.pop()
            continue

        if token in _OPERATORS:
            # NOT is unary and right-associative.
            while (
                operators
                and operators[-1] in _OPERATORS
                and (
                    _PRECEDENCE[operators[-1]] > _PRECEDENCE[token]
                    or (
                        token != "NOT"
                        and _PRECEDENCE[operators[-1]] == _PRECEDENCE[token]
                    )
                )
            ):
                output.append(operators.pop())
            operators.append(token)
            continue

        output.append(token)

    while operators:
        operator = operators.pop()
        if operator == "(":
            raise ValueError("Unmatched '(' in OpenAlex query.")
        output.append(operator)

    return output


def _search_leaf(token: str) -> dict:
    quoted = len(token) >= 2 and token.startswith('"') and token.endswith('"')
    value = token[1:-1] if quoted else token
    if not value:
        raise ValueError("OpenAlex query contains an empty term.")
    if not quoted and ("*" in value or "?" in value):
        raise ValueError(
            "OpenAlex wildcards must be quoted when using title/abstract search."
        )
    return {
        "column_id": (
            "title_and_abstract.search.exact"
            if quoted
            else "title_and_abstract.search"
        ),
        "value": value,
        "operator": "has",
    }


def _negate_filter(node: dict) -> dict:
    node = dict(node)
    node["is_negated"] = not bool(node.get("is_negated", False))
    return node


def _branch(join: str, left: dict, right: dict) -> dict:
    filters = []
    for node in (left, right):
        if (
            node.get("join") == join
            and not node.get("is_negated", False)
        ):
            filters.extend(node["filters"])
        else:
            filters.append(node)
    return {"join": join, "filters": filters}


def _query_filter(query: str) -> dict:
    stack: list[dict] = []

    for token in _to_postfix(_tokenize_query(query)):
        if token not in _OPERATORS:
            stack.append(_search_leaf(token))
            continue

        if token == "NOT":
            if not stack:
                raise ValueError(f"Malformed OpenAlex query near NOT: {query!r}")
            stack.append(_negate_filter(stack.pop()))
            continue

        if len(stack) < 2:
            raise ValueError(f"Malformed OpenAlex query near {token!r}: {query!r}")
        right = stack.pop()
        left = stack.pop()
        stack.append(_branch(token.lower(), left, right))

    if len(stack) != 1:
        raise ValueError(f"Malformed OpenAlex query: {query!r}")
    return stack[0]


def _build_oqo(
    query: str,
    from_publication_date: str | None = None,
    to_publication_date: str | None = None,
) -> dict:
    lower = _parse_iso_date(from_publication_date, "from_publication_date")
    upper = _parse_iso_date(to_publication_date, "to_publication_date")
    if lower and upper and lower > upper:
        raise ValueError("from_publication_date must not exceed to_publication_date.")

    # Avoid future-looking OpenAlex records unless an upper bound was supplied.
    upper = upper or date.today()

    filters = [_query_filter(query)]
    if lower:
        filters.append(
            {
                "column_id": "publication_date",
                "value": lower.isoformat(),
                "operator": ">=",
            }
        )
    filters.append(
        {
            "column_id": "publication_date",
            "value": upper.isoformat(),
            "operator": "<=",
        }
    )
    return {"get_rows": "works", "filter_rows": filters}


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


def iter_openalex_pages(
    query,
    max_results=None,
    mailto=None,
    from_publication_date=None,
    to_publication_date=None,
    sleep_seconds=1.0,
    page_size=100,
    timeout=30,
    max_retries=3,
    session=None,
):
    import os
    from sources._api_common import get_json, validate, batch

    validate(
        query,
        max_results,
        page_size,
        100,
        None,
        None,
        sleep_seconds,
        timeout,
        max_retries,
    )
    oqo = _build_oqo(
        query,
        from_publication_date=from_publication_date,
        to_publication_date=to_publication_date,
    )
    params = {
        "oqo": json.dumps(oqo, ensure_ascii=False, separators=(",", ":")),
        "per-page": page_size,
        "cursor": "*",
        "sort": "publication_date:desc",
    }
    headers = {"Accept": "application/json"}
    if mailto:
        headers["User-Agent"] = f"sota-watcher/0.1 (mailto:{mailto})"

    client = session or requests.Session()
    seen_ids, seen_cursors = set(), {"*"}
    retrieved, total = 0, None
    try:
        while True:
            params["per-page"] = (
                page_size
                if max_results is None
                else min(page_size, max_results - retrieved)
            )
            request_params = dict(params)
            if os.getenv("OPENALEX_API_KEY"):
                request_params["api_key"] = os.environ["OPENALEX_API_KEY"]
            data = get_json(
                client,
                OPENALEX_API_URL,
                params=request_params,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                source="OpenAlex",
            )
            reported = data.get("meta", {}).get("count")
            if not isinstance(reported, int) or reported < 0:
                raise RuntimeError("OpenAlex: missing result count.")
            if total is not None and reported != total:
                raise RuntimeError("OpenAlex: result count changed; rerun the search.")
            total = reported
            entries = data.get("results")
            if not isinstance(entries, list) or len(entries) > params["per-page"]:
                raise RuntimeError("OpenAlex: invalid result page.")
            papers = [_normalise(item, query) for item in entries]
            for paper in papers:
                if not paper["openalex_id"] or paper["paper_id"] in seen_ids:
                    raise RuntimeError("OpenAlex: missing/repeated identifier.")
                seen_ids.add(paper["paper_id"])
            retrieved += len(papers)
            page = batch(
                "openalex",
                query,
                params,
                data,
                papers,
                total,
                retrieved,
                max_results,
            )
            page["search_scope"] = "title_abstract"
            cursor = data.get("meta", {}).get("next_cursor")
            if page["stop_reason"] == "more_pages" and (
                not entries or not cursor or cursor in seen_cursors
            ):
                raise RuntimeError(
                    "OpenAlex: pagination ended early or repeated its cursor."
                )
            yield page
            if page["stop_reason"] != "more_pages":
                return
            seen_cursors.add(cursor)
            params["cursor"] = cursor
            time.sleep(sleep_seconds)
    finally:
        if session is None:
            client.close()


def search_openalex(query, max_results=None, **kwargs):
    import warnings

    papers = []
    for page in iter_openalex_pages(query, max_results=max_results, **kwargs):
        papers.extend(page["papers"])
        if not page["complete"] and page["stop_reason"] != "more_pages":
            warnings.warn(
                "OpenAlex search capped; retrieval is incomplete.",
                stacklevel=2,
            )
    return papers
