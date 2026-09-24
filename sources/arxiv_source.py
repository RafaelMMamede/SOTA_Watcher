# sources/arxiv_source.py

from __future__ import annotations

import re
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Dict, List

import requests

from utils.text import clean_text


ARXIV_OAI_URL = "https://oaipmh.arxiv.org/oai"

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
ARXIV_RAW_NS = "http://arxiv.org/OAI/arXivRaw/"
NS = {
    "oai": OAI_NS,
    "arxivraw": ARXIV_RAW_NS,
}

_QUERY_TOKEN_RE = re.compile(
    r'(?:all|ti|abs):"[^"]*"|"[^"]*"|\(|\)|\bANDNOT\b|\bAND\b|\bOR\b|[^\s()]+',
    flags=re.IGNORECASE,
)
_OPERATORS = {"AND", "OR", "ANDNOT"}
_PRECEDENCE = {"OR": 1, "AND": 2, "ANDNOT": 2}


def _parse_version_date(value: str) -> date | None:
    value = clean_text(value)
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError, OverflowError):
        return None


def _tokenize_query(query: str) -> list[str]:
    tokens = [token.strip() for token in _QUERY_TOKEN_RE.findall(query) if token.strip()]
    if not tokens:
        return []

    with_implicit_and: list[str] = []
    previous_kind: str | None = None

    for token in tokens:
        upper = token.upper()
        if upper in _OPERATORS:
            kind = "operator"
        elif token == "(":
            kind = "lparen"
        elif token == ")":
            kind = "rparen"
        else:
            kind = "term"

        if previous_kind in {"term", "rparen"} and kind in {"term", "lparen"}:
            with_implicit_and.append("AND")

        with_implicit_and.append(upper if kind == "operator" else token)
        previous_kind = kind

    return with_implicit_and


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
                raise ValueError("Unmatched ')' in arXiv query.")
            operators.pop()
            continue

        if token in _OPERATORS:
            while (
                operators
                and operators[-1] in _OPERATORS
                and _PRECEDENCE[operators[-1]] >= _PRECEDENCE[token]
            ):
                output.append(operators.pop())
            operators.append(token)
            continue

        output.append(token)

    while operators:
        operator = operators.pop()
        if operator == "(":
            raise ValueError("Unmatched '(' in arXiv query.")
        output.append(operator)

    return output


def _term_matches(term: str, title: str, abstract: str) -> bool:
    field = "all"
    value = term

    field_match = re.match(r"^(all|ti|abs):(.*)$", term, flags=re.IGNORECASE)
    if field_match:
        field = field_match.group(1).lower()
        value = field_match.group(2)

    value = value.strip().strip('"').lower()
    if not value:
        return False

    if field == "ti":
        haystack = title.lower()
    elif field == "abs":
        haystack = abstract.lower()
    else:
        haystack = f"{title} {abstract}".lower()

    return value in haystack


def _matches_query(title: str, abstract: str, query: str) -> bool:
    tokens = _tokenize_query(query)
    if not tokens:
        return False

    postfix = _to_postfix(tokens)
    stack: list[bool] = []

    for token in postfix:
        if token not in _OPERATORS:
            stack.append(_term_matches(token, title, abstract))
            continue

        if len(stack) < 2:
            raise ValueError(f"Malformed arXiv query near {token!r}: {query!r}")

        right = stack.pop()
        left = stack.pop()

        if token == "AND":
            stack.append(left and right)
        elif token == "OR":
            stack.append(left or right)
        else:
            stack.append(left and not right)

    if len(stack) != 1:
        raise ValueError(f"Malformed arXiv query: {query!r}")

    return stack[0]


def _parse_iso_date(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format.") from exc


def _request_oai(
    params: dict,
    *,
    timeout: int = 60,
    max_retries: int = 5,
    sleep_seconds: float = 3.0,
) -> ET.Element:
    headers = {
        "User-Agent": "sota-watcher/0.1 (mailto:rafaelcmpmamede@gmail.com)",
        "Accept": "text/xml",
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                ARXIV_OAI_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            print("arXiv OAI status:", response.status_code)

            if response.status_code == 200:
                try:
                    return ET.fromstring(response.content)
                except ET.ParseError as exc:
                    raise RuntimeError("arXiv OAI-PMH returned invalid XML.") from exc

            if response.status_code in {429, 503}:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = (
                        float(retry_after)
                        if retry_after
                        else sleep_seconds * (attempt + 1)
                    )
                except ValueError:
                    wait_seconds = sleep_seconds * (attempt + 1)

                print(
                    f"arXiv OAI-PMH is temporarily unavailable/rate limited. "
                    f"Waiting {wait_seconds:g} seconds before retrying..."
                )
                time.sleep(wait_seconds)
                continue

            if 400 <= response.status_code < 500:
                raise RuntimeError(
                    f"arXiv OAI-PMH: HTTP {response.status_code}; request rejected."
                )

            response.raise_for_status()

        except requests.exceptions.Timeout:
            wait_seconds = sleep_seconds * (attempt + 1)
            print(
                f"arXiv OAI-PMH request timed out. "
                f"Waiting {wait_seconds:g} seconds before retrying..."
            )
            time.sleep(wait_seconds)

        except requests.exceptions.RequestException as exc:
            wait_seconds = sleep_seconds * (attempt + 1)
            print(
                f"arXiv OAI-PMH request failed: {exc}. "
                f"Waiting {wait_seconds:g} seconds before retrying..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("arXiv OAI-PMH request failed after retries.")


def _record_to_paper(
    record: ET.Element,
    *,
    query: str,
    lower_publication_date: date | None,
    upper_publication_date: date | None,
) -> Dict | None:
    header = record.find("oai:header", NS)
    if header is None or header.get("status") == "deleted":
        return None

    metadata = record.find("oai:metadata/arxivraw:arXivRaw", NS)
    if metadata is None:
        return None

    arxiv_id = clean_text(metadata.findtext("arxivraw:id", default="", namespaces=NS))
    title = clean_text(metadata.findtext("arxivraw:title", default="", namespaces=NS))
    abstract = clean_text(metadata.findtext("arxivraw:abstract", default="", namespaces=NS))
    authors = clean_text(metadata.findtext("arxivraw:authors", default="", namespaces=NS))
    doi = clean_text(metadata.findtext("arxivraw:doi", default="", namespaces=NS))

    if not arxiv_id:
        return None

    versions: list[tuple[int, date]] = []
    for version in metadata.findall("arxivraw:version", NS):
        version_name = version.get("version", "")
        match = re.fullmatch(r"v(\d+)", version_name)
        version_date = _parse_version_date(
            version.findtext("arxivraw:date", default="", namespaces=NS)
        )
        if match and version_date:
            versions.append((int(match.group(1)), version_date))

    if not versions:
        return None

    versions.sort(key=lambda item: item[0])
    created_date = versions[0][1]
    updated_date = versions[-1][1]

    if lower_publication_date and created_date < lower_publication_date:
        return None
    if upper_publication_date and created_date > upper_publication_date:
        return None

    if not _matches_query(title, abstract, query):
        return None

    return {
        "paper_id": f"arxiv:{arxiv_id}",
        "source": "arxiv",
        "title": title,
        "authors": authors,
        "year": created_date.year,
        "published_date": created_date.isoformat(),
        "updated_date": updated_date.isoformat(),
        "venue": "arXiv",
        "venue_type": "repository",
        "is_repository": True,
        "openalex_type": "preprint",
        "has_abstract": bool(abstract),
        "abstract": abstract,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "doi": doi,
        "arxiv_id": arxiv_id,
        "query": query,
    }


def search_arxiv(
    query: str,
    max_results: int = 25,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    sleep_seconds: float = 3.0,
    native_query: bool = False,
    from_publication_date: str | None = None,
    to_publication_date: str | None = None,
    oai_from_date: str | None = None,
    oai_until_date: str | None = None,
) -> List[Dict]:
    """
    Discover arXiv records through OAI-PMH and apply the query locally.\n\n    OAI-PMH from/until filters record modification datestamps, not original\n    submission dates. arXivRaw version history is used to recover the v1\n    submission date and the most recent version date.

    This is a good fit for recurring monitoring. Historical publication-date
    backfills can be incomplete if a record was modified after the requested
    publication window.
    """
    del native_query  # Kept for compatibility; Boolean forms are parsed locally.

    lower = _parse_iso_date(from_publication_date, "from_publication_date")
    upper = _parse_iso_date(to_publication_date, "to_publication_date")
    if lower and upper and lower > upper:
        raise ValueError("from_publication_date must not exceed to_publication_date.")

    harvest_lower = _parse_iso_date(
        oai_from_date or from_publication_date,
        "oai_from_date",
    )
    harvest_upper = _parse_iso_date(
        oai_until_date or to_publication_date,
        "oai_until_date",
    )
    if harvest_lower and harvest_upper and harvest_lower > harvest_upper:
        raise ValueError("oai_from_date must not exceed oai_until_date.")

    if harvest_lower is None:
        raise ValueError(
            "arXiv OAI-PMH requires from_publication_date or "
            "source_options.arxiv.oai_from_date to avoid unintentionally "
            "harvesting the entire repository."
        )

    if (
        (from_publication_date or to_publication_date)
        and oai_from_date is None
        and oai_until_date is None
    ):
        warnings.warn(
            "arXiv now uses OAI-PMH discovery. Its from/until window is based "
            "on record modification dates, while published_date is filtered "
            "locally using the original <created> date. Historical backfills "
            "may therefore be incomplete for papers modified outside the "
            "requested window.",
            stacklevel=2,
        )

    print("\nOriginal query:", query)
    print("arXiv backend: OAI-PMH")
    print(
        "OAI harvest window:",
        harvest_lower.isoformat(),
        "to",
        harvest_upper.isoformat() if harvest_upper else "latest",
    )

    initial_params = {
        "verb": "ListRecords",
        "metadataPrefix": "arXivRaw",
        "from": harvest_lower.isoformat(),
    }
    if harvest_upper:
        initial_params["until"] = harvest_upper.isoformat()

    papers: list[Dict] = []
    params = initial_params

    while True:
        root = _request_oai(params, sleep_seconds=sleep_seconds)

        error = root.find("oai:error", NS)
        if error is not None:
            code = error.get("code", "unknown")
            message = clean_text(error.text or "")
            if code == "noRecordsMatch":
                break
            raise RuntimeError(f"arXiv OAI-PMH error {code}: {message}")

        records = root.findall("oai:ListRecords/oai:record", NS)

        for record in records:
            paper = _record_to_paper(
                record,
                query=query,
                lower_publication_date=lower,
                upper_publication_date=upper,
            )
            if paper is not None:
                papers.append(paper)
                if len(papers) >= max_results:
                    break

        if len(papers) >= max_results:
            break

        token_element = root.find(
            "oai:ListRecords/oai:resumptionToken",
            NS,
        )
        resumption_token = (
            clean_text(token_element.text or "")
            if token_element is not None
            else ""
        )

        if not resumption_token:
            break

        time.sleep(sleep_seconds)
        params = {
            "verb": "ListRecords",
            "resumptionToken": resumption_token,
        }

    reverse = sort_order.lower() != "ascending"
    if sort_by == "lastUpdatedDate":
        papers.sort(
            key=lambda paper: paper.get("updated_date", ""),
            reverse=reverse,
        )
    else:
        papers.sort(
            key=lambda paper: paper.get("published_date", ""),
            reverse=reverse,
        )

    time.sleep(sleep_seconds)
    return papers[:max_results]
