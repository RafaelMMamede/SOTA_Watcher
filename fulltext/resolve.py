"""Resolve reproducible local/arXiv PDFs and legitimate open full-text copies."""
from __future__ import annotations

import ipaddress
import os
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

ARXIV_ID = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.\-]*/\d{7})(?:v[1-9]\d*)?"
)
DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.I)

OPENALEX_API = "https://api.openalex.org"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
CROSSREF_API = "https://api.crossref.org/works"

AUTO_DOWNLOAD_RESOLVERS = {"openalex", "unpaywall", "semantic_scholar"}


def arxiv_id(value):
    value = str(value or "").strip()
    if value.startswith("arxiv:"):
        value = value[6:]
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.hostname not in {
            "arxiv.org",
            "www.arxiv.org",
            "export.arxiv.org",
        }:
            return None
        value = re.sub(r"^/(?:abs|pdf)/", "", parsed.path)
    value = value.removesuffix(".pdf")
    return value if ARXIV_ID.fullmatch(value) else None


def normalize_doi(value):
    value = str(value or "").strip()
    value = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        value,
        flags=re.I,
    )
    value = value.rstrip(" .;,)")
    return value.lower() if DOI_RE.fullmatch(value) else None


def _safe_https_url(value):
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
        ):
            return None
    return parsed.geturl()


def _normalise_title(value):
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", value)).strip()


def _author_surnames(value):
    words = re.findall(r"[\w'\-]+", _normalise_title(value))
    return {word for word in words if len(word) > 2}


def _request_json(
    client,
    url,
    *,
    params=None,
    headers=None,
    timeout=20,
    max_retries=2,
):
    for attempt in range(max_retries + 1):
        try:
            response = client.get(
                url,
                params=params,
                headers=headers or {},
                timeout=timeout,
            )
        except requests.RequestException:
            if attempt == max_retries:
                return None, "network_error"
            time.sleep(min(2 ** attempt, 5))
            continue

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                return None, "invalid_json"
            return (data, "ok") if isinstance(data, dict) else (None, "invalid_json")

        if response.status_code == 404:
            return None, "not_found"

        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(2 ** attempt, 5)
            except (TypeError, ValueError):
                delay = min(2 ** attempt, 5)
            if delay > 30:
                return None, f"http_{response.status_code}"
            time.sleep(max(0, delay))
            continue

        return None, f"http_{response.status_code}"

    return None, "network_error"


def _openalex_headers(email):
    headers = {"Accept": "application/json"}
    if email:
        headers["User-Agent"] = f"SOTA-Watcher/0.1 (mailto:{email})"
    return headers


def _openalex_params(email):
    params = {}
    key = os.getenv("OPENALEX_API_KEY", "").strip()
    if key:
        params["api_key"] = key
    if email:
        params["mailto"] = email
    return params


def _openalex_identifier(paper, doi):
    value = str(paper.get("openalex_id") or "").strip()
    if value:
        key = value.rstrip("/").split("/")[-1]
        if re.fullmatch(r"W\d+", key, flags=re.I):
            return key, "openalex_id"
    if doi:
        return f"https://doi.org/{doi}", "doi"
    return None, None


def _openalex_location_candidate(work):
    locations = []
    best = work.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    for location in work.get("locations") or []:
        if isinstance(location, dict) and location not in locations:
            locations.append(location)

    version_rank = {
        "publishedVersion": 3,
        "acceptedVersion": 2,
        "submittedVersion": 1,
    }
    locations.sort(
        key=lambda loc: (
            bool(loc.get("is_oa")),
            bool(loc.get("pdf_url")),
            version_rank.get(loc.get("version"), 0),
        ),
        reverse=True,
    )

    for location in locations:
        if location.get("is_oa") is not True:
            continue
        pdf_url = _safe_https_url(location.get("pdf_url"))
        if not pdf_url:
            continue
        source = location.get("source") or {}
        return {
            "status": "resolved",
            "kind": "remote",
            "resolver": "openalex",
            "is_oa": True,
            "url": pdf_url,
            "landing_page_url": _safe_https_url(
                location.get("landing_page_url")
            ),
            "version": location.get("version"),
            "license": location.get("license"),
            "fulltext_source": source.get("display_name") or "",
            "fulltext_source_type": source.get("type") or "",
        }
    return None


def _openalex_lookup(client, paper, doi, *, email, timeout, retries):
    identifier, matched_via = _openalex_identifier(paper, doi)
    if not identifier:
        return None, {"resolver": "openalex", "status": "no_identifier"}, None

    url = f"{OPENALEX_API}/works/{quote(identifier, safe=':/')}"
    work, status = _request_json(
        client,
        url,
        params=_openalex_params(email),
        headers=_openalex_headers(email),
        timeout=timeout,
        max_retries=retries,
    )
    if not work:
        return None, {"resolver": "openalex", "status": status}, None

    candidate = _openalex_location_candidate(work)
    enriched = {
        "doi": normalize_doi(work.get("doi")),
        "openalex_id": str(work.get("id") or "").rstrip("/").split("/")[-1],
    }
    if candidate:
        candidate["resolved_via"] = matched_via
        candidate["resolved_doi"] = enriched["doi"]
        candidate["resolved_openalex_id"] = enriched["openalex_id"]
        return candidate, {"resolver": "openalex", "status": "resolved"}, enriched

    return (
        None,
        {"resolver": "openalex", "status": "no_open_pdf"},
        enriched,
    )


def _openalex_title_lookup(client, paper, *, email, timeout, retries):
    title = str(paper.get("title") or "").strip()
    if not title:
        return None, {"resolver": "openalex_title", "status": "no_title"}, None

    params = {
        **_openalex_params(email),
        "search": title,
        "per-page": 10,
    }
    data, status = _request_json(
        client,
        f"{OPENALEX_API}/works",
        params=params,
        headers=_openalex_headers(email),
        timeout=timeout,
        max_retries=retries,
    )
    if not data:
        return None, {"resolver": "openalex_title", "status": status}, None

    expected_title = _normalise_title(title)
    expected_year = paper.get("year")
    try:
        expected_year = int(float(expected_year)) if expected_year not in {None, ""} else None
    except (TypeError, ValueError):
        expected_year = None
    paper_author_words = _author_surnames(paper.get("authors"))

    matches = []
    for work in data.get("results") or []:
        if _normalise_title(work.get("title")) != expected_title:
            continue
        if expected_year is not None and work.get("publication_year") != expected_year:
            continue
        candidate_authors = [
            (authorship.get("author") or {}).get("display_name", "")
            for authorship in work.get("authorships") or []
            if isinstance(authorship, dict)
        ]
        candidate_words = _author_surnames(" ".join(candidate_authors))
        if paper_author_words and candidate_words and not (
            paper_author_words & candidate_words
        ):
            continue
        matches.append(work)

    if len(matches) != 1:
        return (
            None,
            {
                "resolver": "openalex_title",
                "status": "ambiguous" if matches else "no_match",
                "matches": len(matches),
            },
            None,
        )

    work = matches[0]
    enriched = {
        "doi": normalize_doi(work.get("doi")),
        "openalex_id": str(work.get("id") or "").rstrip("/").split("/")[-1],
    }
    candidate = _openalex_location_candidate(work)
    if candidate:
        candidate["resolved_via"] = "title_year_author"
        candidate["resolved_doi"] = enriched["doi"]
        candidate["resolved_openalex_id"] = enriched["openalex_id"]
        return (
            candidate,
            {"resolver": "openalex_title", "status": "resolved"},
            enriched,
        )
    return (
        None,
        {"resolver": "openalex_title", "status": "matched_no_open_pdf"},
        enriched,
    )


def _unpaywall_lookup(client, doi, *, email, timeout, retries):
    if not doi:
        return None, {"resolver": "unpaywall", "status": "no_doi"}
    if not email:
        return None, {"resolver": "unpaywall", "status": "no_email"}

    data, status = _request_json(
        client,
        f"{UNPAYWALL_API}/{quote(doi, safe='/')}",
        params={"email": email},
        headers={"Accept": "application/json"},
        timeout=timeout,
        max_retries=retries,
    )
    if not data:
        return None, {"resolver": "unpaywall", "status": status}
    if data.get("is_oa") is not True:
        return None, {"resolver": "unpaywall", "status": "not_oa"}

    locations = []
    best = data.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    for location in data.get("oa_locations") or []:
        if isinstance(location, dict) and location not in locations:
            locations.append(location)

    for location in locations:
        pdf_url = _safe_https_url(location.get("url_for_pdf"))
        if not pdf_url:
            continue
        return (
            {
                "status": "resolved",
                "kind": "remote",
                "resolver": "unpaywall",
                "is_oa": True,
                "url": pdf_url,
                "landing_page_url": _safe_https_url(
                    location.get("url_for_landing_page")
                ),
                "version": location.get("version"),
                "license": location.get("license"),
                "fulltext_source": location.get("host_type") or "",
                "resolved_via": "doi",
                "resolved_doi": doi,
            },
            {"resolver": "unpaywall", "status": "resolved"},
        )

    return None, {"resolver": "unpaywall", "status": "oa_without_pdf"}


def _semantic_scholar_lookup(client, doi, *, timeout, retries):
    if not doi:
        return None, {"resolver": "semantic_scholar", "status": "no_doi"}

    headers = {"Accept": "application/json"}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key

    data, status = _request_json(
        client,
        f"{SEMANTIC_SCHOLAR_API}/paper/{quote('DOI:' + doi, safe=':/')}",
        params={"fields": "title,year,externalIds,openAccessPdf,url"},
        headers=headers,
        timeout=timeout,
        max_retries=retries,
    )
    if not data:
        return None, {"resolver": "semantic_scholar", "status": status}

    oa = data.get("openAccessPdf")
    if not isinstance(oa, dict):
        return None, {"resolver": "semantic_scholar", "status": "no_open_pdf"}
    pdf_url = _safe_https_url(oa.get("url"))
    if not pdf_url:
        return None, {"resolver": "semantic_scholar", "status": "no_open_pdf"}

    return (
        {
            "status": "resolved",
            "kind": "remote",
            "resolver": "semantic_scholar",
            "is_oa": True,
            "url": pdf_url,
            "landing_page_url": _safe_https_url(data.get("url")),
            "version": None,
            "license": oa.get("license"),
            "fulltext_source": "Semantic Scholar openAccessPdf",
            "resolved_via": "doi",
            "resolved_doi": doi,
            "semantic_scholar_id": data.get("paperId"),
        },
        {"resolver": "semantic_scholar", "status": "resolved"},
    )


def _crossref_candidates(client, doi, *, email, timeout, retries):
    if not doi:
        return [], {"resolver": "crossref", "status": "no_doi"}

    params = {"mailto": email} if email else None
    data, status = _request_json(
        client,
        f"{CROSSREF_API}/{quote(doi, safe='/')}",
        params=params,
        headers={"Accept": "application/json"},
        timeout=timeout,
        max_retries=retries,
    )
    if not data:
        return [], {"resolver": "crossref", "status": status}

    message = data.get("message") or {}
    candidates = []
    for link in message.get("link") or []:
        if not isinstance(link, dict):
            continue
        url = _safe_https_url(link.get("URL"))
        if not url:
            continue
        candidates.append(
            {
                "resolver": "crossref",
                "url": url,
                "content_type": link.get("content-type"),
                "content_version": link.get("content-version"),
                "intended_application": link.get("intended-application"),
                "auto_download": False,
            }
        )

    return (
        candidates,
        {
            "resolver": "crossref",
            "status": "manual_candidates" if candidates else "no_links",
            "count": len(candidates),
        },
    )


def resolve_paper(
    paper=None,
    *,
    local_pdf=None,
    resolver_config=None,
    session=None,
):
    """Resolve the best legitimate full-text source for a paper.

    Resolution order:
      local file -> arXiv -> OpenAlex OA -> Unpaywall -> Semantic Scholar.
    If identifiers are sparse, an exact-title/year/author-checked OpenAlex match
    can enrich DOI/OpenAlex identifiers before the DOI resolvers run.

    Crossref full-text/TDM links are retained only as manual candidates because
    their presence does not establish open access.
    """
    paper = paper or {}
    cfg = resolver_config or {}

    if local_pdf is not None:
        path = Path(local_pdf).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return {
            "status": "resolved",
            "kind": "local",
            "path": str(path),
            "resolver": "local",
            "resolved_via": "manual_mapping",
            "is_oa": None,
            "resolver_attempts": [],
            "manual_candidates": [],
        }

    for field in ("pdf_url", "arxiv_id", "paper_id", "url"):
        identifier = arxiv_id(paper.get(field))
        if identifier:
            return {
                "status": "resolved",
                "kind": "arxiv",
                "arxiv_id": identifier,
                "version_pinned": bool(re.search(r"v\d+$", identifier)),
                "url": f"https://arxiv.org/pdf/{identifier}",
                "resolver": "arxiv",
                "resolved_via": field,
                "is_oa": True,
                "resolver_attempts": [],
                "manual_candidates": [],
            }

    if cfg.get("enabled", True) is False:
        return {
            "status": "unavailable",
            "reason": "Network full-text resolution disabled.",
            "resolver_attempts": [],
            "manual_candidates": [],
        }

    timeout = float(cfg.get("timeout_seconds", 20))
    retries = int(cfg.get("max_retries", 2))
    if timeout <= 0 or retries < 0 or retries > 10:
        raise ValueError("Invalid fulltext_resolution timeout/retry settings.")

    email = (
        str(cfg.get("email") or "").strip()
        or os.getenv("UNPAYWALL_EMAIL", "").strip()
    )
    client = session or requests.Session()
    attempts = []
    manual_candidates = []
    enriched = {
        "doi": normalize_doi(paper.get("doi")),
        "openalex_id": None,
    }

    try:
        if cfg.get("openalex", True):
            candidate, attempt, found = _openalex_lookup(
                client,
                paper,
                enriched["doi"],
                email=email,
                timeout=timeout,
                retries=retries,
            )
            attempts.append(attempt)
            if found:
                enriched.update({k: v for k, v in found.items() if v})
            if candidate:
                candidate["resolver_attempts"] = attempts
                candidate["manual_candidates"] = manual_candidates
                return candidate

            if cfg.get("allow_title_fallback", True) and not (
                enriched.get("doi") or enriched.get("openalex_id")
            ):
                candidate, attempt, found = _openalex_title_lookup(
                    client,
                    paper,
                    email=email,
                    timeout=timeout,
                    retries=retries,
                )
                attempts.append(attempt)
                if found:
                    enriched.update({k: v for k, v in found.items() if v})
                if candidate:
                    candidate["resolver_attempts"] = attempts
                    candidate["manual_candidates"] = manual_candidates
                    return candidate

        doi = enriched.get("doi")

        if cfg.get("unpaywall", True):
            candidate, attempt = _unpaywall_lookup(
                client,
                doi,
                email=email,
                timeout=timeout,
                retries=retries,
            )
            attempts.append(attempt)
            if candidate:
                candidate["resolved_openalex_id"] = enriched.get("openalex_id")
                candidate["resolver_attempts"] = attempts
                candidate["manual_candidates"] = manual_candidates
                return candidate

        if cfg.get("semantic_scholar", True):
            candidate, attempt = _semantic_scholar_lookup(
                client,
                doi,
                timeout=timeout,
                retries=retries,
            )
            attempts.append(attempt)
            if candidate:
                candidate["resolved_openalex_id"] = enriched.get("openalex_id")
                candidate["resolver_attempts"] = attempts
                candidate["manual_candidates"] = manual_candidates
                return candidate

        if cfg.get("crossref_metadata", True):
            candidates, attempt = _crossref_candidates(
                client,
                doi,
                email=email,
                timeout=timeout,
                retries=retries,
            )
            attempts.append(attempt)
            manual_candidates.extend(candidates)

        return {
            "status": "unavailable",
            "reason": "No verified open PDF found through configured resolvers.",
            "resolved_doi": enriched.get("doi"),
            "resolved_openalex_id": enriched.get("openalex_id"),
            "resolver_attempts": attempts,
            "manual_candidates": manual_candidates,
        }
    finally:
        if session is None:
            client.close()
