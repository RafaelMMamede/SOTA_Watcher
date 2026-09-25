"""Resumable PDF retrieval and provisional eligibility for every candidate."""
import hashlib
from pathlib import Path
import time

from fulltext import resolve_paper, fetch_pdf, extract_pdf
from utils.deduplication import decode, get_dedup_key
from utils.eligibility import initialize_eligibility
from fulltext._common import read_json, write_json
from .ollama import screen_extraction


def _screening_topics(paper):
    topics = decode(paper.get("search_topics"), [])
    if not topics and paper.get("search_topic"):
        topics = [paper["search_topic"]]
    return [
        str(topic).strip()
        for topic in topics
        if str(topic).strip()
    ]


def _criteria_for_paper(criteria, paper):
    topics = set(_screening_topics(paper))
    active = []
    inactive = []
    for criterion in criteria:
        applies = criterion.get("applies_to_search_topics")
        skips = criterion.get("skip_if_search_topics")
        if skips is not None and topics.intersection(skips):
            inactive.append(criterion["id"])
        elif applies is None or topics.intersection(applies):
            active.append(criterion)
        else:
            inactive.append(criterion["id"])
    return active, inactive, sorted(topics)


def _resolution_fields(paper, resolution):
    paper["fulltext_resolver"] = (
        resolution.get("resolver")
        or resolution.get("kind")
        or ""
    )
    paper["fulltext_source"] = resolution.get("fulltext_source", "")
    paper["fulltext_version"] = resolution.get("version", "")
    paper["fulltext_license"] = resolution.get("license", "")
    paper["fulltext_url"] = resolution.get("url", "")
    paper["fulltext_resolved_via"] = resolution.get("resolved_via", "")
    paper["fulltext_resolved_doi"] = resolution.get("resolved_doi", "")
    paper["fulltext_resolved_openalex_id"] = resolution.get(
        "resolved_openalex_id",
        "",
    )
    paper["fulltext_manual_candidates"] = resolution.get(
        "manual_candidates",
        [],
    )
    paper["fulltext_resolver_attempts"] = resolution.get(
        "resolver_attempts",
        [],
    )


def screen_papers(papers, protocol, config, audit=None):
    cfg = config.get("screening", {})
    resolver_cfg = dict(config.get("fulltext_resolution", {}))
    resolver_cfg.setdefault("email", config.get("mailto"))
    criteria = protocol.get("eligibility", {}).get("criteria", [])
    enabled = cfg.get("enabled", False)
    limit = cfg.get("max_papers_per_run")

    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(
            "screening.max_papers_per_run must be positive or null."
        )

    for index, paper in enumerate(papers):
        initialize_eligibility(paper)

        if not enabled or (limit is not None and index >= limit):
            paper["eligibility_status"] = (
                "not_screened" if not enabled else "deferred"
            )
            paper["eligibility_reason"] = (
                "Screening disabled."
                if not enabled
                else "Per-run screening limit reached."
            )
            continue

        key = get_dedup_key(paper) or paper.get("title") or str(index)
        folder = (
            Path(cfg.get("papers_dir", "output/papers"))
            / hashlib.sha256(key.encode()).hexdigest()[:24]
        )
        paper["fulltext_folder"] = str(folder)
        folder.mkdir(parents=True, exist_ok=True)

        stage = "resolution"
        try:
            local = (
                config.get("local_pdfs", {}).get(paper.get("paper_id"))
                or config.get("local_pdfs", {}).get(paper.get("doi"))
            )
            resolution_path = folder / "resolution.json"

            # A newly supplied local PDF always wins over a cached remote
            # resolution. Otherwise reuse the resolver result unless explicitly
            # refreshed, avoiding repeated external API calls across runs.
            cached_resolution = (
                None
                if local or cfg.get("refresh_resolution", False)
                else read_json(resolution_path)
            )

            if (
                cached_resolution
                and cached_resolution.get("status")
                in {"resolved", "unavailable"}
            ):
                resolution = cached_resolution
                resolution_cache_hit = True
            else:
                resolution = resolve_paper(
                    paper,
                    local_pdf=local,
                    resolver_config=resolver_cfg,
                )
                resolution_cache_hit = False
                write_json(resolution_path, resolution)

            _resolution_fields(paper, resolution)
            paper["fulltext_resolution_cache_hit"] = resolution_cache_hit

            stage = "download"
            download = fetch_pdf(
                resolution,
                folder,
                refresh=cfg.get("refresh_pdf", False),
            )
            paper["fulltext_status"] = download["status"]

            if download["status"] != "downloaded":
                paper["eligibility_status"] = "fulltext_unavailable"
                paper["eligibility_reason"] = download.get(
                    "reason",
                    "Full text unavailable.",
                )
            else:
                # Preserve the existing polite arXiv pacing. Other OA resolver
                # URLs are hosted by their source repository/publisher.
                if (
                    resolution["kind"] == "arxiv"
                    and not download.get("cache_hit")
                ):
                    time.sleep(3)

                stage = "extraction"
                extraction = extract_pdf(folder / "paper.pdf", folder)
                paper["pdf_sha256"] = extraction["pdf_sha256"]
                paper["fulltext_status"] = extraction["status"]
                stage = "screening"
                active_criteria, inactive_criteria, screening_topics = (
                    _criteria_for_paper(criteria, paper)
                )
                paper["screening_search_topics"] = screening_topics
                paper["screening_inactive_criteria"] = inactive_criteria
                write_json(
                    folder / "screening_applicability.json",
                    {
                        "search_topics": screening_topics,
                        "active_criteria": [
                            criterion["id"] for criterion in active_criteria
                        ],
                        "inactive_criteria": inactive_criteria,
                    },
                )
                if not active_criteria:
                    raise ValueError(
                        "No eligibility criteria apply to this paper's "
                        "search-topic provenance."
                    )
                screened = screen_extraction(
                    extraction,
                    active_criteria,
                    cfg,
                    folder,
                )
                paper.update(screened)

        except Exception as exc:
            paper["eligibility_decision"] = "uncertain"
            paper["eligibility_status"] = "error"
            message = str(exc).strip()
            paper["eligibility_reason"] = (
                f"{type(exc).__name__} during {stage}: "
                + (
                    message
                    if message
                    else "full-text pipeline failed; inspect artifacts and retry."
                )
            )
            paper["eligibility_error_type"] = type(exc).__name__
            paper["eligibility_error_stage"] = stage
            paper["eligibility_error_message"] = message

        write_json(
            folder / "latest_eligibility.json",
            {
                k: v
                for k, v in paper.items()
                if k.startswith(
                    (
                        "eligibility_",
                        "fulltext_",
                        "screening_",
                        "pdf_",
                    )
                )
            },
        )

        if audit:
            audit.event(
                "eligibility_result",
                paper_id=paper.get("paper_id"),
                decision=paper["eligibility_decision"],
                status=paper["eligibility_status"],
                folder=str(folder),
                fulltext_resolver=paper.get("fulltext_resolver"),
                fulltext_status=paper.get("fulltext_status"),
            )

    return papers
