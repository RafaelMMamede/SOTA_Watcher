"""Persistent screening queue over the SQLite corpus."""
from __future__ import annotations

from dataclasses import dataclass

from screening.ollama import get_model_digest, screening_signature
from screening.pipeline import _criteria_for_paper, screen_papers


RETRYABLE_STATUSES = {
    "error",
    "fulltext_unavailable",
}


@dataclass
class QueueSummary:
    selected: int = 0
    completed: int = 0
    errors: int = 0
    unavailable: int = 0
    stale: int = 0
    skipped_complete: int = 0
    skipped_retry_required: int = 0


def _expected_signature(paper, criteria, cfg, model_digest):
    pdf_sha = paper.get("pdf_sha256") or ""
    if not pdf_sha:
        return ""
    return screening_signature(criteria, cfg, pdf_sha, model_digest)


def select_for_screening(store, protocol, config, *, limit=None, retry=None):
    """Return a stable batch that advances past completed/error records."""
    cfg = config.get("screening", {})
    criteria = protocol.get("eligibility", {}).get("criteria", [])
    if not criteria:
        raise ValueError("Configure eligibility.criteria before screening.")

    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("screen limit must be a positive integer or null.")

    retry = set(retry or [])
    unknown = retry - RETRYABLE_STATUSES
    if unknown:
        raise ValueError(f"Unsupported retry statuses: {sorted(unknown)}")

    model_digest = get_model_digest(cfg)
    selected = []
    summary = QueueSummary()

    for paper in store.all_papers():
        active, _, _ = _criteria_for_paper(criteria, paper)
        if not active:
            continue

        status = paper.get("eligibility_status") or "not_screened"
        expected = _expected_signature(paper, active, cfg, model_digest)
        stored = paper.get("screening_signature") or ""

        if status == "screened":
            if expected and stored == expected:
                summary.skipped_complete += 1
                continue
            summary.stale += 1
        elif status in RETRYABLE_STATUSES:
            if status not in retry:
                summary.skipped_retry_required += 1
                continue
        elif status not in {"not_screened", "deferred", ""}:
            if status not in retry:
                summary.skipped_retry_required += 1
                continue

        selected.append((paper, active, model_digest))
        if limit is not None and len(selected) >= limit:
            break

    summary.selected = len(selected)
    return selected, summary


def screen_saved_corpus(
    store,
    protocol,
    config,
    *,
    limit=None,
    retry=None,
    audit=None,
):
    """Process one advancing batch and persist each paper immediately."""
    selected, summary = select_for_screening(
        store,
        protocol,
        config,
        limit=limit,
        retry=retry,
    )

    for paper, active_criteria, model_digest in selected:
        corpus_id = paper["corpus_id"]
        one_config = dict(config)
        screen_cfg = dict(config.get("screening", {}))
        screen_cfg["enabled"] = True
        screen_cfg["max_papers_per_run"] = None
        one_config["screening"] = screen_cfg

        # screen_papers may enrich metadata through resolution; use the same
        # stable corpus ID for its artifact folder.
        screen_papers([paper], protocol, one_config, audit=audit)

        # Merge any newly discovered identifiers/metadata before saving state.
        stable_id = store.upsert_paper(paper)
        if stable_id != corpus_id:
            corpus_id = stable_id

        signature = ""
        if paper.get("pdf_sha256"):
            active_criteria, _, _ = _criteria_for_paper(
                protocol.get("eligibility", {}).get("criteria", []),
                paper,
            )
            signature = screening_signature(
                active_criteria,
                screen_cfg,
                paper.get("pdf_sha256"),
                model_digest,
            )

        store.update_processing(
            corpus_id,
            paper,
            screening_signature=signature,
        )

        status = paper.get("eligibility_status")
        if status == "screened":
            summary.completed += 1
        elif status == "fulltext_unavailable":
            summary.unavailable += 1
        else:
            summary.errors += 1

    return summary
