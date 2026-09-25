"""Persistent screening queue over the SQLite corpus."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

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
    pending_total: int = 0
    remaining_pending: int = 0
    elapsed_seconds: float = 0.0
    papers_per_minute: float = 0.0
    estimated_remaining_minutes: float | None = None


def _local_pdf_changed(paper, config):
    mapping = config.get("local_pdfs", {})
    path = None
    for key in (
        paper.get("corpus_id"),
        paper.get("paper_id"),
        paper.get("doi"),
    ):
        if key and key in mapping:
            path = Path(mapping[key])
            break
    if path is None or not path.exists() or not paper.get("pdf_sha256"):
        return False

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() != paper.get("pdf_sha256")


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
            forced_refresh = (
                cfg.get("force", False)
                or cfg.get("refresh_pdf", False)
                or cfg.get("refresh_resolution", False)
                or _local_pdf_changed(paper, config)
            )
            if not forced_refresh and expected and stored == expected:
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

        summary.pending_total += 1
        if limit is None or len(selected) < limit:
            selected.append((paper, active, model_digest))

    summary.selected = len(selected)
    summary.remaining_pending = max(0, summary.pending_total - summary.selected)
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
    started = time.monotonic()
    retry = set(retry or [])
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
        if (
            paper.get("eligibility_status") == "fulltext_unavailable"
            and "fulltext_unavailable" in retry
        ):
            screen_cfg["refresh_resolution"] = True
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

    summary.elapsed_seconds = round(time.monotonic() - started, 3)
    processed = summary.completed + summary.unavailable + summary.errors
    if processed and summary.elapsed_seconds > 0:
        summary.papers_per_minute = round(
            processed * 60.0 / summary.elapsed_seconds,
            3,
        )
        if summary.papers_per_minute > 0:
            summary.estimated_remaining_minutes = round(
                summary.remaining_pending / summary.papers_per_minute,
                2,
            )

    store.record_screening_batch(summary.__dict__)
    return summary
