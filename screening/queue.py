"""Persistent screening queue over the SQLite corpus."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

from screening.metadata import metadata_screening_signature
from screening.metadata_queue import metadata_config
from screening.ollama import get_model_digest, screening_signature
from screening.pipeline import _criteria_for_paper, screen_papers


RETRY_KEYS = {
    "error",
    "fulltext_unavailable",
    "resolution",
    "download",
    "extraction",
    "screening",
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
    skipped_metadata_pending: int = 0
    skipped_metadata_excluded: int = 0
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
    unknown = retry - RETRY_KEYS
    if unknown:
        raise ValueError(f"Unsupported retry statuses: {sorted(unknown)}")

    model_digest = get_model_digest(cfg)
    require_metadata = cfg.get("require_metadata_screening", True)
    metadata_cfg = metadata_config(config) if require_metadata else None
    metadata_model_digest = (
        get_model_digest(metadata_cfg) if require_metadata else None
    )
    selected = []
    summary = QueueSummary()

    for paper in store.all_papers():
        active, _, _ = _criteria_for_paper(criteria, paper)
        if not active:
            continue

        manual = paper.get("manual_decision") or ""
        if manual == "exclude":
            summary.skipped_metadata_excluded += 1
            continue

        if require_metadata and manual != "include":
            metadata_status = (
                paper.get("metadata_screening_status") or "not_screened"
            )
            metadata_decision = (
                paper.get("metadata_screening_decision") or ""
            )
            expected_metadata_signature = metadata_screening_signature(
                paper,
                active,
                metadata_cfg,
                metadata_model_digest,
            )
            stored_metadata_signature = (
                paper.get("metadata_screening_signature") or ""
            )

            if (
                metadata_status != "screened"
                or stored_metadata_signature != expected_metadata_signature
            ):
                summary.skipped_metadata_pending += 1
                continue

            if metadata_decision == "exclude":
                summary.skipped_metadata_excluded += 1
                continue

            if metadata_decision not in {"include", "uncertain"}:
                summary.skipped_metadata_pending += 1
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
        elif status == "fulltext_unavailable":
            if "fulltext_unavailable" not in retry:
                summary.skipped_retry_required += 1
                continue
        elif status == "error":
            stage = paper.get("eligibility_error_stage") or ""
            if "error" not in retry and stage not in retry:
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

        # Metadata enrichment and the new screening signature/result commit
        # together, so a crash cannot pair a new result with an old signature.
        corpus_id = store.save_screened_paper(
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
