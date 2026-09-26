"""Persistent title/abstract screening queue."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from screening.metadata import (
    metadata_screening_signature,
    screen_metadata_paper,
)
from screening.ollama import get_model_digest
from screening.pipeline import _criteria_for_paper


@dataclass
class MetadataQueueSummary:
    selected: int = 0
    completed: int = 0
    included: int = 0
    excluded: int = 0
    uncertain: int = 0
    errors: int = 0
    stale: int = 0
    skipped_complete: int = 0
    skipped_retry_required: int = 0
    pending_total: int = 0
    remaining_pending: int = 0
    elapsed_seconds: float = 0.0
    papers_per_minute: float = 0.0
    estimated_remaining_minutes: float | None = None


def metadata_config(config):
    """Build metadata-screening settings without inheriting full-text budgets."""
    full = config.get("screening", {})
    explicit = config.get("metadata_screening", {})
    output_dir = config.get("output_dir", "output")
    return {
        "model": explicit.get("model", full.get("model", "qwen3.5:9b")),
        "base_url": explicit.get(
            "base_url",
            full.get("base_url", "http://localhost:11434"),
        ),
        "num_ctx": explicit.get("num_ctx", 8192),
        "fast_num_predict": explicit.get("fast_num_predict", 1200),
        "fallback_enabled": explicit.get("fallback_enabled", True),
        "fallback_think": explicit.get("fallback_think", "low"),
        "fallback_num_predict": explicit.get("fallback_num_predict", 3072),
        "timeout_seconds": explicit.get(
            "timeout_seconds",
            full.get("timeout_seconds", 180),
        ),
        "force": explicit.get("force", False),
        "artifacts_dir": explicit.get(
            "artifacts_dir",
            str(Path(output_dir) / "metadata_screening"),
        ),
    }


def select_for_metadata_screening(
    store,
    protocol,
    config,
    *,
    limit=None,
    retry_error=False,
):
    criteria = protocol.get("eligibility", {}).get("criteria", [])
    if not criteria:
        raise ValueError("Configure eligibility.criteria before metadata screening.")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("metadata screening limit must be positive or null.")

    cfg = metadata_config(config)
    model_digest = get_model_digest(cfg)
    selected = []
    summary = MetadataQueueSummary()

    for paper in store.all_papers():
        active, inactive, topics = _criteria_for_paper(criteria, paper)
        if not active:
            continue

        signature = metadata_screening_signature(
            paper,
            active,
            cfg,
            model_digest,
        )
        status = paper.get("metadata_screening_status") or "not_screened"
        stored = paper.get("metadata_screening_signature") or ""

        if status == "screened":
            if not cfg.get("force", False) and stored == signature:
                summary.skipped_complete += 1
                continue
            summary.stale += 1
        elif status == "error":
            if not retry_error:
                summary.skipped_retry_required += 1
                continue
        elif status not in {"not_screened", ""}:
            if not retry_error:
                summary.skipped_retry_required += 1
                continue

        summary.pending_total += 1
        if limit is None or len(selected) < limit:
            selected.append(
                (paper, active, inactive, topics, model_digest, signature)
            )

    summary.selected = len(selected)
    summary.remaining_pending = max(
        0,
        summary.pending_total - summary.selected,
    )
    return selected, summary


def screen_saved_metadata(
    store,
    protocol,
    config,
    *,
    limit=None,
    retry_error=False,
):
    """Process one advancing metadata batch and persist every paper immediately."""
    started = time.monotonic()
    cfg = metadata_config(config)
    selected, summary = select_for_metadata_screening(
        store,
        protocol,
        config,
        limit=limit,
        retry_error=retry_error,
    )

    root = Path(cfg["artifacts_dir"])
    root.mkdir(parents=True, exist_ok=True)

    for (
        paper,
        active,
        inactive,
        topics,
        model_digest,
        signature,
    ) in selected:
        corpus_id = paper["corpus_id"]
        folder = root / corpus_id

        # Remove only the previous machine-generated metadata-screening bundle.
        for key in list(paper):
            if key.startswith("metadata_screening_"):
                del paper[key]

        paper["metadata_screening_search_topics"] = topics
        paper["metadata_screening_inactive_criteria"] = inactive

        try:
            result = screen_metadata_paper(
                paper,
                active,
                cfg,
                folder,
                model_digest,
            )
            paper.update(result)
            summary.completed += 1
            decision = paper["metadata_screening_decision"]
            if decision == "include":
                summary.included += 1
            elif decision == "exclude":
                summary.excluded += 1
            else:
                summary.uncertain += 1
        except Exception as exc:
            message = str(exc).strip()
            paper.update({
                "metadata_screening_status": "error",
                "metadata_screening_decision": "uncertain",
                "metadata_screening_reason": (
                    f"{type(exc).__name__}: "
                    + (
                        message
                        if message
                        else "title/abstract screening failed; retry required."
                    )
                ),
                "metadata_screening_error_type": type(exc).__name__,
                "metadata_screening_error_message": message,
                "metadata_screening_artifact": str(folder),
            })
            summary.errors += 1

        store.save_metadata_screened_paper(
            paper,
            metadata_screening_signature=signature,
        )

    summary.elapsed_seconds = round(time.monotonic() - started, 3)
    processed = summary.completed + summary.errors
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

    store.record_metadata_screening_batch(summary.__dict__)
    return summary
