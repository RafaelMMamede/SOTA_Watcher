"""Persistent title/abstract screening queue."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

from screening.metadata import (
    metadata_screening_signature,
    screen_metadata_paper,
)
from screening.ollama import get_model_digest
from screening.pipeline import _criteria_for_paper
from utils.deduplication import decode, present


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
    sampling_mode: str = ""
    sample_seed: int | None = None
    sample_manifest: str = ""
    sample_strata: dict | None = None


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
        "repair_num_predict": explicit.get("repair_num_predict", 1600),
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


def _paper_sources(paper):
    sources = decode(paper.get("sources"), [])
    if not sources and present(paper.get("source")):
        sources = [paper.get("source")]
    return sorted({
        str(source).strip()
        for source in sources
        if present(source)
    })


def _sample_stratum(candidate):
    paper, _, _, topics, _, _ = candidate
    topic_key = tuple(sorted(set(topics))) or ("unscoped",)
    source_key = tuple(_paper_sources(paper)) or ("unknown",)
    return topic_key, source_key


def _proportional_quotas(groups, sample_size):
    """Hamilton allocation across mutually exclusive strata."""
    total = sum(len(items) for items in groups.values())
    if sample_size >= total:
        return {key: len(items) for key, items in groups.items()}

    ideals = {
        key: sample_size * len(items) / total
        for key, items in groups.items()
    }
    quotas = {
        key: min(len(groups[key]), int(value))
        for key, value in ideals.items()
    }
    remaining = sample_size - sum(quotas.values())

    order = sorted(
        groups,
        key=lambda key: (
            -(ideals[key] - int(ideals[key])),
            -len(groups[key]),
            repr(key),
        ),
    )
    for key in order:
        if remaining <= 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1

    return quotas


def stratified_metadata_sample(candidates, sample_size, seed):
    """Return a deterministic proportional sample by topic x source signature."""
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("stratified sample size must be a positive integer.")
    if type(seed) is not int:
        raise ValueError("stratified sample seed must be an integer.")

    groups = defaultdict(list)
    for candidate in candidates:
        groups[_sample_stratum(candidate)].append(candidate)

    quotas = _proportional_quotas(groups, min(sample_size, len(candidates)))
    rng = random.Random(seed)
    sampled = []
    strata = {}

    for key in sorted(groups, key=repr):
        items = list(groups[key])
        rng.shuffle(items)
        chosen = items[:quotas.get(key, 0)]
        sampled.extend(chosen)
        topic_key, source_key = key
        label = (
            "topics=" + "+".join(topic_key)
            + "|sources=" + "+".join(source_key)
        )
        strata[label] = {
            "population": len(items),
            "selected": len(chosen),
        }

    # Do not expose insertion/source order as the screening order.
    rng.shuffle(sampled)
    return sampled, strata


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
    stratified_sample=None,
    sample_seed=42,
):
    """Process one advancing metadata batch and persist every paper immediately."""
    started = time.monotonic()
    cfg = metadata_config(config)
    selected, summary = select_for_metadata_screening(
        store,
        protocol,
        config,
        limit=None if stratified_sample is not None else limit,
        retry_error=retry_error,
    )

    root = Path(cfg["artifacts_dir"])
    root.mkdir(parents=True, exist_ok=True)

    if stratified_sample is not None:
        summary.sampling_mode = "proportional_topic_source_stratified"
        summary.sample_seed = sample_seed

        manifest_dir = root / "samples"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = (
            manifest_dir
            / f"metadata_sample_n{stratified_sample}_seed{sample_seed}.json"
        )

        if manifest.exists():
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                manifest_payload.get("sampling_mode") != summary.sampling_mode
                or manifest_payload.get("requested_size") != stratified_sample
                or manifest_payload.get("seed") != sample_seed
            ):
                raise ValueError(
                    "Existing metadata sample manifest does not match the "
                    "requested sampling configuration."
                )
            by_id = {
                candidate[0].get("corpus_id"): candidate
                for candidate in selected
            }
            selected = [
                by_id[item["corpus_id"]]
                for item in manifest_payload.get("papers", [])
                if item.get("corpus_id") in by_id
            ]
            strata = manifest_payload.get("strata", {})
        else:
            selected, strata = stratified_metadata_sample(
                selected,
                stratified_sample,
                sample_seed,
            )
            manifest_payload = {
                "sampling_mode": summary.sampling_mode,
                "requested_size": stratified_sample,
                "selected_size": len(selected),
                "seed": sample_seed,
                "pending_population": summary.pending_total,
                "strata": strata,
                "papers": [
                    {
                        "corpus_id": candidate[0].get("corpus_id"),
                        "title": candidate[0].get("title"),
                        "year": candidate[0].get("year"),
                        "search_topics": candidate[3],
                        "sources": _paper_sources(candidate[0]),
                    }
                    for candidate in selected
                ],
            }
            manifest.write_text(
                json.dumps(
                    manifest_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        summary.selected = len(selected)
        summary.remaining_pending = max(
            0,
            summary.pending_total - summary.selected,
        )
        summary.sample_strata = strata
        summary.sample_manifest = str(manifest)

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
