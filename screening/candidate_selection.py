"""Deterministic pre-screen candidate selection for the systematic corpus."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from utils.deduplication import decode, present


DEFAULT_DEEPFAKE_TERMS = (
    "deepfake",
    "deep fake",
    "face forgery",
    "facial forgery",
    "face manipulation",
    "facial manipulation",
    "manipulated face",
    "synthetic face",
    "fake face",
    "forged face",
    "image forgery",
    "forged image",
    "ai-generated image",
    "ai generated image",
    "synthetic image",
    "forgery detection",
)

# Deliberately avoid bare "adversarial": deepfake papers often mention
# generative adversarial networks without studying adversarial examples.
DEFAULT_ADVERSARIAL_TERMS = (
    "adversarial attack",
    "adversarial attacks",
    "adversarial example",
    "adversarial examples",
    "adversarial perturbation",
    "adversarial perturbations",
    "adversarial robustness",
    "adversarial defense",
    "adversarial defence",
    "adversarial training",
    "evasion attack",
    "evasion attacks",
    "black-box attack",
    "black box attack",
    "white-box attack",
    "white box attack",
    "adversarial transferability",
    "transfer attack",
    "transfer attacks",
)

STREAM_VISUAL = "visual_forgery_detection"
STREAM_ADVERSARIAL = "adversarial_vision"


@dataclass
class CandidateSelectionSummary:
    total: int = 0
    selected: int = 0
    not_selected: int = 0
    unresolved: int = 0
    core_protected: int = 0
    recent_unfiltered: int = 0
    recent_venue_selected: int = 0
    recent_venue_not_whitelisted: int = 0
    recent_venue_unavailable: int = 0
    citation_selected: int = 0
    citation_below_quota: int = 0
    citation_unavailable: int = 0
    missing_year: int = 0
    other_stream: int = 0
    selected_by_stream: dict | None = None
    status_by_stream: dict | None = None
    selected_by_year: dict | None = None
    recent_venue_selected_by_stream: dict | None = None
    recent_venue_selected_by_year: dict | None = None
    recent_venue_selected_by_venue: dict | None = None
    manifest_path: str = ""
    run_id: str = ""
    dry_run: bool = False


def _normalize_venue_policy(raw) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "candidate_selection.recent_venue_filter.venues must be a list."
        )

    result = []
    seen_names = set()
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            aliases = [name]
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            aliases = item.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, (list, tuple)):
                raise ValueError(
                    "Each recent venue aliases field must be a list or string."
                )
            aliases = [str(value).strip() for value in aliases if str(value).strip()]
            if name and name not in aliases:
                aliases.insert(0, name)
        else:
            raise ValueError(
                "Each recent venue must be a string or {name, aliases} mapping."
            )

        if not name:
            raise ValueError("Each recent venue must have a non-empty name.")
        if not aliases:
            raise ValueError(
                f"Recent venue {name!r} must define at least one alias."
            )
        key = name.casefold()
        if key in seen_names:
            raise ValueError(f"Duplicate recent venue name: {name}")
        seen_names.add(key)
        result.append({"name": name, "aliases": aliases})
    return result


def candidate_selection_config(config: dict) -> dict:
    explicit = config.get("candidate_selection", {})
    recent_venue_filter = explicit.get("recent_venue_filter", {})
    if recent_venue_filter is None:
        recent_venue_filter = {}
    if not isinstance(recent_venue_filter, dict):
        raise ValueError(
            "candidate_selection.recent_venue_filter must be a mapping."
        )
    output_dir = config.get("output_dir", "output")
    return {
        "enabled": explicit.get("enabled", False),
        "historical_start_year": explicit.get("historical_start_year", 2020),
        "historical_through_year": explicit.get("historical_through_year", 2024),
        "recent_from_year": explicit.get("recent_from_year", 2025),
        "recent_venue_filter": {
            "enabled": recent_venue_filter.get("enabled", False),
            "venues": _normalize_venue_policy(
                recent_venue_filter.get("venues", [])
            ),
        },
        "top_n_per_year_per_stream": explicit.get(
            "top_n_per_year_per_stream",
            50,
        ),
        "include_ties": explicit.get("include_ties", True),
        "protect_both_streams": explicit.get("protect_both_streams", True),
        "protect_core_keywords": explicit.get("protect_core_keywords", True),
        "deepfake_terms": tuple(
            explicit.get("deepfake_terms", DEFAULT_DEEPFAKE_TERMS)
        ),
        "adversarial_terms": tuple(
            explicit.get("adversarial_terms", DEFAULT_ADVERSARIAL_TERMS)
        ),
        # The current corpus stores one merged citation_count, taking the
        # maximum observed provider count. Preserve that fact in the audit.
        "citation_metric": explicit.get(
            "citation_metric",
            "merged_max_provider_citation_count",
        ),
        "manifest_dir": explicit.get(
            "manifest_dir",
            str(Path(output_dir) / "candidate_selection"),
        ),
    }


def _validate_config(cfg: dict):
    for key in (
        "historical_start_year",
        "historical_through_year",
        "recent_from_year",
        "top_n_per_year_per_stream",
    ):
        if type(cfg[key]) is not int:
            raise ValueError(f"candidate_selection.{key} must be an integer.")
    if cfg["top_n_per_year_per_stream"] < 1:
        raise ValueError(
            "candidate_selection.top_n_per_year_per_stream must be positive."
        )
    if cfg["historical_start_year"] > cfg["historical_through_year"]:
        raise ValueError("candidate selection historical year range is invalid.")
    if cfg["recent_from_year"] <= cfg["historical_through_year"]:
        raise ValueError(
            "candidate_selection.recent_from_year must be after "
            "historical_through_year."
        )
    recent_filter = cfg["recent_venue_filter"]
    if type(recent_filter.get("enabled")) is not bool:
        raise ValueError(
            "candidate_selection.recent_venue_filter.enabled must be boolean."
        )
    if recent_filter["enabled"] and not recent_filter["venues"]:
        raise ValueError(
            "candidate_selection.recent_venue_filter.venues cannot be empty "
            "when the recent venue filter is enabled."
        )


def _topics(paper: dict) -> list[str]:
    values = decode(paper.get("search_topics"), [])
    if not values and present(paper.get("search_topic")):
        values = [paper.get("search_topic")]
    return sorted({
        str(value).strip()
        for value in values
        if present(value)
    })


def paper_stream(paper: dict) -> str:
    topics = set(_topics(paper))
    visual = STREAM_VISUAL in topics
    adversarial = STREAM_ADVERSARIAL in topics
    if visual and adversarial:
        return "both"
    if visual:
        return "visual"
    if adversarial:
        return "adversarial"
    return "other"


def _normalized_text(paper: dict) -> str:
    text = " ".join(
        str(paper.get(key) or "")
        for key in ("title", "abstract")
    ).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, terms) -> bool:
    return any(str(term).casefold() in text for term in terms if str(term).strip())


def _normalize_venue(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    ).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _paper_venues(paper: dict) -> list[str]:
    values = [paper.get("venue")]
    variants = decode(paper.get("metadata_variants"), {})
    if isinstance(variants, dict):
        alternate = variants.get("venue", [])
        if isinstance(alternate, (list, tuple)):
            values.extend(alternate)
        elif present(alternate):
            values.append(alternate)

    result = []
    seen = set()
    for value in values:
        if not present(value):
            continue
        venue = str(value).strip()
        normalized = _normalize_venue(venue)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(venue)
    return result


def _recent_venue_match(
    paper: dict,
    cfg: dict,
) -> tuple[str, list[str]]:
    observed = _paper_venues(paper)
    normalized_observed = [
        (_normalize_venue(value), value)
        for value in observed
    ]

    for venue_entry in cfg["recent_venue_filter"]["venues"]:
        for alias in venue_entry["aliases"]:
            normalized_alias = _normalize_venue(alias)
            if not normalized_alias:
                continue
            needle = f" {normalized_alias} "
            for normalized_venue, _ in normalized_observed:
                haystack = f" {normalized_venue} "
                if (
                    normalized_alias == normalized_venue
                    or needle in haystack
                ):
                    return venue_entry["name"], observed
    return "", observed


def is_potential_core(paper: dict, cfg: dict) -> tuple[bool, str]:
    stream = paper_stream(paper)
    if cfg.get("protect_both_streams", True) and stream == "both":
        return True, "retrieved_by_both_search_streams"

    if cfg.get("protect_core_keywords", True):
        text = _normalized_text(paper)
        if (
            _contains_any(text, cfg["deepfake_terms"])
            and _contains_any(text, cfg["adversarial_terms"])
        ):
            return True, "deepfake_and_adversarial_concepts"

    return False, ""


def _year(paper: dict) -> int | None:
    try:
        return int(float(paper.get("year")))
    except (TypeError, ValueError):
        return None


def _citation(paper: dict) -> float | None:
    value = paper.get("citation_count")
    if value in (None, ""):
        return None
    try:
        citation = float(value)
    except (TypeError, ValueError):
        return None
    return citation if math.isfinite(citation) else None


def _rank_with_ties(items: list[dict]) -> dict[str, int]:
    ordered = sorted(
        items,
        key=lambda row: (
            -row["citation_count"],
            row["corpus_id"],
        ),
    )
    result = {}
    prior = None
    prior_rank = 0
    for index, row in enumerate(ordered, start=1):
        value = row["citation_count"]
        if prior is None or value != prior:
            prior_rank = index
            prior = value
        result[row["corpus_id"]] = prior_rank
    return result


def build_candidate_selection(
    papers: list[dict],
    config: dict,
    *,
    top_n: int | None = None,
) -> tuple[list[dict], dict, CandidateSelectionSummary]:
    """Apply the configured core-protection + citation-selection policy."""
    cfg = candidate_selection_config(config)
    if top_n is not None:
        cfg["top_n_per_year_per_stream"] = top_n
    _validate_config(cfg)

    records = []
    historical_rankable = defaultdict(list)

    for paper in papers:
        corpus_id = paper["corpus_id"]
        year = _year(paper)
        citation = _citation(paper)
        stream = paper_stream(paper)
        protected, protected_reason = is_potential_core(paper, cfg)

        observed_venues = _paper_venues(paper)
        row = {
            "corpus_id": corpus_id,
            "candidate_selection_title": str(paper.get("title") or ""),
            "candidate_selection_search_topics": _topics(paper),
            "candidate_selection_venues": observed_venues,
            "candidate_selection_recent_venue_match": "",
            "candidate_selection_status": "",
            "candidate_selection_stream": stream,
            "candidate_selection_reason": "",
            "candidate_selection_year": year,
            "candidate_selection_citation_count": citation,
            "candidate_selection_citation_rank": None,
            "candidate_selection_citation_quota": (
                cfg["top_n_per_year_per_stream"]
            ),
            "candidate_selection_core_protected": protected,
            "candidate_selection_core_reason": protected_reason,
            "candidate_selection_citation_metric": cfg["citation_metric"],
        }

        if protected:
            row["candidate_selection_status"] = "selected"
            row["candidate_selection_reason"] = "core_protected"
        elif year is None:
            row["candidate_selection_status"] = "unresolved"
            row["candidate_selection_reason"] = "missing_year"
        elif year >= cfg["recent_from_year"]:
            if not cfg["recent_venue_filter"]["enabled"]:
                row["candidate_selection_status"] = "selected"
                row["candidate_selection_reason"] = "recent_unfiltered"
            elif stream not in {"visual", "adversarial"}:
                row["candidate_selection_status"] = "unresolved"
                row["candidate_selection_reason"] = "other_stream"
            else:
                venue_match, observed_venues = _recent_venue_match(paper, cfg)
                row["candidate_selection_venues"] = observed_venues
                row["candidate_selection_recent_venue_match"] = venue_match
                if venue_match:
                    row["candidate_selection_status"] = "selected"
                    row["candidate_selection_reason"] = "recent_reputable_venue"
                elif observed_venues:
                    row["candidate_selection_status"] = "not_selected"
                    row["candidate_selection_reason"] = (
                        "recent_venue_not_whitelisted"
                    )
                else:
                    row["candidate_selection_status"] = "unresolved"
                    row["candidate_selection_reason"] = (
                        "recent_venue_unavailable"
                    )
        elif (
            cfg["historical_start_year"]
            <= year
            <= cfg["historical_through_year"]
        ):
            if stream not in {"visual", "adversarial"}:
                row["candidate_selection_status"] = "unresolved"
                row["candidate_selection_reason"] = "other_stream"
            elif citation is None:
                row["candidate_selection_status"] = "unresolved"
                row["candidate_selection_reason"] = "citation_unavailable"
            else:
                historical_rankable[(year, stream)].append({
                    "corpus_id": corpus_id,
                    "citation_count": citation,
                })
                row["candidate_selection_status"] = "pending_rank"
                row["candidate_selection_reason"] = "pending_citation_rank"
        else:
            row["candidate_selection_status"] = "unresolved"
            row["candidate_selection_reason"] = "outside_selection_year_policy"

        records.append(row)

    rows_by_id = {row["corpus_id"]: row for row in records}
    quota = cfg["top_n_per_year_per_stream"]

    for (year, stream), items in historical_rankable.items():
        ranks = _rank_with_ties(items)
        ordered_counts = sorted(
            (row["citation_count"] for row in items),
            reverse=True,
        )
        cutoff = (
            ordered_counts[min(quota, len(ordered_counts)) - 1]
            if ordered_counts
            else None
        )

        for item in items:
            row = rows_by_id[item["corpus_id"]]
            rank = ranks[item["corpus_id"]]
            row["candidate_selection_citation_rank"] = rank

            selected = rank <= quota
            if cfg.get("include_ties", True) and cutoff is not None:
                selected = item["citation_count"] >= cutoff

            if selected:
                row["candidate_selection_status"] = "selected"
                row["candidate_selection_reason"] = "citation_top_n"
            else:
                row["candidate_selection_status"] = "not_selected"
                row["candidate_selection_reason"] = "citation_below_quota"

    status_counts = Counter(row["candidate_selection_status"] for row in records)
    reason_counts = Counter(row["candidate_selection_reason"] for row in records)
    selected_by_stream = Counter(
        row["candidate_selection_stream"]
        for row in records
        if row["candidate_selection_status"] == "selected"
    )
    selected_by_year = Counter(
        str(row["candidate_selection_year"])
        for row in records
        if row["candidate_selection_status"] == "selected"
    )
    status_by_stream = defaultdict(Counter)
    for row in records:
        status_by_stream[row["candidate_selection_stream"]][
            row["candidate_selection_status"]
        ] += 1

    recent_venue_rows = [
        row for row in records
        if row["candidate_selection_reason"] == "recent_reputable_venue"
    ]
    recent_venue_selected_by_stream = Counter(
        row["candidate_selection_stream"] for row in recent_venue_rows
    )
    recent_venue_selected_by_year = Counter(
        str(row["candidate_selection_year"]) for row in recent_venue_rows
    )
    recent_venue_selected_by_venue = Counter(
        row["candidate_selection_recent_venue_match"]
        for row in recent_venue_rows
    )

    summary = CandidateSelectionSummary(
        total=len(records),
        selected=status_counts["selected"],
        not_selected=status_counts["not_selected"],
        unresolved=status_counts["unresolved"],
        core_protected=reason_counts["core_protected"],
        recent_unfiltered=reason_counts["recent_unfiltered"],
        recent_venue_selected=reason_counts["recent_reputable_venue"],
        recent_venue_not_whitelisted=reason_counts[
            "recent_venue_not_whitelisted"
        ],
        recent_venue_unavailable=reason_counts["recent_venue_unavailable"],
        citation_selected=reason_counts["citation_top_n"],
        citation_below_quota=reason_counts["citation_below_quota"],
        citation_unavailable=reason_counts["citation_unavailable"],
        missing_year=reason_counts["missing_year"],
        other_stream=reason_counts["other_stream"],
        selected_by_stream=dict(sorted(selected_by_stream.items())),
        status_by_stream={
            stream: dict(sorted(counts.items()))
            for stream, counts in sorted(status_by_stream.items())
        },
        selected_by_year=dict(
            sorted(selected_by_year.items(), key=lambda item: item[0])
        ),
        recent_venue_selected_by_stream=dict(
            sorted(recent_venue_selected_by_stream.items())
        ),
        recent_venue_selected_by_year=dict(
            sorted(recent_venue_selected_by_year.items())
        ),
        recent_venue_selected_by_venue=dict(
            sorted(recent_venue_selected_by_venue.items())
        ),
    )
    return records, cfg, summary


def selection_policy_hash(policy: dict) -> str:
    payload = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        default=list,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def run_candidate_selection(
    store,
    config: dict,
    *,
    top_n: int | None = None,
    dry_run: bool = False,
):
    records, policy, summary = build_candidate_selection(
        store.all_papers(),
        config,
        top_n=top_n,
    )
    summary.dry_run = dry_run

    manifest_dir = Path(policy["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    digest = selection_policy_hash(policy)
    if dry_run:
        manifest = (
            manifest_dir
            / f"candidate_selection_{digest[:12]}_dry_run.json"
        )
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest = (
            manifest_dir
            / f"candidate_selection_{digest[:12]}_{stamp}.json"
        )
        suffix = 1
        while manifest.exists():
            manifest = (
                manifest_dir
                / f"candidate_selection_{digest[:12]}_{stamp}_{suffix}.json"
            )
            suffix += 1

    manifest_payload = {
        "policy": policy,
        "policy_hash": digest,
        "summary": {
            key: value
            for key, value in summary.__dict__.items()
            if key not in {"manifest_path", "run_id"}
        },
        "records": records,
    }
    manifest.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=list,
        ),
        encoding="utf-8",
    )
    summary.manifest_path = str(manifest)

    if not dry_run:
        run_id = store.replace_candidate_selection(
            policy,
            records,
            summary.__dict__,
            manifest_path=str(manifest),
        )
        summary.run_id = run_id

    return summary
