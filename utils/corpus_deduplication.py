"""Safe post-discovery bibliographic deduplication for the persistent corpus."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from utils.deduplication import decode, present


_TITLE_STOPWORDS = {
    "a", "an", "the", "of", "for", "on", "in", "to", "and", "with",
}
_AUTHOR_STOPWORDS = {"and", "et", "al"}


@dataclass
class CorpusDedupSummary:
    total_before: int = 0
    duplicate_clusters: int = 0
    records_to_merge: int = 0
    exact_title_clusters: int = 0
    fuzzy_title_clusters: int = 0
    reviewed_records_in_clusters: int = 0
    human_metadata_conflicts: int = 0
    blocked_manual_conflicts: int = 0
    applicable_clusters: int = 0
    applied_clusters: int = 0
    total_after: int = 0
    dry_run: bool = True
    backup_path: str = ""
    manifest_path: str = ""


def _norm(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalized_title(paper: dict) -> str:
    return _norm(paper.get("title"))


def _year(paper: dict) -> int | None:
    try:
        return int(float(paper.get("year")))
    except (TypeError, ValueError):
        return None


def _author_tokens(paper: dict) -> set[str]:
    value = paper.get("authors")
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    tokens = {
        token
        for token in _norm(value).split()
        if len(token) > 1 and token not in _AUTHOR_STOPWORDS
    }
    return tokens


def _author_overlap(a: dict, b: dict) -> float:
    left = _author_tokens(a)
    right = _author_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _year_compatible(a: dict, b: dict) -> bool:
    left = _year(a)
    right = _year(b)
    return left is not None and right is not None and abs(left - right) <= 1


def _authors_compatible(a: dict, b: dict) -> bool:
    return _author_overlap(a, b) >= 0.60


def _title_tokens(title: str) -> list[str]:
    return [
        token
        for token in title.split()
        if token not in _TITLE_STOPWORDS
    ]


def _fuzzy_block(title: str) -> tuple[str, ...] | None:
    tokens = _title_tokens(title)
    if len(tokens) < 5:
        return None
    return tuple(tokens[:3])


def _title_similarity(left: str, right: str) -> tuple[float, float]:
    sequence = SequenceMatcher(None, left, right, autojunk=False).ratio()
    left_tokens = set(_title_tokens(left))
    right_tokens = set(_title_tokens(right))
    union = left_tokens | right_tokens
    jaccard = (
        len(left_tokens & right_tokens) / len(union)
        if union
        else 0.0
    )
    return sequence, jaccard


def _is_published_record(paper: dict) -> bool:
    venue_type = str(paper.get("venue_type") or "").casefold()
    openalex_type = str(paper.get("openalex_type") or "").casefold()
    publication_type = str(paper.get("publication_type") or "").casefold()
    if venue_type in {"journal", "conference"}:
        return True
    if openalex_type in {"article", "conference-paper"}:
        return True
    return any(
        token in publication_type
        for token in ("article", "journal", "conference paper", "conferences")
    )


def _canonical_score(paper: dict) -> tuple:
    identifiers = sum(
        bool(present(paper.get(field)))
        for field in (
            "doi", "arxiv_id", "openalex_id", "scopus_id",
            "ieee_id", "semantic_scholar_id",
        )
    )
    return (
        int(_is_published_record(paper)),
        int(present(paper.get("doi"))),
        int(paper.get("metadata_screening_status") == "screened"),
        int(present(paper.get("abstract"))),
        identifiers,
        len(str(paper.get("abstract") or "")),
    )


def choose_canonical(cluster: list[dict]) -> dict:
    return max(
        cluster,
        key=lambda paper: (
            _canonical_score(paper),
            # Stable deterministic tie break: lexicographically smaller ID wins.
            tuple(-ord(ch) for ch in str(paper.get("corpus_id") or "")),
        ),
    )


def _manual_conflict(cluster: list[dict]) -> bool:
    decisions = {
        paper.get("manual_decision")
        for paper in cluster
        if paper.get("manual_decision")
    }
    return len(decisions) > 1


def _human_metadata_conflict(cluster: list[dict]) -> bool:
    decisions = {
        paper.get("human_metadata_screening_decision")
        for paper in cluster
        if paper.get("human_metadata_screening_decision")
    }
    return len(decisions) > 1


def _edge(a: dict, b: dict, method: str, similarity=None) -> dict:
    payload = {
        "left": a["corpus_id"],
        "right": b["corpus_id"],
        "method": method,
    }
    if similarity is not None:
        payload["sequence_similarity"] = round(similarity[0], 6)
        payload["token_jaccard"] = round(similarity[1], 6)
    return payload


def find_duplicate_clusters(
    papers: list[dict],
    *,
    include_fuzzy: bool = True,
) -> list[dict]:
    """Find conservative bibliographic duplicate clusters.

    Stable identifier duplicates should already be merged by CorpusStore.
    This pass targets cross-provider records lacking a shared identifier bridge.
    """
    papers = [
        paper
        for paper in papers
        if paper.get("corpus_id") and normalized_title(paper)
    ]
    by_id = {paper["corpus_id"]: paper for paper in papers}
    ids = list(by_id)
    parent = {corpus_id: corpus_id for corpus_id in ids}
    edges: list[dict] = []

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        lroot = root(left)
        rroot = root(right)
        if lroot != rroot:
            parent[rroot] = lroot

    exact_groups = defaultdict(list)
    for paper in papers:
        exact_groups[normalized_title(paper)].append(paper)

    for title, group in exact_groups.items():
        if len(group) < 2:
            continue
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                if not _year_compatible(left, right):
                    continue
                if not _authors_compatible(left, right):
                    continue
                union(left["corpus_id"], right["corpus_id"])
                edges.append(_edge(left, right, "exact_normalized_title"))

    if include_fuzzy:
        blocks = defaultdict(list)
        for paper in papers:
            title = normalized_title(paper)
            block = _fuzzy_block(title)
            if block is not None:
                blocks[block].append(paper)

        seen_pairs = set()
        for group in blocks.values():
            if len(group) < 2:
                continue
            for index, left in enumerate(group):
                for right in group[index + 1:]:
                    pair = tuple(sorted(
                        (left["corpus_id"], right["corpus_id"])
                    ))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    if normalized_title(left) == normalized_title(right):
                        continue
                    if not _year_compatible(left, right):
                        continue
                    if not _authors_compatible(left, right):
                        continue
                    similarity = _title_similarity(
                        normalized_title(left),
                        normalized_title(right),
                    )
                    if similarity[0] < 0.965 or similarity[1] < 0.90:
                        continue
                    union(left["corpus_id"], right["corpus_id"])
                    edges.append(_edge(
                        left,
                        right,
                        "fuzzy_title_author_year",
                        similarity,
                    ))

    grouped = defaultdict(list)
    for corpus_id in ids:
        grouped[root(corpus_id)].append(by_id[corpus_id])

    edge_by_cluster = defaultdict(list)
    for edge in edges:
        edge_by_cluster[root(edge["left"])].append(edge)

    result = []
    for cluster in grouped.values():
        if len(cluster) < 2:
            continue
        canonical = choose_canonical(cluster)
        cluster_ids = {paper["corpus_id"] for paper in cluster}
        cluster_edges = [
            edge
            for edge in edges
            if edge["left"] in cluster_ids and edge["right"] in cluster_ids
        ]
        methods = sorted({edge["method"] for edge in cluster_edges})
        result.append({
            "canonical_id": canonical["corpus_id"],
            "member_ids": sorted(cluster_ids),
            "source_ids": sorted(
                cluster_ids - {canonical["corpus_id"]}
            ),
            "title": canonical.get("title") or "",
            "years": sorted({
                year
                for paper in cluster
                for year in [_year(paper)]
                if year is not None
            }),
            "authors": [
                str(paper.get("authors") or "")
                for paper in cluster
            ],
            "methods": methods,
            "edges": cluster_edges,
            "human_metadata_decisions": {
                paper["corpus_id"]: (
                    paper.get("human_metadata_screening_decision") or ""
                )
                for paper in cluster
            },
            "manual_decisions": {
                paper["corpus_id"]: paper.get("manual_decision") or ""
                for paper in cluster
            },
            "metadata_screening_statuses": {
                paper["corpus_id"]: (
                    paper.get("metadata_screening_status") or ""
                )
                for paper in cluster
            },
            "reviewed_records": sum(
                bool(paper.get("human_metadata_screening_decision"))
                for paper in cluster
            ),
            "human_metadata_conflict": _human_metadata_conflict(cluster),
            "blocked_manual_conflict": _manual_conflict(cluster),
        })

    result.sort(key=lambda row: (row["title"].casefold(), row["canonical_id"]))
    return result


def _backup_database(store, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_dir / f"sota_corpus_before_dedup_{stamp}.sqlite3"
    suffix = 1
    while destination.exists():
        destination = (
            backup_dir
            / f"sota_corpus_before_dedup_{stamp}_{suffix}.sqlite3"
        )
        suffix += 1
    target = sqlite3.connect(destination)
    try:
        store.conn.backup(target)
    finally:
        target.close()
    return destination


def run_corpus_deduplication(
    store,
    config: dict,
    *,
    apply: bool = False,
    include_fuzzy: bool = True,
) -> CorpusDedupSummary:
    papers = store.all_papers()
    clusters = find_duplicate_clusters(
        papers,
        include_fuzzy=include_fuzzy,
    )
    applicable = [
        cluster
        for cluster in clusters
        if not cluster["blocked_manual_conflict"]
    ]

    summary = CorpusDedupSummary(
        total_before=len(papers),
        duplicate_clusters=len(clusters),
        records_to_merge=sum(
            len(cluster["source_ids"]) for cluster in applicable
        ),
        exact_title_clusters=sum(
            "fuzzy_title_author_year" not in cluster["methods"]
            for cluster in clusters
        ),
        fuzzy_title_clusters=sum(
            "fuzzy_title_author_year" in cluster["methods"]
            for cluster in clusters
        ),
        reviewed_records_in_clusters=sum(
            cluster["reviewed_records"] for cluster in clusters
        ),
        human_metadata_conflicts=sum(
            cluster["human_metadata_conflict"] for cluster in clusters
        ),
        blocked_manual_conflicts=sum(
            cluster["blocked_manual_conflict"] for cluster in clusters
        ),
        applicable_clusters=len(applicable),
        total_after=len(papers),
        dry_run=not apply,
    )

    output_dir = Path(config.get("output_dir", "output"))
    manifest_dir = Path(
        config.get(
            "deduplication",
            {},
        ).get(
            "manifest_dir",
            str(output_dir / "deduplication"),
        )
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "apply" if apply else "dry_run"
    manifest = manifest_dir / f"dedup_{mode}_{stamp}.json"
    suffix = 1
    while manifest.exists():
        manifest = manifest_dir / f"dedup_{mode}_{stamp}_{suffix}.json"
        suffix += 1

    if apply and applicable:
        backup = _backup_database(
            store,
            manifest_dir / "backups",
        )
        summary.backup_path = str(backup)

        for cluster in applicable:
            store.merge_corpus_ids(
                cluster["canonical_id"],
                cluster["source_ids"],
            )
            summary.applied_clusters += 1

        summary.total_after = len(store.all_papers())

    summary.manifest_path = str(manifest)
    manifest.write_text(
        json.dumps(
            {
                "summary": summary.__dict__,
                "policy": {
                    "stable_identifier_duplicates": (
                        "handled by normal CorpusStore identifier merging"
                    ),
                    "exact_title": (
                        "exact normalized title + year difference <= 1 "
                        "+ >=60% author-token containment"
                    ),
                    "fuzzy_title": (
                        "same first 3 informative title tokens + year "
                        "difference <= 1 + >=60% author-token containment "
                        "+ sequence similarity >=0.965 + token Jaccard >=0.90"
                    ),
                    "human_metadata_conflict": (
                        "any disagreement resolves to include"
                    ),
                    "manual_final_conflict": (
                        "cluster is blocked from automatic merge"
                    ),
                    "include_fuzzy": include_fuzzy,
                },
                "clusters": clusters,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return summary
