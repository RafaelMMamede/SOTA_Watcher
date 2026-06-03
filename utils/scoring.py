# utils/scoring.py

from __future__ import annotations

import math
from typing import Any, Dict


def _to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value)


def _contains(text: str, term: str) -> bool:
    return term.lower() in text.lower()


def _weighted_term_score(text: str, terms: Dict[str, int]) -> int:
    score = 0

    text = _to_text(text).lower()

    for term, weight in terms.items():
        if term.lower() in text:
            score += int(weight)

    return score


def infer_topic_scores(
    paper: dict,
    topics: Dict[str, dict],
    scoring_config: Dict,
) -> Dict[str, int]:
    title = _to_text(paper.get("title", ""))
    abstract = _to_text(paper.get("abstract", ""))
    query = _to_text(paper.get("query", ""))

    title_weight = scoring_config.get("title_weight", 3)
    abstract_weight = scoring_config.get("abstract_weight", 1)
    query_weight = scoring_config.get("query_weight", 2)

    scores = {}

    for topic_key, topic_cfg in topics.items():
        topic_terms = topic_cfg.get("topic_terms", {})

        score = 0
        score += title_weight * _weighted_term_score(title, topic_terms)
        score += abstract_weight * _weighted_term_score(abstract, topic_terms)
        score += query_weight * _weighted_term_score(query, topic_terms)

        scores[topic_key] = score

    return scores


def infer_best_topic(
    paper: dict,
    topics: Dict[str, dict],
    scoring_config: Dict,
) -> tuple[str, str, int]:
    topic_scores = infer_topic_scores(paper, topics, scoring_config)

    if not topic_scores:
        return "out_of_scope", "Out of scope", 0

    best_topic = max(topic_scores, key=topic_scores.get)
    best_score = topic_scores[best_topic]

    if best_score <= 0:
        return "out_of_scope", "Out of scope", 0

    label = topics.get(best_topic, {}).get("label", best_topic)

    return best_topic, label, best_score


def score_paper(
    paper: dict,
    search_terms: Dict,
    scoring_config: Dict,
) -> dict:
    topics = search_terms.get("topics", {})
    generic_priority_terms = search_terms.get("generic_priority_terms", {})

    title = _to_text(paper.get("title", ""))
    abstract = _to_text(paper.get("abstract", ""))
    query = _to_text(paper.get("query", ""))

    text_all = f"{title} {abstract} {query}"

    topic_key, topic_label, topic_score = infer_best_topic(
        paper=paper,
        topics=topics,
        scoring_config=scoring_config,
    )

    if topic_key == "out_of_scope":
        return {
            "topic": topic_key,
            "topic_label": topic_label,
            "topic_score": 0,
            "triage_score": 0,
            "generic_priority_score": 0,
        }

    generic_priority_score = _weighted_term_score(
        text_all,
        generic_priority_terms,
    )

    triage_score = topic_score
    triage_score += scoring_config.get("generic_priority_weight", 1) * generic_priority_score

    venue_type = _to_text(paper.get("venue_type", "")).lower()
    openalex_type = _to_text(paper.get("openalex_type", "")).lower()
    has_abstract = bool(paper.get("has_abstract", False))
    is_repository = bool(paper.get("is_repository", False))

    if venue_type == "journal":
        triage_score += scoring_config.get("journal_bonus", 2)

    if venue_type == "conference":
        triage_score += scoring_config.get("conference_bonus", 3)

    if openalex_type == "preprint":
        triage_score += scoring_config.get("preprint_bonus", 1)

    if openalex_type == "dataset":
        triage_score += scoring_config.get("dataset_bonus", 1)

    if is_repository:
        triage_score -= scoring_config.get("repository_penalty", 3)

    if openalex_type == "report":
        triage_score -= scoring_config.get("report_penalty", 4)

    if not has_abstract:
        triage_score -= scoring_config.get("missing_abstract_penalty", 1)

    citation_count = paper.get("citation_count", 0) or 0

    try:
        citation_count = float(citation_count)
    except (TypeError, ValueError):
        citation_count = 0

    citation_weight = scoring_config.get("citation_weight", 0.25)
    max_citation_bonus = scoring_config.get("max_citation_bonus", 5)

    citation_bonus = min(
        max_citation_bonus,
        citation_weight * math.log1p(citation_count),
    )

    triage_score += citation_bonus

    return {
        "topic": topic_key,
        "topic_label": topic_label,
        "topic_score": round(topic_score, 3),
        "generic_priority_score": round(generic_priority_score, 3),
        "triage_score": round(max(triage_score, 0), 3),
    }