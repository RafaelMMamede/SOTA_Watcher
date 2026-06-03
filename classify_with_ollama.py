# classify_with_ollama.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import requests


OLLAMA_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "llm_priority": {
            "type": "string",
            "enum": ["read", "skim", "ignore"],
        },
        "llm_relevance": {
            "type": "string",
            "enum": ["direct", "partial", "adjacent", "not_relevant"],
        },
        "llm_topic": {
            "type": "string",
            "enum": [
                "deepfake_ai_image_detection",
                "adversarial_robustness_attacks_defenses",
                "both",
                "other",
            ],
        },
        "llm_reason": {
            "type": "string",
        },
        "llm_key_terms": {
            "type": "array",
            "items": {"type": "string"},
        },
        "llm_is_dataset_or_benchmark": {
            "type": "boolean",
        },
        "llm_mentions_code_or_release": {
            "type": "boolean",
        },
        "llm_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "llm_priority",
        "llm_relevance",
        "llm_topic",
        "llm_reason",
        "llm_key_terms",
        "llm_is_dataset_or_benchmark",
        "llm_mentions_code_or_release",
        "llm_confidence",
    ],
}


def load_prompt(path: str = "prompts/paper_triage_prompt.txt") -> str:
    return Path(path).read_text(encoding="utf-8")


def _paper_to_prompt(paper: Dict[str, Any]) -> str:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    venue = paper.get("venue", "")
    venue_type = paper.get("venue_type", "")
    openalex_type = paper.get("openalex_type", "")
    query = paper.get("query", "")
    topic_label = paper.get("topic_label", "")
    triage_score = paper.get("triage_score", "")

    return f"""
Paper metadata:

Title:
{title}

Abstract:
{abstract}

Venue:
{venue}

Venue type:
{venue_type}

Publication type:
{openalex_type}

Discovery query:
{query}

Keyword topic label:
{topic_label}

Keyword triage score:
{triage_score}

Classify this paper for the literature review.
""".strip()


def classify_paper_with_ollama(
    paper: Dict[str, Any],
    model: str,
    base_url: str = "http://localhost:11434",
    system_prompt_path: str = "prompts/paper_triage_prompt.txt",
    timeout_seconds: int = 120,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    system_prompt = load_prompt(system_prompt_path)
    user_prompt = _paper_to_prompt(paper)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "format": OLLAMA_TRIAGE_SCHEMA,
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=timeout_seconds,
    )

    response.raise_for_status()

    data = response.json()
    content = data.get("message", {}).get("content", "")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {
            "llm_priority": "ignore",
            "llm_relevance": "not_relevant",
            "llm_topic": "other",
            "llm_reason": f"Failed to parse Ollama JSON output: {content[:500]}",
            "llm_key_terms": [],
            "llm_is_dataset_or_benchmark": False,
            "llm_mentions_code_or_release": False,
            "llm_confidence": 0.0,
        }

    return parsed


def classify_papers_with_ollama(
    papers: list[dict],
    config: dict,
) -> list[dict]:
    ollama_cfg = config.get("ollama", {})

    if not ollama_cfg.get("enabled", False):
        print("Ollama classification disabled.")
        return papers

    model = ollama_cfg.get("model", "qwen2.5:7b")
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    min_score = ollama_cfg.get("classify_min_triage_score", 0)
    timeout_seconds = ollama_cfg.get("timeout_seconds", 120)
    temperature = ollama_cfg.get("temperature", 0.0)
    max_papers = ollama_cfg.get("max_papers_per_run", None)

    candidates = [
        paper for paper in papers
        if paper.get("triage_score", 0) >= min_score
    ]

    if max_papers is not None:
        candidates = candidates[: int(max_papers)]

    candidate_ids = {id(paper) for paper in candidates}

    print(
        f"\nOllama classification: classifying {len(candidates)} papers "
        f"with model {model}"
    )

    for i, paper in enumerate(papers, start=1):
        if id(paper) not in candidate_ids:
            paper.setdefault("llm_priority", "")
            paper.setdefault("llm_relevance", "")
            paper.setdefault("llm_topic", "")
            paper.setdefault("llm_reason", "")
            paper.setdefault("llm_key_terms", "")
            paper.setdefault("llm_is_dataset_or_benchmark", "")
            paper.setdefault("llm_mentions_code_or_release", "")
            paper.setdefault("llm_confidence", "")
            continue

        print(f"[{i}/{len(papers)}] Ollama triage: {paper.get('title', '')[:80]}")

        try:
            result = classify_paper_with_ollama(
                paper=paper,
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
            )

            # Convert list to string for Excel readability.
            if isinstance(result.get("llm_key_terms"), list):
                result["llm_key_terms"] = ", ".join(result["llm_key_terms"])

            paper.update(result)

        except requests.exceptions.RequestException as exc:
            paper.update(
                {
                    "llm_priority": "error",
                    "llm_relevance": "unknown",
                    "llm_topic": "unknown",
                    "llm_reason": f"Ollama request failed: {exc}",
                    "llm_key_terms": "",
                    "llm_is_dataset_or_benchmark": "",
                    "llm_mentions_code_or_release": "",
                    "llm_confidence": 0.0,
                }
            )

    return papers