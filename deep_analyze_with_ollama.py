# deep_analyze_with_ollama.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests


OLLAMA_DEEP_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "deep_summary": {
            "type": "string",
        },
        "deep_problem": {
            "type": "string",
        },
        "deep_method": {
            "type": "string",
        },
        "deep_datasets": {
            "type": "array",
            "items": {"type": "string"},
        },
        "deep_results": {
            "type": "string",
        },
        "deep_limitations": {
            "type": "string",
        },
        "deep_relevance_to_review": {
            "type": "string",
        },
        "deep_reading_notes": {
            "type": "string",
        },
        "deep_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "deep_summary",
        "deep_problem",
        "deep_method",
        "deep_datasets",
        "deep_results",
        "deep_limitations",
        "deep_relevance_to_review",
        "deep_reading_notes",
        "deep_confidence",
    ],
}


def load_prompt(path: str = "prompts/paper_deep_analysis_prompt.txt") -> str:
    return Path(path).read_text(encoding="utf-8")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _paper_to_prompt(paper: Dict[str, Any]) -> str:
    title = _clean_text(paper.get("title", ""))
    abstract = _clean_text(paper.get("abstract", ""))
    authors = _clean_text(paper.get("authors", ""))
    year = _clean_text(paper.get("year", ""))
    venue = _clean_text(paper.get("venue", ""))
    venue_type = _clean_text(paper.get("venue_type", ""))
    openalex_type = _clean_text(paper.get("openalex_type", ""))
    query = _clean_text(paper.get("query", ""))

    topic_label = _clean_text(paper.get("topic_label", ""))
    triage_score = _clean_text(paper.get("triage_score", ""))

    llm_priority = _clean_text(paper.get("llm_priority", ""))
    llm_relevance = _clean_text(paper.get("llm_relevance", ""))
    llm_reason = _clean_text(paper.get("llm_reason", ""))
    llm_key_terms = _clean_text(paper.get("llm_key_terms", ""))
    llm_summary = _clean_text(paper.get("llm_summary", ""))

    doi = _clean_text(paper.get("doi", ""))
    url = _clean_text(paper.get("url", ""))

    return f"""
Paper metadata:

Title:
{title}

Authors:
{authors}

Year:
{year}

Venue:
{venue}

Venue type:
{venue_type}

Publication type:
{openalex_type}

DOI:
{doi}

URL:
{url}

Discovery query:
{query}

Keyword topic:
{topic_label}

Keyword triage score:
{triage_score}

Fast LLM priority:
{llm_priority}

Fast LLM relevance:
{llm_relevance}

Fast LLM reason:
{llm_reason}

Fast LLM key terms:
{llm_key_terms}

Fast LLM summary, if available:
{llm_summary}

Abstract:
{abstract}

Now produce a deeper structured analysis of this paper for literature-review screening.
""".strip()


def deep_analyze_paper_with_ollama(
    paper: Dict[str, Any],
    model: str,
    base_url: str = "http://localhost:11434",
    system_prompt_path: str = "prompts/paper_deep_analysis_prompt.txt",
    timeout_seconds: int = 300,
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
        "format": OLLAMA_DEEP_ANALYSIS_SCHEMA,
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
            "deep_summary": "",
            "deep_problem": "",
            "deep_method": "",
            "deep_datasets": "",
            "deep_results": "",
            "deep_limitations": "",
            "deep_relevance_to_review": "",
            "deep_reading_notes": f"Failed to parse Ollama JSON output: {content[:500]}",
            "deep_confidence": 0.0,
        }

    return parsed


def _should_deep_analyze(paper: Dict[str, Any], allowed_priorities: List[str]) -> bool:
    priority = _clean_text(paper.get("llm_priority", "")).lower()
    return priority in allowed_priorities


def _set_empty_deep_fields(paper: Dict[str, Any]) -> None:
    paper.setdefault("deep_summary", "")
    paper.setdefault("deep_problem", "")
    paper.setdefault("deep_method", "")
    paper.setdefault("deep_datasets", "")
    paper.setdefault("deep_results", "")
    paper.setdefault("deep_limitations", "")
    paper.setdefault("deep_relevance_to_review", "")
    paper.setdefault("deep_reading_notes", "")
    paper.setdefault("deep_confidence", "")


def deep_analyze_recommended_papers_with_ollama(
    papers: list[dict],
    config: dict,
) -> list[dict]:
    deep_cfg = config.get("ollama_deep_analysis", {})

    if not deep_cfg.get("enabled", False):
        print("Deep Ollama analysis disabled.")
        for paper in papers:
            _set_empty_deep_fields(paper)
        return papers

    model = deep_cfg.get("model", "deepseek-r1:8b")
    base_url = deep_cfg.get("base_url", "http://localhost:11434")
    timeout_seconds = deep_cfg.get("timeout_seconds", 300)
    temperature = deep_cfg.get("temperature", 0.0)
    max_papers = deep_cfg.get("max_papers_per_run", None)

    allowed_priorities = [
        str(p).lower().strip()
        for p in deep_cfg.get("only_priorities", ["read", "skim"])
    ]

    candidates = [
        paper for paper in papers
        if _should_deep_analyze(paper, allowed_priorities)
    ]

    if max_papers is not None:
        candidates = candidates[: int(max_papers)]

    candidate_ids = {id(paper) for paper in candidates}

    print(
        f"\nDeep Ollama analysis: analyzing {len(candidates)} papers "
        f"with model {model}"
    )

    for i, paper in enumerate(papers, start=1):
        if id(paper) not in candidate_ids:
            _set_empty_deep_fields(paper)
            continue

        title = paper.get("title", "")
        print(f"[{i}/{len(papers)}] Deep analysis: {str(title)[:90]}")

        try:
            result = deep_analyze_paper_with_ollama(
                paper=paper,
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
            )

            if isinstance(result.get("deep_datasets"), list):
                result["deep_datasets"] = ", ".join(result["deep_datasets"])

            paper.update(result)

        except requests.exceptions.RequestException as exc:
            paper.update(
                {
                    "deep_summary": "",
                    "deep_problem": "",
                    "deep_method": "",
                    "deep_datasets": "",
                    "deep_results": "",
                    "deep_limitations": "",
                    "deep_relevance_to_review": "",
                    "deep_reading_notes": f"Deep Ollama request failed: {exc}",
                    "deep_confidence": 0.0,
                }
            )

    return papers