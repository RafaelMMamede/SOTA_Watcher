# SOTA Watcher

SOTA Watcher is a lightweight literature-monitoring pipeline for discovering, triaging, and summarizing recent academic papers.

It retrieves paper metadata from OpenAlex, scores candidate papers using configurable keyword rules, optionally performs LLM-based triage with a local Ollama model, and exports a structured Excel table and Markdown summary of recommended papers.

The project is designed for researchers who want to monitor new papers in a specific topic area without manually checking multiple search engines every week.

## Features

- Search academic papers through OpenAlex
- Configure custom review topics and keyword scoring rules
- Deduplicate papers by DOI, OpenAlex ID, URL, or title
- Rank papers using a configurable triage score
- Use a local Ollama model for `read` / `skim` / `ignore` screening
- Optionally run deeper abstract-level analysis on recommended papers
- Export results to Excel
- Generate a Markdown summary of papers recommended to read or skim
- Keep private configs, prompts, and outputs outside Git

## Repository structure

```text
SOTA_Watcher/
├── sources/
│   └── openalex_source.py
├── utils/
│   ├── config.py
│   ├── deduplication.py
│   ├── io.py
│   ├── scoring.py
│   └── text.py
├── prompts.example/
│   ├── paper_triage_prompt.example.txt
│   └── paper_deep_analysis_prompt.example.txt
├── classify_with_ollama.py
├── deep_analyze_with_ollama.py
├── summarize_recommended.py
├── sota_watcher.py
├── search_terms.yaml
├── config.example.yaml
├── requirements.txt
└── README.md