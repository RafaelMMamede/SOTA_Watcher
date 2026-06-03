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
```

## Installation

Create and activate a Python environment:

```bash
conda create -n sota-watcher python=3.11
conda activate sota-watcher
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Ollama setup

This project can use local Ollama models for paper triage and deeper abstract-level analysis.

Install Ollama following the official instructions, then pull the models you want to use. For example:

```bash
ollama pull qwen2.5:7b
ollama pull deepseek-r1:8b
```

The default configuration assumes Ollama is available at:

```text
http://localhost:11434
```

You can check that Ollama is running with:

```bash
curl http://localhost:11434/api/tags
```

## Configuration

The real `config.yaml` file is ignored by Git.

Create it from the example template:

```bash
cp config.example.yaml config.yaml
```

Then edit:

```yaml
mailto: your_email@example.com
```

The email is used for polite OpenAlex API access.

You can also change the search date range:

```yaml
from_publication_date: "2024-01-01"
to_publication_date: "2024-12-31"
```

For historical searches, it is recommended to run the pipeline year by year, for example:

```text
2022-01-01 to 2022-12-31
2023-01-01 to 2023-12-31
2024-01-01 to 2024-12-31
2025-01-01 to 2025-12-31
```

## Prompts

The real `prompts/` directory is ignored by Git.

Create it from the examples:

```bash
mkdir -p prompts
cp prompts.example/paper_triage_prompt.example.txt prompts/paper_triage_prompt.txt
cp prompts.example/paper_deep_analysis_prompt.example.txt prompts/paper_deep_analysis_prompt.txt
```

Edit the private prompt files in `prompts/` to describe your own literature-review scope.

The example prompts are intentionally domain-neutral.

## Search terms

Search topics and scoring keywords are defined in:

```text
search_terms.yaml
```

This file controls:

- Which queries are sent to OpenAlex
- Which topic labels are assigned
- Which terms increase the triage score
- Which general terms are treated as priority indicators

Edit this file to match your own review topic.

## Basic usage

Run the main watcher:

```bash
python sota_watcher.py
```

This will:

1. Query OpenAlex
2. Deduplicate papers
3. Score papers using keyword rules
4. Filter papers by triage score
5. Optionally run Ollama triage
6. Save the result table

By default, outputs are written to:

```text
output/sota_table.xlsx
```

## Generate a recommended-papers summary

After running the watcher, generate a Markdown report of papers marked as `read` or `skim`:

```bash
python summarize_recommended.py
```

This creates:

```text
output/recommended_papers_summary.md
```


## Recommended historical workflow

For large-scale historical extraction:

```bash
# 1. Set config.yaml to one year, e.g. 2022
python sota_watcher.py

# 2. Change config.yaml to 2023
python sota_watcher.py

# 3. Repeat for each year

# 4. Generate final recommended summary
python summarize_recommended.py
```

The cumulative Excel table is updated across runs.

## Output files

Generated outputs are ignored by Git.

Typical files include:

```text
output/sota_table.xlsx
output/recommended_papers_summary.md
output/seen_papers.csv
```

## Notes on full-paper analysis

The current pipeline performs metadata-level and abstract-level analysis.

OpenAlex can provide metadata, abstracts, open-access URLs, and sometimes PDF links, but this project does not yet download or parse full papers by default.

A future extension could add:

```text
PDF URL extraction
PDF download
PDF text extraction
full-paper chunking
full-paper LLM analysis
```

