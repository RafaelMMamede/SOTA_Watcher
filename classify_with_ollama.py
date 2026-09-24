"""Retired abstract-only triage API. Use screening.screen_papers instead."""


def classify_papers_with_ollama(papers, config):
    raise RuntimeError('Abstract read/skim/ignore triage is retired. Configure screening and run sota_watcher.py for full-text eligibility.')
