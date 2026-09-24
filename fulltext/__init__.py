"""Full-text acquisition and extraction, independent of discovery and LLMs."""
from .resolve import resolve_paper
from .download import fetch_pdf
from .extract import extract_pdf

__all__ = ['resolve_paper', 'fetch_pdf', 'extract_pdf']
