"""Resolve arXiv metadata or a user-supplied local PDF without network calls."""
import re
from pathlib import Path
from urllib.parse import urlsplit

ARXIV_ID = re.compile(r'(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.\-]*/\d{7})(?:v[1-9]\d*)?')


def arxiv_id(value):
    value = str(value or '').strip()
    if value.startswith('arxiv:'):
        value = value[6:]
    if '://' in value:
        parsed = urlsplit(value)
        if parsed.hostname not in {'arxiv.org', 'www.arxiv.org', 'export.arxiv.org'}:
            return None
        value = re.sub(r'^/(?:abs|pdf)/', '', parsed.path)
    value = value.removesuffix('.pdf')
    return value if ARXIV_ID.fullmatch(value) else None


def resolve_paper(paper=None, *, local_pdf=None):
    """Return a resolution record; unknown sources are explicitly unavailable.

    An unversioned arXiv identifier resolves the latest version at retrieval time.
    Pass a versioned identifier for reproducible selection. Arbitrary publisher
    pdf_url fields are not treated as proof of open access.
    """
    paper = paper or {}
    if local_pdf is not None:
        path = Path(local_pdf).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return {'status': 'resolved', 'kind': 'local', 'path': str(path)}
    for field in ('arxiv_id', 'paper_id', 'url', 'pdf_url'):
        identifier = arxiv_id(paper.get(field))
        if identifier:
            return {'status': 'resolved', 'kind': 'arxiv', 'arxiv_id': identifier,
                    'version_pinned': bool(re.search(r'v\d+$', identifier)),
                    'url': f'https://arxiv.org/pdf/{identifier}'}
    return {'status': 'unavailable', 'reason': 'No arXiv identifier or local PDF supplied.'}
