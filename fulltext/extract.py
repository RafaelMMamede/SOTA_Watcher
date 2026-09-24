"""Page-level Markdown for preliminary reading, not lossless PDF conversion."""
from importlib.metadata import version
from pathlib import Path
from ._common import now, read_json, sha256, validate_pdf, write_json, write_text


def extract_pdf(pdf, output_dir=None, *, force=False):
    import pymupdf4llm
    pdf = Path(pdf)
    folder = Path(output_dir) if output_dir else pdf.parent
    folder.mkdir(parents=True, exist_ok=True)
    count = validate_pdf(pdf)
    settings = {'page_chunks': True, 'use_ocr': False}
    identity = {'schema_version': 1, 'pdf_sha256': sha256(pdf),
                'parser': 'pymupdf4llm', 'parser_version': version('pymupdf4llm'),
                'pymupdf_version': version('pymupdf'), 'settings': settings}
    manifest = folder / 'extraction_layout.json'
    markdown_path = folder / 'paper_layout.md'
    previous = read_json(manifest)
    if not force and previous and all(previous.get(k) == v for k, v in identity.items()):
        if markdown_path.is_file() and sha256(markdown_path) == previous.get('markdown_sha256'):
            return {**previous, 'cache_hit': True}
    chunks = pymupdf4llm.to_markdown(str(pdf), **settings)
    if not isinstance(chunks, list) or len(chunks) != count:
        raise RuntimeError('Expected one extraction chunk per PDF page.')
    pages = []
    for index, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict) or not isinstance(chunk.get('text'), str):
            raise RuntimeError(f'Invalid extraction chunk at PDF page {index}.')
        pages.append({'page': index, 'text': chunk['text']})
    markdown = '\n\n'.join(f"<!-- PDF PAGE {p['page']} -->\n\n{p['text']}" for p in pages)
    empty = [p['page'] for p in pages if not p['text'].strip()]
    record = {**identity, 'extracted_at': now(), 'page_numbering': '1-based PDF page order',
              'pages': pages, 'page_count': count, 'empty_pages': empty,
              'status': 'needs_review' if empty else 'extracted',
              'warnings': ['OCR disabled. Equations, tables, figures and reading order may be incomplete.']}
    write_text(markdown_path, markdown)
    record['markdown_sha256'] = sha256(markdown_path)
    write_json(manifest, record)
    return {**record, 'cache_hit': False}
