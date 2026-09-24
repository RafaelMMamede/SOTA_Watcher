# Full-text acquisition and extraction

This package is independent of metadata discovery and Ollama. It supports arXiv
identifiers/URLs and local PDFs. Publisher DOI/open-access resolution is not yet
implemented; an arbitrary metadata `pdf_url` is not assumed freely accessible.

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Fetch a versioned arXiv paper and extract its text:

```bash
python -m fulltext --arxiv 2609.10002v1 --output output/papers/2609.10002v1
```

Or import a PDF you already have:

```bash
python -m fulltext --pdf /path/to/paper.pdf --output output/papers/my-paper
```

Use a dedicated output directory per paper/version. Versioned arXiv IDs are
recommended. An unversioned ID means latest at first retrieval; subsequent runs
reuse that snapshot until `--refresh`. The resolver records whether the requested
version was pinned, but does not determine the resolved version of an unversioned
request. The downloaded file's hash identifies the exact saved bytes.

Outputs:

- `paper.pdf`: validated, unencrypted PDF.
- `download.json`: original resolution, final URL, retrieval time, size, page count,
  and SHA-256. Local imports record their source path.
- `paper_layout.md`: Markdown separated by `<!-- PDF PAGE N -->` markers.
- `extraction_layout.json`: PDF/Markdown hashes, parser versions, settings,
  one-based PDF pages and text, empty-page warnings, extraction time.
- `resolution.json`: unavailable resolution status when metadata cannot resolve.
- `download_error.json`: timestamp and exception type for a failed acquisition.
  This is an error-event record, not the status of the last valid cached PDF.

A repeated run verifies and reuses cached downloads/extractions. Changes to a local
source, PDF hash, parser version, settings, or generated Markdown invalidate the
relevant cache. `--refresh` fetches again; `--force-extract` regenerates extraction.
Downloads are streamed with a 100 MiB default limit and validated before atomic
replacement. Individual output files are written atomically. Run only one process
per paper directory (no cross-process transaction/locking is provided).

No automatic download retries or bulk scheduling: failures are explicit and may
be retried later. When building a batch caller, respect arXiv's request pacing.
Existing PDFs are retained if a replacement fetch fails. After replacement, rerun
extraction to bring derived text up to date; the CLI does this automatically.

OCR is disabled explicitly for reproducibility and to avoid an implicit OCR
installation requirement. Scanned/empty pages are flagged for review. Extraction
is a preliminary reading aid: equations, figures, tables, and reading order may be
incomplete. Page count and nonempty text are not guarantees of extraction quality.
Preserve the PDF for manual reading. Render citations in LLM outputs as `(PDF p. N)`,
not HTML comments.

## Python API

```python
from pathlib import Path
from fulltext import resolve_paper, fetch_pdf, extract_pdf

folder = Path('output/papers/2609.10002v1')
resolution = resolve_paper({'arxiv_id': '2609.10002v1'})
record = fetch_pdf(resolution, folder)
if record['status'] == 'downloaded':
    extraction = extract_pdf(folder / 'paper.pdf')
```

`resolve_paper` also accepts the dictionaries returned by the discovery adapters.
Unresolved papers return `status='unavailable'`, not an empty successful extraction.
Invalid local paths, bad PDFs, HTTP errors, and parser failures raise exceptions.
The main runner now calls this package through `screening.pipeline` when
`screening.enabled` is true. This package itself only resolves, downloads and
extracts PDFs; eligibility decisions are handled by the screening module.
