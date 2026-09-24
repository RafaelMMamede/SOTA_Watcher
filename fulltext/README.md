# Full-text acquisition and extraction

This package resolves legitimate full-text copies independently of the database
that discovered the paper. A record found in Scopus or IEEE can therefore be
screened from an open repository copy without changing its discovery provenance.

## Resolution order

For each paper the resolver tries, in order:

1. A user-supplied local PDF from `local_pdfs`.
2. A direct arXiv identifier/URL when present.
3. OpenAlex, using an OpenAlex ID or DOI and its open-access locations.
4. A conservative OpenAlex title fallback when no stable identifier exists.
5. Unpaywall exact DOI lookup.
6. Semantic Scholar `openAccessPdf` exact DOI lookup.
7. Crossref DOI metadata links as **manual candidates only**.

Only PDFs explicitly identified as open/public by OpenAlex, Unpaywall, or
Semantic Scholar are automatically downloaded. Crossref full-text/TDM links are
not treated as proof of open access and are retained only for manual follow-up.
Arbitrary `pdf_url` metadata from discovery adapters is likewise not trusted.

The title fallback requires one exact normalized title match, a compatible
publication year when known, and author-token overlap when both sides provide
authors. Ambiguous matches are never attached automatically.

Configure the network resolvers in `config.yaml`:

```yaml
fulltext_resolution:
  enabled: true
  openalex: true
  unpaywall: true
  semantic_scholar: true
  crossref_metadata: true
  allow_title_fallback: true
  timeout_seconds: 20
  max_retries: 2
```

Set a real contact address in `mailto` or `UNPAYWALL_EMAIL`. An OpenAlex API
key can be supplied through `OPENALEX_API_KEY`; a Semantic Scholar key is
optional through `SEMANTIC_SCHOLAR_API_KEY`.

The screening pipeline caches each resolver outcome in
`output/papers/<paper>/resolution.json`. Set:

```yaml
screening:
  refresh_resolution: true
```

to query the resolvers again, for example when a previously unavailable paper
may have become open.

## Fetch and extract

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Fetch a versioned arXiv paper and extract its text:

```bash
python -m fulltext --arxiv 2609.10002v1 --output output/papers/2609.10002v1
```

Resolve a DOI through the open-full-text chain without downloading it:

```bash
python -m fulltext \
  --doi 10.1234/example \
  --output output/resolver_test \
  --resolve-only
```

Resolve, download, and extract when an open copy exists:

```bash
python -m fulltext \
  --doi 10.1234/example \
  --output output/papers/example
```

You can also use `--openalex W...` or an exact `--title` with optional
`--year` and `--authors` for the conservative metadata fallback.

Or import a PDF you already have:

```bash
python -m fulltext --pdf /path/to/paper.pdf --output output/papers/my-paper
```

Use a dedicated output directory per paper/version. Versioned arXiv IDs are
recommended. An unversioned ID means latest at first retrieval; subsequent runs
reuse that snapshot until `--refresh`. The downloaded file's hash identifies the
exact saved bytes.

Outputs:

- `resolution.json`: resolver selection, identifier enrichment, attempts, and
  manual candidates.
- `paper.pdf`: validated, unencrypted PDF.
- `download.json`: resolution record, final URL, retrieval time, size, page
  count, and SHA-256.
- `paper_layout.md`: Markdown separated by `<!-- PDF PAGE N -->` markers.
- `extraction_layout.json`: PDF/Markdown hashes, parser versions, settings,
  one-based PDF pages and text, empty-page warnings, extraction time.
- `download_error.json`: timestamp and exception type for a failed acquisition.

A repeated run verifies and reuses cached downloads/extractions. Changes to a
local source, PDF hash, parser version, settings, or generated Markdown invalidate
the relevant cache. `screening.refresh_pdf` fetches the selected PDF again;
`screening.refresh_resolution` re-runs the external resolver chain.

Downloads are streamed with a 100 MiB default limit and validated before atomic
replacement. Resolver-generated remote downloads require HTTPS and reject local
or private literal addresses. Existing PDFs are retained if replacement fails.

OCR is disabled explicitly for reproducibility. Scanned/empty pages are flagged
for manual review. Equations, figures, tables, and reading order may be incomplete.
Preserve the source PDF for manual reading.

## Python API

```python
from pathlib import Path
from fulltext import resolve_paper, fetch_pdf, extract_pdf

paper = {
    "doi": "https://doi.org/10.1234/example",
    "title": "Example paper",
    "year": 2026,
}

resolution = resolve_paper(
    paper,
    resolver_config={"email": "researcher@example.org"},
)

folder = Path("output/papers/example")
record = fetch_pdf(resolution, folder)

if record["status"] == "downloaded":
    extraction = extract_pdf(folder / "paper.pdf")
```

Unresolved papers return `status="unavailable"`, retain resolver attempts and
manual candidates, and remain uncertain in eligibility screening. Invalid local
paths, bad PDFs, HTTP download failures, and parser failures raise explicit
errors.

The main runner calls this package through `screening.pipeline` when
`screening.enabled` is true. Eligibility decisions remain the responsibility of
the screening module and human reviewer.
