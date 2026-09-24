# SOTA Watcher

Systematic paper discovery, auditable metadata merging, and provisional full-text
eligibility screening with local Ollama. Every retrieved candidate is retained.
Keyword scores and read/skim/ignore labels no longer control inclusion.

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp search_terms.example.yaml search_terms.yaml
cp .env.example .env
chmod 600 .env
```

Edit `.env` with your keys. `main()` loads it without overriding exported values.
IEEE requires an approved key; Scopus STANDARD may omit abstracts. Set
OPENALEX_API_KEY if required by your access tier. Credentials are excluded from
archives and `.env` is ignored by Git. Never put keys in query text or tracked YAML.

## Define the review protocol

The version-2 search YAML separates `review`, `searches` and `eligibility`.
Each search has a stable `id`, label and explicit query lists for each source.
There is no automatic cross-database Boolean translation. A missing/empty source
query list skips that search for the source. An enabled source with no queries
is reported as skipped, not successfully searched.

The example reflects two retrieval families: visual forgery detection and
adversarial computer vision. It is a starting point, not a validated systematic
search strategy. Pilot it against known relevant papers, revise synonyms/fields,
and freeze the final protocol before the production search.

Eligibility criteria have a unique `id`, `kind` (`inclusion` or `exclusion`) and
`description`. All inclusion criteria must be met; any met exclusion criterion
excludes. Full-text model decisions are provisional; human decisions are separate.

Legacy `topics` YAML still loads with a warning. Weighted topic terms and old
`min_triage_score`, repository filters, and `drop_existing_below_min_score` settings
no longer remove any records. Old `ollama` and `ollama_deep_analysis` settings are
not used: migrate to `screening`. The legacy Python triage entry points now fail
with migration instructions rather than returning obsolete labels.

## Discovery

```bash
python sota_watcher.py
```

First use `max_results_per_query: 3`, `screening.enabled: false`, and a separate
`sota_table_path` for a smoke test. An arXiv OAI smoke test should use a recent
explicit `oai_from_date`/`oai_until_date` and `max_pages: 1`; this is an incomplete
update-window test, not a historical search. Then remove caps for production.

`max_results_per_query: null` follows pagination to exhaustion. Per-source
`max_results` overrides the global cap. OpenAlex uses cursor pagination with
pages up to 100. Scopus prefers cursor pagination. If cursor access is rejected,
`pagination_mode: auto` falls back to offset paging using the configurable
`offset_page_size` (default 25) rather than assuming the account accepts the
documented STANDARD maximum of 200. Offset fallback explicitly retains the
5,000-source-record limit. Result totals changing mid-pagination or repeated
records/cursors cause failures rather than silent partial success. IEEE retains
its documented paging limits. Source access/rate limits still apply.

OpenAlex searches title and abstract only and applies exact publication-date
bounds. IEEE uses inclusive publication years and warns about the coarser precision.
Scopus uses provider-side year bounds as a prefilter, then enforces the exact
configured YYYY-MM-DD range locally on `prism:coverDate`. Adapter request
parameters are saved; OpenAlex defaults an omitted upper bound to the day the run starts.

### arXiv date semantics

arXiv uses OAI-PMH (`arXivRaw`) and local Boolean matching, not the legacy search
endpoint. Local `all:` means title+abstract, with case-insensitive substring
matching. `ti:`, `abs:`, phrases, AND/OR/ANDNOT and parentheses are supported;
wildcards and other native API fields are rejected. This is not identical to
arXiv's web search semantics.

OAI from/until select **metadata modification dates**. `from_publication_date` and
`to_publication_date` filter original v1 submission dates recovered from version
history. For a historical backfill, the harvest starts at the publication lower
bound and ends today, so older papers updated after the publication cutoff remain
eligible. This may require a large all-subject harvest for each query.

Explicit `oai_from_date`/`oai_until_date` configure recurring update monitoring;
records within that update window do not constitute a historical census. Use an
inclusive overlap on your last successful day and let identifier merging remove
duplicates. The tool logs update-window scope, OAI datestamps, version dates and
deleted headers separately. Tombstones are logged for human review; they do not
automatically delete existing table records. No exact point-in-time snapshot is
guaranteed for a changing remote repository. Missing v1 dates fail the harvest.
A result/page cap is explicitly incomplete; matching totals are unknown until
local harvesting ends. Capped OAI results are in harvest order, not globally ranked.

## Full-text screening

Set `screening.enabled: true`, review the eligibility criteria, then ensure your
configured Ollama model is installed and the server is running. The initial
model setting is `qwen3.5:9b`, low thinking, 32K context, and an 8K generation budget. The request supplies
the JSON schema both through Ollama's `format` field and in the prompt. A single
surrounding Markdown JSON fence is tolerated before the same strict schema and
page-evidence validation is applied.

All current candidates are processed, up to `max_papers_per_run`; the remainder
are `deferred` and retained. Full-text acquisition is independent of discovery:
a Scopus/IEEE/OpenAlex record may be screened from a legitimate open repository
copy. Resolution order is local mapping, arXiv, OpenAlex OA locations, Unpaywall
DOI lookup, Semantic Scholar `openAccessPdf`, then Crossref metadata links for
manual follow-up. A conservative exact-title/year/author-checked OpenAlex lookup
can enrich papers that lack stable identifiers.

Only resolver-verified open/public PDF URLs are downloaded automatically.
Crossref/TDM links and arbitrary provider `pdf_url` fields are never assumed open.
Unavailable full text remains **uncertain**, never excluded. Resolver attempts,
selected source/version/license, identifier enrichment and manual candidates are
retained in the paper artifacts and Excel table. Set `UNPAYWALL_EMAIL` when the
main `mailto` is not your contact address; `SEMANTIC_SCHOLAR_API_KEY` is optional.

Downloads, resolver outcomes and page extraction reuse validated caches. Set
`screening.refresh_resolution: true` to re-check external OA sources and
`screening.refresh_pdf: true` to re-fetch the selected PDF. Versioned arXiv PDF
URLs are preferred when metadata supplies them. See
[fulltext/README.md](fulltext/README.md) for resolver and standalone PDF details.
Extraction disables OCR; empty/scanned pages require manual review. Tables,
equations, figures and reading order may remain imperfect.

Qwen 3.5 users should avoid `think: false` when relying on Ollama structured
outputs; some Ollama/Qwen 3.5 combinations have returned unconstrained prose in
that mode. The example configuration therefore uses `think: low`.

With thinking enabled, `num_predict` is the maximum generated-token budget for the
request, so reasoning can consume part of it. The example uses `num_predict: 8192`.
If the primary response truncates or fails strict schema/evidence validation, the
pipeline performs at most one compact repair call (`repair_num_predict: 2048`)
with thinking disabled and no server-side `format` constraint. The repair prompt
still includes the schema and supplied pages, and its output must pass the same
local exact-quote/schema validator. Primary and repair artifacts are retained
separately; no invalid or truncated response is accepted as a screening result.

Every extracted page is sent in bounded parts to `/api/chat`. No pages are silently
truncated. A conservative UTF-8 byte budget leaves space for instructions/schema
and output; this is not a model-specific tokenizer. Larger papers can require many
calls. Each part uses the same criterion schema. Definite assessments require an
exact quote from a supplied PDF page. Invalid JSON, invented quotes/pages, missing
criteria, truncated output, transport failure and empty extraction stay uncertain.
Conflicting evidence across parts becomes uncertain for that criterion. Absence
of evidence in a part must not be interpreted as a failed criterion.

Saved screening artifacts include model digest, prompt/schema, criteria, PDF and
text hashes, timestamps, part requests/responses and validated results. Cache keys
change with model digest, PDF text, criteria and inference settings. Successful
parts can resume after failure. Set `screening.force: true` to regenerate output.
Evidence checks establish quote presence, not that a quote logically proves the
criterion. Human review is still required.

## Data and reports

Each run creates `output/discovery_runs/<run_id>/`:

- `search_terms.json`, `search_plan.json`: protocol and effective source settings.
- `events.jsonl`: page records, outcomes, partial progress and screening events.
- `raw_<query>_<page>.json`: provider JSON or OAI XML (disable with `save_raw_responses: false`).
- `retrieval_summary.json`: totals, caps, completion/failure, and scope per query.
- `discovered.json`, `deduplicated.json`, `screening.json`: stage snapshots.
- `merged_table_before_filter.json`: accumulated table snapshot (no filtering).
- `review_counts.json`, `review_summary.md`: counts and evidence for this run.

Failed retrieval stops the run. Previously received pages stay in the archive;
later-stage snapshots may not exist. `run_finished` is not evidence of exhaustive
retrieval: inspect completion and scope fields. A killed process may leave a
started event without a terminal event. Source completion is independent of
eligibility decisions, and missing sources are not zero-result searches.

Excel includes source/query provenance, full-text status, proposed eligibility,
criterion assessments, page evidence, model/PDF identity, and independent human
`manual_decision`, `manual_reason`, and `notes`. Human decisions are `include`,
`exclude` or `uncertain`. Existing human values survive updates. New screening
replaces the prior assessment as a whole; disabled/deferred screening does not
wipe an existing assessment. No record is dropped because it is excluded.

Structured Excel cells contain JSON; the loader restores them. Oversized cells
fail explicitly rather than silently truncating; the JSON archive remains intact.
Excel writes are atomic. Deduplication uses connected normalized identifiers,
retains conflicts and provenance, and only uses exact title+year when no stronger
identifier is available. This conservative matching can leave duplicates for
manual review. Longer text is a completeness heuristic, not a quality guarantee.

Counts separate raw records, duplicate records, unique candidates, full-text
statuses, model decisions and human decisions. They are **not a validated PRISMA
flow** and are not automatically study-level counts. Per-run counts and the
accumulated table are different populations; do not add overlapping run counts.

Regenerate the accumulated summary after editing human decisions:

```bash
python summarize_recommended.py
```

This command retains all decisions, including uncertain/excluded records. It
replaces the old read/skim-only summary generator.

## Verification

```bash
python -m unittest discover -s tests -q
```

Tests use mocked APIs/Ollama plus actual synthetic PDFs and Excel round-trips.
They do not establish live key entitlements, retrieval recall, or model accuracy.

References: [OpenAlex paging](https://help.openalex.org/api/paging/),
[arXiv OAI](https://info.arxiv.org/help/oa/index.html),
[Ollama structured outputs](https://ollama.com/blog/structured-outputs).
