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

## Persistent workflow

SQLite is the authoritative working state (`corpus_db_path`, default
`output/sota_corpus.sqlite3`). Excel remains the human review/export interface and
PDF/JSON artifacts remain the screening evidence. Discovery and screening are
independently restartable:

```bash
# One-time migration of an existing reviewed workbook.
python sota_watcher.py import-workbook output/sota_table.xlsx

# Retrieve and persist candidates. Re-running with --resume continues unfinished
# source/query/date partitions and skips completed work.
python sota_watcher.py discover --resume

# Inspect discovery, screening, full-text and human-review state.
python sota_watcher.py status

# High-recall title/abstract eligibility screening.
python sota_watcher.py screen-metadata --limit 100

# Then process only metadata include/uncertain papers at full text.
python sota_watcher.py screen --limit 100

# Retry only prior failures/unavailable full text when desired.
python sota_watcher.py screen --limit 100 --retry screening
python sota_watcher.py screen --limit 100 --retry extraction
python sota_watcher.py screen --limit 100 --retry fulltext_unavailable
# "error" remains available as an explicit catch-all for all error stages.

# Pull human edits from the existing workbook, then export current SQLite state.
python sota_watcher.py export
```

Each candidate receives a stable internal `corpus_id`. DOI, arXiv, OpenAlex,
Scopus and provider IDs are aliases, so later identifier enrichment does not move
the paper's artifact directory or reset its human decision. Discovery merges new
provenance into the corpus before screening applicability is evaluated.

Running `python sota_watcher.py` with no subcommand still executes the former
single-run workflow for compatibility. Use the subcommands above for unattended
or production work.

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
excludes. A criterion may also define `applies_to_search_topics` with one or more
search IDs. Such a criterion is screened only for candidates retrieved by at least
one of those search families; criteria without this field are universal. A
criterion may also define `skip_if_search_topics`: any matching retrieval family
makes that criterion inactive. Unknown search IDs fail protocol validation. The
active and inactive criteria for each paper are retained in the screening
artifacts/table. Full-text model decisions are provisional; human decisions are
separate.

For example, the review is a union of visual-forgery and adversarial-vision scope.
A paper retrieved by both routes can still be eligible because of its deepfake
detection content, so a GAN-only false-positive exclusion is applied only to
adversarial-only candidates:

```yaml
- id: generative_adversarial_only
  kind: exclusion
  applies_to_search_topics: [adversarial_vision]
  skip_if_search_topics: [visual_forgery_detection]
  description: For candidates retrieved only through the adversarial-vision family, uses "adversarial" only for generative adversarial networks or generative-model training, without studying adversarial examples, evasion attacks, perturbations, defenses, or robustness.
```

Legacy `topics` YAML still loads with a warning. Weighted topic terms and old
`min_triage_score`, repository filters, and `drop_existing_below_min_score` settings
no longer remove any records. Old `ollama` and `ollama_deep_analysis` settings are
not used: migrate to `screening`. The legacy Python triage entry points now fail
with migration instructions rather than returning obsolete labels.

## Discovery

```bash
python sota_watcher.py discover --resume
```

The persistent runner freezes the effective retrieval configuration/dates for a
run and tracks completion separately for each source, search-family query and date
partition. Screening criteria are deliberately excluded from discovery identity,
so changing eligibility rules does not rerun retrieval. Each provider page, its
normalized candidate records, raw response and the checkpoint for the **next**
request are committed in one SQLite transaction.
A crash therefore either commits both the page and checkpoint or neither.

OpenAlex and Scopus resume from saved cursors; IEEE resumes from its saved offset.
If a saved cursor has expired, only that affected partition is restarted and
already merged candidates remain deduplicated in the corpus. A provider failure
marks that task/run incomplete but does not discard successful pages or stop other
providers from producing usable candidates. `--resume` retries unfinished/error
tasks and skips completed tasks whose frozen configuration still matches.

First use `max_results_per_query: 3` for a smoke test; caps are explicitly
incomplete. Remove them for the production search. `max_results_per_query: null`
exhausts configured searches.

OpenAlex searches title and abstract only and applies exact publication-date
bounds. IEEE uses inclusive publication years and warns about the coarser
precision. Scopus uses provider-side year bounds as a prefilter, then enforces the
exact configured YYYY-MM-DD range locally on `prism:coverDate`. Cursor pagination
is preferred. If Scopus must fall back to offset pagination and a partition
exceeds the 5,000-source-record ceiling, that date interval is recursively split
until each child can be exhausted. If even one day still exceeds the limit, that
leaf is explicitly recorded as an error/incomplete coverage rather than silently
truncated.

### arXiv date semantics

arXiv uses OAI-PMH (`arXivRaw`) and local Boolean matching, not the legacy search
endpoint. Local `all:` means title+abstract, with case-insensitive substring
matching. `ti:`, `abs:`, phrases, AND/OR/ANDNOT and parentheses are supported;
wildcards and other native API fields are rejected. This is not identical to
arXiv's web search semantics.

OAI from/until select **metadata modification dates**. `from_publication_date` and
`to_publication_date` filter original v1 submission dates recovered from version
history. For a historical backfill, the harvest starts at the publication lower
bound and ends at the frozen run date, so older papers updated after the
publication cutoff remain eligible. The OAI metadata window is harvested **once**
and stored locally; every configured arXiv query is then evaluated against that
shared collection. Adding or revising a query can reuse an already completed
matching metadata harvest rather than downloading the same window again.

Explicit `oai_from_date`/`oai_until_date` configure recurring update monitoring;
records within that update window do not constitute a historical census. Use an
inclusive overlap on your last successful day and let identifier merging remove
duplicates. The tool logs update-window scope, OAI datestamps, version dates and
deleted headers separately. Tombstones are logged for human review; they do not
automatically delete existing table records. No exact point-in-time snapshot is
guaranteed for a changing remote repository. Missing v1 dates fail the harvest.
A result/page cap is explicitly incomplete; matching totals are unknown until
local harvesting ends. Capped OAI results are in harvest order, not globally ranked.

## Title and abstract screening

Run metadata eligibility after discovery and before PDF retrieval:

```bash
python sota_watcher.py screen-metadata --limit 100
```

This stage uses only the saved title and abstract plus the provenance-aware
eligibility criteria. It is deliberately high recall: missing detail is
`uncertain`, not evidence for exclusion. A paper is excluded at this stage only
when the title/abstract explicitly demonstrates a failed inclusion criterion or a
met exclusion criterion. Both `include` and `uncertain` advance to full-text
screening.

Metadata assessments have independent SQLite state, artifacts and signatures.
Successive batches skip valid completed assessments. A changed title/abstract,
model/prompt/settings, eligibility criteria, or search-family provenance makes the
old result stale and returns the paper to the metadata queue. Retry prior metadata
errors explicitly:

```bash
python sota_watcher.py screen-metadata --limit 100 --retry error
```

The default full-text queue requires a current metadata assessment
(`screening.require_metadata_screening: true`). Clear metadata exclusions do not
trigger PDF resolution/download. A human `manual_decision: include` can still
advance a record, while a human `manual_decision: exclude` blocks full-text work.

## Full-text screening

Review the eligibility criteria and ensure your configured Ollama model is
installed and the server is running. By default, the standalone `screen` command
selects only papers whose current title/abstract assessment is `include` or
`uncertain`; metadata `exclude` records are skipped before PDF retrieval.
The command enables full-text screening for its selected batch;
`screening.enabled` remains relevant to the legacy no-subcommand workflow. The
initial model setting is `qwen3.5:9b` with a 32K context. Screening uses a fast-first pipeline: a no-thinking 2K primary pass
receives the JSON schema in the prompt and is checked locally; only invalid fast
output is retried with Ollama structured output, low thinking, and an 8K fallback
generation budget. A single surrounding Markdown JSON fence is tolerated before
the same strict schema and page-evidence validation is applied.

`python sota_watcher.py screen --limit N` counts only papers that actually require
processing. Valid completed assessments are skipped, so successive batches advance
through the backlog instead of repeatedly selecting the first N corpus rows.
Prior `error` and `fulltext_unavailable` records are not retried automatically.
Use `--retry screening`, `extraction`, `download`, or `resolution` to target
one failed stage; `--retry error` is the explicit all-error catch-all, and
`--retry fulltext_unavailable` rechecks unavailable full text. Progress is written
after each paper. Batch output includes completed/error/unavailable counts,
papers/minute and an estimated remaining time.

A completed screening result is reusable only while its screening signature still
matches the PDF SHA-256, Ollama model digest, prompt version, inference settings
and currently applicable eligibility criteria. A changed mapped local PDF,
`refresh_pdf`/`refresh_resolution`, model/prompt changes, or new discovery
provenance that changes criterion applicability returns that paper to the queue.

Full-text acquisition is independent of discovery:
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

The fast primary deliberately uses `think: false` **without** Ollama's server-side
`format` constraint. A dedicated fast-screening prompt explicitly distinguishes
inclusion from exclusion semantics, requires `uncertain` when the supplied pages
do not settle a criterion, and requires evidence to name the page containing the
quote. The returned text must still pass the strict local JSON/schema/evidence
validator. This avoids spending large reasoning budgets on routine chunks while
containing the Qwen/Ollama structured-output behavior previously observed with
no-thinking mode.

If the fast pass is invalid, the same chunk is retried once with
`fallback_think: low`, Ollama structured output, and
`fallback_num_predict: 8192`. With thinking enabled, that generation budget can
be consumed by reasoning. If and only if this fallback ends with
`done_reason="length"`, the chunk is split in half and both children re-enter the
fast-first pipeline. Multi-page chunks split between consecutive pages; a
single-page chunk splits its text while preserving the PDF page number. Splitting
is bounded by `max_split_depth` (default 6). Schema/evidence failures do not
trigger recursive splitting. Fast, fallback, split, and child artifacts are saved
separately; no invalid or truncated response is accepted as a screening result.

The explicit configuration keys are `fast_num_predict`, `fallback_enabled`,
`fallback_think`, `fallback_num_predict`, and `max_split_depth`. Legacy
`think`, `num_predict`, `repair_invalid_output`, and `repair_num_predict`
settings remain accepted as aliases for existing configurations.

Extracted pages are greedily packed in order into bounded multi-page parts for
`/api/chat`. Whole pages are kept together whenever they fit; only a single page
that exceeds the entire part budget is split, with its page number preserved on
every fragment. No text is silently dropped or reordered. A conservative UTF-8
byte budget reserves space for instructions/schema and the larger reasoning
fallback output; this is not a model-specific tokenizer. Each part uses the same
criterion schema. Definite assessments require grounded evidence from a supplied
PDF page. Safe PDF/model typography differences are canonicalized for matching:
curly single quotes may match straight apostrophes, curly double quotes may match
straight double quotes, and non-breaking spaces are treated as ordinary whitespace.
The match must be unique and the stored quote uses the source spelling. Content
changes such as omitted punctuation, changed words, or paraphrases still fail.
If a model gives the wrong page number but its quote resolves to exactly one
supplied page, the validator canonicalizes the evidence to that page; ambiguous
cross-page matches still fail. Invalid JSON, invented quotes, ambiguous evidence,
missing criteria, transport failure and empty extraction stay uncertain; fallback
length exhaustion is first handled by bounded adaptive splitting as described above.
Conflicting evidence across parts becomes uncertain for that criterion. Absence
of evidence in a part must not be interpreted as a failed criterion.

Saved screening artifacts include model digest, prompt/schema, criteria, PDF and
text hashes, timestamps, part requests/responses and validated results. Cache keys
change with model digest, PDF text, criteria and inference settings. Successful
parts can resume after failure. Set `screening.force: true` to regenerate output.
Evidence checks establish quote presence, not that a quote logically proves the
criterion. Human review is still required.

## Data and reports

The persistent corpus stores:

- canonical candidate metadata plus stable `corpus_id` and external identifier aliases;
- all source/query/search-family provenance;
- independent human `manual_decision`, `manual_reason` and `notes`;
- full-text and screening state/signatures/errors;
- discovery runs, tasks, page envelopes, raw responses and next-page checkpoints;
- shared arXiv harvest records/pages; and
- screening batch throughput/ETA metrics.

`python sota_watcher.py status` reports persistent discovery run/task/arXiv-harvest
states, committed page/record totals, screening/full-text counts, retry-required
counts, screening error stages, human decisions and the last screening batch's
rate/ETA.

`python sota_watcher.py export` first imports human fields from an existing Excel
file without importing its stale model/full-text state, then atomically writes the
current corpus. Use `import-workbook` for the one-time migration when you *do*
want existing screening/full-text columns loaded into the new database. Human
decisions remain independent from provisional model decisions.

Structured Excel cells contain JSON; the loader restores them. Oversized cells
fail explicitly rather than silently truncating. Deduplication uses connected
normalized identifiers and only falls back to exact title+year when no stronger
identifier exists. Discovering a new identifier/source match merges into the same
stable corpus record.

The legacy no-subcommand workflow still writes per-run
`output/discovery_runs/<run_id>/` JSON/event snapshots. Those artifacts remain
useful for compatibility, but the SQLite page/task state is the authoritative
restart point for the new subcommand workflow.

## Verification

```bash
python -m py_compile sota_watcher.py screening/queue.py sources/restartable.py sources/arxiv_cache.py utils/corpus_store.py
python -m unittest tests.test_corpus_store tests.test_restartable_discovery -v
python -m unittest discover -s tests -q
```

Regression tests cover advancing screening batches, stable IDs/workbook migration,
provenance-triggered re-screening, interrupted discovery resume, provider failure
isolation, shared arXiv harvesting, and Scopus partitioning. Tests use mocked
APIs/Ollama plus synthetic PDFs and Excel round-trips; they do not establish live
key entitlements, retrieval recall, or model accuracy.

References: [OpenAlex paging](https://help.openalex.org/api/paging/),
[arXiv OAI](https://info.arxiv.org/help/oa/index.html),
[Ollama structured outputs](https://ollama.com/blog/structured-outputs).
