# IEEE and Scopus metadata adapters

Run Python from the repository root, with its existing requirements installed.
These modules also require `sources/_api_common.py` and `utils/text.py`.
No additional dependencies or API keys committed to the repository are needed.

Set `IEEE_API_KEY` and `SCOPUS_API_KEY` in your environment. Optionally set
`SCOPUS_INSTTOKEN` to an institutional token issued by Elsevier. Each function
also accepts `api_key=`, and Scopus accepts `insttoken=`. Do not commit secrets.

```python
from sources.ieee_source import search_ieee
from sources.scopus_source import search_scopus

ieee_papers = search_ieee(
    '(deepfake OR "face forgery") AND (adversarial OR evasion)',
    start_year=2018, end_year=2026,
)
scopus_papers = search_scopus(
    'TITLE-ABS-KEY((deepfake OR "face forgery") AND (adversarial OR evasion))',
    start_year=2018, end_year=2026,
    view="COMPLETE",  # Requires appropriate institutional entitlement.
)
```

Both return `list[dict]` with the existing pipeline's metadata fields, plus source
identifiers and available access information. DOI values use lowercase DOI URLs
to match OpenAlex-style identifiers. `pdf_url` is only a candidate link: its
presence does not establish open access. IEEE publication-date strings are kept
as returned because their precision/format varies.

`max_results=None` (default) retrieves all accessible matches. An explicit cap
returns a warning if it truncates the query. Search failure raises an exception,
not a successful partial list. Scopus uses offset paging and rejects searches
requiring more than 5,000 records; narrow by query/year before rerunning.

Scopus defaults to STANDARD view, which may omit abstracts and all but the first
author. COMPLETE requests richer metadata but does not guarantee every abstract
exists. A 401/403 requires checking credentials, institutional network/token and
view entitlement; the adapter never silently changes the requested view.

IEEE uses `querytext`, with at most two wildcard words and at least three
characters before each `*`. The earlier broad database queries need translation
or splitting. Both adapters take native query strings unchanged, apart from
explicit year filters. Year filters refer to publication years, not insertion dates.

## Preserve search evidence page by page

```python
import json
from pathlib import Path
from sources.ieee_source import iter_ieee_pages

# Use a fresh directory for each run to avoid overwriting previous evidence.
run_dir = Path("output/ieee_run_001")
run_dir.mkdir(parents=True, exist_ok=False)
for index, page in enumerate(iter_ieee_pages("deepfake AND adversarial"), 1):
    (run_dir / f"page_{index:05d}.json").write_text(
        json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(page["retrieved_count"], page["total_results"], page["stop_reason"])
```

`iter_scopus_pages` has the same page structure. Pages contain normalised papers,
raw responses, source-native request parameters (without credentials), timestamps,
reported totals and cumulative retrieved counts. Only a final `complete=True`
indicates that the reported result count was reached. A crash, API error or
`stop_reason="max_results"` must not be reported as an exhaustive search.

Retries cover transient network errors, HTTP 429 and common server errors. Long
Retry-After values stop with an explicit error so that the caller can schedule a
later retry. There is no persistent resume or automatic deduplication here.
Check provider quotas before large runs. Raw search responses should remain in
your research storage rather than being published without checking provider terms.

The main runner dispatches all enabled sources and archives page completion metadata.
Use explicit source queries in the version-2 protocol; see the root README.

## Offline checks

```bash
python -m unittest discover -s tests -v
```

Live authentication and account-specific entitlements require your own keys.

## Official API references

- https://developer.ieee.org/docs/read/Metadata_API_details
- https://developer.ieee.org/docs/read/metadata_api_details/Sorting_and_Paging_Parameters
- https://developer.ieee.org/docs/read/Metadata_API_responses
- https://dev.elsevier.com/documentation/SCOPUSSearchAPI.wadl
- https://dev.elsevier.com/api_key_settings.html
