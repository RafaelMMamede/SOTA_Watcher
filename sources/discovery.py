"""Dispatch configured discovery sources; validate the plan before requests."""
from __future__ import annotations

from datetime import date
import inspect
import warnings

from utils.discovery_log import utc_now

from sources.openalex_source import search_openalex
from sources.arxiv_source import search_arxiv
from sources.ieee_source import search_ieee, iter_ieee_pages
from sources.scopus_source import search_scopus, iter_scopus_pages


def get_queries_from_search_terms(search_terms: dict, source: str = "openalex") -> list[tuple[str, str]]:
    items = []
    for topic, settings in search_terms.get("topics", {}).items():
        overrides = settings.get("source_queries", {})
        unknown = set(overrides) - {"openalex", "arxiv", "ieee", "scopus"}
        if unknown:
            raise ValueError(f"Unknown source_queries: {sorted(unknown)}")
        if source == "scopus" and source not in overrides:
            raise ValueError(f"Topic {topic}: add source_queries.scopus using native Scopus syntax (or [] to skip).")
        queries = overrides.get(source, settings.get("queries", []))
        if not isinstance(queries, list) or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError(f"Topic {topic}: queries for {source} must be a list of nonempty strings.")
        items.extend((topic, q) for q in queries)
    return items


def fetch_papers(config: dict, search_terms: dict, *, audit=None) -> list[dict]:
    adapters = {"openalex": search_openalex, "arxiv": search_arxiv,
                "ieee": search_ieee, "scopus": search_scopus}
    signatures = {"openalex": search_openalex, "arxiv": search_arxiv,
                  "ieee": iter_ieee_pages, "scopus": iter_scopus_pages}
    enabled = config.get("sources", ["openalex"])
    if not isinstance(enabled, list) or not enabled or any(not isinstance(s, str) or s not in adapters for s in enabled):
        raise ValueError("sources must be a nonempty list containing openalex, arxiv, ieee or scopus.")
    if len(set(enabled)) != len(enabled):
        raise ValueError("sources contains duplicates.")
    source_options = config.get("source_options", {})
    if not isinstance(source_options, dict) or set(source_options) - set(adapters):
        raise ValueError("source_options must map supported source names to options.")
    plan = []
    for source in enabled:
        options = source_options.get(source, {})
        if not isinstance(options, dict):
            raise ValueError(f"source_options.{source} must be a mapping.")
        allowed = set(inspect.signature(signatures[source]).parameters) - {"query", "session", "api_key", "insttoken"}
        if set(options) - allowed:
            raise ValueError(f"Unsupported options for {source}: {sorted(set(options) - allowed)}. Set credentials in environment variables.")
        kwargs = {"max_results": config.get("max_results_per_query", 25)}
        if source == "openalex":
            kwargs.update({k: config[k] for k in ("mailto", "from_publication_date", "to_publication_date", "sleep_seconds") if k in config})
        elif source == "arxiv":
            kwargs.update({k: config[k] for k in ("from_publication_date", "to_publication_date") if config.get(k)})
            # Never inherit the one-second OpenAlex delay for arXiv.
            kwargs["sleep_seconds"] = max(3.0, config.get("sleep_seconds", 3.0))
        else:
            kwargs["sleep_seconds"] = config.get("sleep_seconds", 1.0)
            for global_key, year_key in (("from_publication_date", "start_year"), ("to_publication_date", "end_year")):
                if config.get(global_key) and year_key not in options:
                    value = date.fromisoformat(str(config[global_key]))
                    kwargs[year_key] = value.year
                    warnings.warn(f"{source}: {global_key} uses the whole inclusive year {value.year}; day/month precision is unavailable.", stacklevel=2)
        kwargs.update(options)
        cap = kwargs["max_results"]
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError(f"{source}: max_results must be a positive integer.")
        if source in {"openalex", "arxiv"} and cap is None:
            raise ValueError(f"{source}: a finite max_results is required.")
        if source == "openalex" and cap > 200:
            raise ValueError("OpenAlex adapter currently supports at most 200 results per query.")
        plan.append((source, kwargs, get_queries_from_search_terms(search_terms, source)))

    if audit:
        audit.snapshot('search_plan', [{'source': source, 'options': kwargs,
                        'queries': [{'topic': topic, 'query': query} for topic, query in queries]}
                       for source, kwargs, queries in plan])
    papers = []
    query_number = 0
    for source, kwargs, queries in plan:
        for topic, query in queries:
            query_number += 1
            context = {'query_id': str(query_number), 'source': source,
                       'search_topic': topic, 'query': query}
            print(f"\nSearching {source} [{topic}]: {query}")
            if audit:
                audit.event('query_started', **context, options=kwargs)
            coverage = 'not_verified'
            try:
                if audit and source in {'ieee', 'scopus'}:
                    iterator = iter_ieee_pages if source == 'ieee' else iter_scopus_pages
                    results = []
                    for page_number, page in enumerate(iterator(query=query, **kwargs), 1):
                        rows = [{**paper, 'search_topic': topic, 'retrieved_at': utc_now(),
                                 'run_id': audit.run_id, 'query_id': str(query_number)}
                                for paper in page['papers']]
                        audit.event('query_page', **context, page_number=page_number,
                                    request_params=page['request_params'],
                                    total_results=page['total_results'], complete=page['complete'],
                                    stop_reason=page['stop_reason'], records=rows)
                        results.extend(rows)
                        coverage = 'exhausted' if page['complete'] else page['stop_reason']
                        if page['stop_reason'] == 'max_results':
                            warnings.warn(f"{source}: search capped at {len(results)} of {page['total_results']} matches.", stacklevel=2)
                else:
                    results = adapters[source](query=query, **kwargs)
            except BaseException as exc:
                if audit:
                    audit.event('query_failed', **context, error_type=type(exc).__name__)
                raise
            timestamp = utc_now()
            results = [{**paper, 'search_topic': topic, 'retrieved_at': paper.get('retrieved_at', timestamp),
                        **({'run_id': audit.run_id, 'query_id': str(query_number)} if audit else {})}
                       for paper in results]
            if audit:
                audit.event('query_succeeded', **context, retrieved_count=len(results),
                            cap_reached=kwargs['max_results'] is not None and len(results) >= kwargs['max_results'],
                            coverage=coverage, records=results)
            papers.extend(results)
    return papers
