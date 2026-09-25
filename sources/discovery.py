"""Dispatch configured discovery sources; validate the plan before requests."""
from __future__ import annotations

from datetime import date
import inspect
import warnings

from utils.discovery_log import utc_now
from utils.protocol import queries_for, validate_protocol

from sources.openalex_source import search_openalex, iter_openalex_pages
from sources.arxiv_source import search_arxiv, iter_arxiv_pages
from sources.ieee_source import search_ieee, iter_ieee_pages
from sources.scopus_source import search_scopus, iter_scopus_pages


def _source_iterators():
    """Return current iterator callables so tests/clients can patch them safely."""
    return {
        "openalex": iter_openalex_pages,
        "arxiv": iter_arxiv_pages,
        "ieee": iter_ieee_pages,
        "scopus": iter_scopus_pages,
    }


def get_queries_from_search_terms(search_terms: dict, source: str = "openalex") -> list[tuple[str, str]]:
    structured = queries_for(search_terms, source)
    if structured is not None:
        return structured
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


def build_search_plan(config: dict, search_terms: dict):
    """Validate and return the exact source/query plan used for retrieval."""
    validate_protocol(search_terms)
    adapters = {"openalex": search_openalex, "arxiv": search_arxiv,
                "ieee": search_ieee, "scopus": search_scopus}
    signatures = _source_iterators()
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
        allowed = set(inspect.signature(signatures[source]).parameters) - {
            "query", "session", "api_key", "insttoken", "resume"
        }
        if set(options) - allowed:
            raise ValueError(f"Unsupported options for {source}: {sorted(set(options) - allowed)}. Set credentials in environment variables.")

        kwargs = {"max_results": config.get("max_results_per_query")}

        if source == "openalex":
            kwargs.update({
                k: config[k]
                for k in (
                    "mailto",
                    "from_publication_date",
                    "to_publication_date",
                    "sleep_seconds",
                )
                if k in config
            })

        elif source == "arxiv":
            kwargs.update({
                k: config[k]
                for k in ("from_publication_date", "to_publication_date")
                if config.get(k)
            })
            # Never inherit the one-second OpenAlex delay for arXiv.
            kwargs["sleep_seconds"] = max(3.0, config.get("sleep_seconds", 3.0))

        elif source == "scopus":
            kwargs["sleep_seconds"] = config.get("sleep_seconds", 1.0)
            for global_key, year_key in (
                ("from_publication_date", "start_year"),
                ("to_publication_date", "end_year"),
            ):
                exact_value = options.get(global_key, config.get(global_key))
                if exact_value:
                    # Scopus filters the provider request by year, then the
                    # adapter enforces this exact YYYY-MM-DD bound locally.
                    kwargs[global_key] = exact_value
                    if year_key not in options:
                        kwargs[year_key] = date.fromisoformat(
                            str(exact_value)
                        ).year

        else:
            kwargs["sleep_seconds"] = config.get("sleep_seconds", 1.0)
            for global_key, year_key in (
                ("from_publication_date", "start_year"),
                ("to_publication_date", "end_year"),
            ):
                if config.get(global_key) and year_key not in options:
                    value = date.fromisoformat(str(config[global_key]))
                    kwargs[year_key] = value.year
                    warnings.warn(
                        f"{source}: {global_key} uses the whole inclusive year "
                        f"{value.year}; day/month precision is unavailable.",
                        stacklevel=2,
                    )

        kwargs.update(options)

        cap = kwargs["max_results"]
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError(f"{source}: max_results must be a positive integer.")

        plan.append((source, kwargs, get_queries_from_search_terms(search_terms, source)))

    return plan


def fetch_papers(config: dict, search_terms: dict, *, audit=None) -> list[dict]:
    plan = build_search_plan(config, search_terms)
    signatures = _source_iterators()

    if audit:
        audit.snapshot(
            "search_plan",
            [
                {
                    "source": source,
                    "options": kwargs,
                    "queries": [
                        {"topic": topic, "query": query}
                        for topic, query in queries
                    ],
                }
                for source, kwargs, queries in plan
            ],
        )

    papers, outcomes = [], []
    query_number = 0

    for source, kwargs, queries in plan:
        if not queries:
            outcomes.append({
                "source": source,
                "status": "skipped",
                "complete": False,
                "reason": "No queries configured for enabled source.",
            })

        for topic, query in queries:
            query_number += 1
            context = {
                "query_id": str(query_number),
                "source": source,
                "search_topic": topic,
                "query": query,
            }
            print(f"\nSearching {source} [{topic}]: {query}")

            if audit:
                audit.event("query_started", **context, options=kwargs)

            results, last = [], None

            try:
                for page_number, page in enumerate(
                    signatures[source](query=query, **kwargs),
                    1,
                ):
                    rows = [
                        {
                            **paper,
                            "search_topic": topic,
                            "retrieved_at": utc_now(),
                            **(
                                {
                                    "run_id": audit.run_id,
                                    "query_id": str(query_number),
                                }
                                if audit
                                else {}
                            ),
                        }
                        for paper in page["papers"]
                    ]
                    details = {
                        k: v
                        for k, v in page.items()
                        if k not in {
                            "papers",
                            "raw_response",
                            "source",
                            "query",
                        }
                    }

                    if audit:
                        audit.event(
                            "query_page",
                            **context,
                            page_number=page_number,
                            **details,
                            records=rows,
                        )
                        if (
                            config.get("save_raw_responses", True)
                            and "raw_response" in page
                        ):
                            audit.snapshot(
                                f"raw_{query_number}_{page_number}",
                                page["raw_response"],
                            )

                    results.extend(rows)
                    last = details

                if last is None:
                    raise RuntimeError(
                        f"{source}: no completion record returned."
                    )

            except BaseException as exc:
                outcomes.append({
                    **context,
                    "status": "failed",
                    "complete": False,
                    "retrieved_count": len(results),
                    "error_type": type(exc).__name__,
                })
                if audit:
                    audit.event("query_failed", **outcomes[-1])
                    audit.snapshot("retrieval_summary", outcomes)
                raise

            outcome = {
                **context,
                **last,
                "retrieved_count": len(results),
                "status": "complete" if last["complete"] else "incomplete",
            }
            outcomes.append(outcome)

            if not last["complete"]:
                warnings.warn(
                    f"{source}: incomplete search ({last['stop_reason']}); "
                    f"{len(results)} records retrieved.",
                    stacklevel=2,
                )

            if audit:
                audit.event("query_succeeded", **outcome)

            papers.extend(results)

    if audit:
        audit.snapshot("retrieval_summary", outcomes)

    return papers
