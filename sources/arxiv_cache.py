"""Shared persistent arXiv OAI harvest with local query matching."""
from __future__ import annotations

from datetime import date
import hashlib
import json

from sources.arxiv_source import _matches_query, iter_arxiv_harvest_pages
from utils.discovery_log import utc_now


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _harvest_config(kwargs):
    return {
        "from_publication_date": kwargs.get("from_publication_date"),
        "to_publication_date": kwargs.get("to_publication_date"),
        "oai_from_date": kwargs.get("oai_from_date"),
        "oai_until_date": kwargs.get("oai_until_date"),
        "max_pages": kwargs.get("max_pages"),
        "sleep_seconds": kwargs.get("sleep_seconds", 3.0),
        "timeout": kwargs.get("timeout", 60),
        "max_retries": kwargs.get("max_retries", 5),
    }


def _within_publication_window(paper, lower, upper):
    published = paper.get("published_date")
    if not published:
        return False
    value = date.fromisoformat(str(published)[:10])
    if lower and value < lower:
        return False
    if upper and value > upper:
        return False
    return True


def _run_harvest(store, config, *, resume):
    harvest = store.ensure_arxiv_harvest(config)
    if harvest["status"] in {"complete", "incomplete"}:
        return harvest

    resume_state = harvest["checkpoint"] if resume else {}
    restarted = False

    while True:
        try:
            yielded = False
            for page in iter_arxiv_harvest_pages(
                resume=resume_state,
                **config,
            ):
                yielded = True
                status = (
                    "running"
                    if page["stop_reason"] == "more_pages"
                    else "complete"
                    if page["complete"]
                    else "incomplete"
                )
                store.commit_arxiv_harvest_page(
                    harvest["harvest_key"],
                    page["records"],
                    page["checkpoint"],
                    status=status,
                    page_payload={
                        key: value
                        for key, value in page.items()
                        if key not in {"records", "raw_response"}
                    },
                    raw_response=page.get("raw_response"),
                )
                resume_state = page["checkpoint"]

            if not yielded:
                current = store.get_arxiv_harvest(harvest["harvest_key"])
                if current["status"] not in {"complete", "incomplete"}:
                    raise RuntimeError("arXiv harvest returned no completion page.")
            break
        except Exception:
            if resume_state and not restarted:
                # OAI resumption tokens can expire. Restart only this cached
                # harvest window; query/corpus state is independently deduped.
                store.reset_arxiv_harvest(harvest["harvest_key"])
                resume_state = {}
                restarted = True
                continue
            raise

    return store.get_arxiv_harvest(harvest["harvest_key"])


def _task_partition(harvest_key, kwargs):
    return _json({
        "harvest_key": harvest_key,
        "from_publication_date": kwargs.get("from_publication_date"),
        "to_publication_date": kwargs.get("to_publication_date"),
        "max_results": kwargs.get("max_results"),
    })


def run_shared_arxiv_discovery(
    store,
    *,
    run_id,
    queries,
    kwargs,
    resume=True,
):
    """Harvest once, then evaluate every configured query locally."""
    harvest_config = _harvest_config(kwargs)
    try:
        harvest = _run_harvest(store, harvest_config, resume=resume)
    except Exception as exc:
        results = []
        # Represent every query as independently failed while keeping other
        # providers available to the discovery run.
        for topic, query in queries:
            task = store.ensure_discovery_task(
                run_id,
                "arxiv",
                topic,
                query,
                "harvest:" + hashlib.sha256(_json(harvest_config).encode()).hexdigest()[:24],
                {
                    "source": "arxiv",
                    "query": query,
                    "harvest": harvest_config,
                    "max_results": kwargs.get("max_results"),
                },
            )
            store.mark_discovery_task_error(task["task_id"], exc)
            results.append({
                "task_id": task["task_id"],
                "source": "arxiv",
                "search_topic": topic,
                "query": query,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
        return results

    records = store.arxiv_harvest_records(harvest["harvest_key"])
    lower = (
        date.fromisoformat(str(kwargs["from_publication_date"]))
        if kwargs.get("from_publication_date")
        else None
    )
    upper = (
        date.fromisoformat(str(kwargs["to_publication_date"]))
        if kwargs.get("to_publication_date")
        else None
    )
    max_results = kwargs.get("max_results")
    results = []

    for topic, query in queries:
        task = store.ensure_discovery_task(
            run_id,
            "arxiv",
            topic,
            query,
            _task_partition(harvest["harvest_key"], kwargs),
            {
                "source": "arxiv",
                "query": query,
                "harvest_key": harvest["harvest_key"],
                "from_publication_date": kwargs.get("from_publication_date"),
                "to_publication_date": kwargs.get("to_publication_date"),
                "max_results": max_results,
            },
        )
        if task["status"] in {"complete", "incomplete"}:
            results.append({
                "task_id": task["task_id"],
                "source": "arxiv",
                "search_topic": topic,
                "query": query,
                "status": task["status"],
                "skipped": True,
                "harvest_cache_hit": True,
            })
            continue

        matches = [
            {
                **paper,
                "query": query,
                "search_topic": topic,
                "retrieved_at": utc_now(),
                "run_id": run_id,
                "query_id": task["task_id"],
            }
            for paper in records
            if _within_publication_window(paper, lower, upper)
            and _matches_query(
                paper.get("title", ""),
                paper.get("abstract", ""),
                query,
            )
        ]

        cap_reached = max_results is not None and len(matches) > max_results
        selected = matches[:max_results] if max_results is not None else matches
        complete = harvest["status"] == "complete" and not cap_reached
        task_status = "complete" if complete else "incomplete"
        stop_reason = (
            "exhausted"
            if complete
            else "max_results"
            if cap_reached
            else "harvest_incomplete"
        )

        store.commit_discovery_page(
            task["task_id"],
            selected,
            {
                "harvest_key": harvest["harvest_key"],
                "matched": len(selected),
            },
            task_status=task_status,
            page_payload={
                "source": "arxiv",
                "query": query,
                "harvest_key": harvest["harvest_key"],
                "harvest_status": harvest["status"],
                "coverage_scope": (
                    "historical_publication_window"
                    if kwargs.get("from_publication_date")
                    else "oai_update_window"
                ),
                "publication_from": kwargs.get("from_publication_date"),
                "publication_until": kwargs.get("to_publication_date"),
                "oai_from": harvest_config.get("oai_from_date")
                or harvest_config.get("from_publication_date"),
                "oai_until": harvest_config.get("oai_until_date"),
                "total_cached_records": len(records),
                "matched_records": len(selected),
                "complete": complete,
                "stop_reason": stop_reason,
            },
        )
        results.append({
            "task_id": task["task_id"],
            "source": "arxiv",
            "search_topic": topic,
            "query": query,
            "status": task_status,
            "matched_records": len(selected),
            "harvest_key": harvest["harvest_key"],
            "harvest_cache_hit": True,
        })

    return results
