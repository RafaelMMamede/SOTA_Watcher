"""Restartable discovery orchestration over persistent SQLite state."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json

from sources.discovery import build_search_plan
from sources.openalex_source import iter_openalex_pages
from sources.ieee_source import iter_ieee_pages
from sources.scopus_source import iter_scopus_pages
from utils.discovery_log import utc_now


ITERATORS = {
    "openalex": iter_openalex_pages,
    "ieee": iter_ieee_pages,
    "scopus": iter_scopus_pages,
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _freeze_config(config):
    frozen = deepcopy(config)
    # A missing upper publication date otherwise means "whatever today is"
    # when a source executes. Freeze it once per discovery run.
    if not frozen.get("to_publication_date"):
        frozen["to_publication_date"] = date.today().isoformat()
    return frozen


def _plan_payload(plan):
    return [
        {
            "source": source,
            "options": kwargs,
            "queries": [
                {"search_topic": topic, "query": query}
                for topic, query in queries
            ],
        }
        for source, kwargs, queries in plan
    ]


def _partition_key(kwargs):
    return _json({
        "from_publication_date": kwargs.get("from_publication_date"),
        "to_publication_date": kwargs.get("to_publication_date"),
        "start_year": kwargs.get("start_year"),
        "end_year": kwargs.get("end_year"),
    })


def _annotate_records(page, source, topic, query, run_id, task_id):
    return [
        {
            **paper,
            "source": paper.get("source") or source,
            "search_topic": topic,
            "query": paper.get("query") or query,
            "retrieved_at": utc_now(),
            "run_id": run_id,
            "query_id": task_id,
        }
        for paper in page.get("papers", [])
    ]


def _page_metadata(page):
    return {
        key: value
        for key, value in page.items()
        if key not in {"papers", "raw_response"}
    }


def _run_standard_task(
    store,
    *,
    run_id,
    source,
    topic,
    query,
    kwargs,
    resume,
):
    partition_key = _partition_key(kwargs)
    task = store.ensure_discovery_task(
        run_id,
        source,
        topic,
        query,
        partition_key,
        {"source": source, "query": query, "options": kwargs},
    )

    if task["status"] in {"complete", "incomplete"}:
        return {
            "task_id": task["task_id"],
            "source": source,
            "search_topic": topic,
            "query": query,
            "status": task["status"],
            "skipped": True,
        }

    resume_state = task["checkpoint"] if resume else {}
    iterator = ITERATORS[source]
    restarted_expired_cursor = False

    while True:
        try:
            yielded = False
            for page in iterator(
                query=query,
                resume=resume_state,
                **kwargs,
            ):
                yielded = True
                records = _annotate_records(
                    page,
                    source,
                    topic,
                    query,
                    run_id,
                    task["task_id"],
                )
                stop = page.get("stop_reason")
                task_status = (
                    "running"
                    if stop == "more_pages"
                    else "complete"
                    if page.get("complete") is True
                    else "incomplete"
                )
                store.commit_discovery_page(
                    task["task_id"],
                    records,
                    page.get("checkpoint", {}),
                    task_status=task_status,
                    page_payload=_page_metadata(page),
                    raw_response=page.get("raw_response"),
                )
                resume_state = page.get("checkpoint", {})

            if not yielded:
                current = store.get_discovery_task(task["task_id"])
                if current and current["status"] not in {"complete", "incomplete"}:
                    raise RuntimeError(
                        f"{source}: no completion page returned for discovery task."
                    )
            break

        except Exception as exc:
            if (
                source == "scopus"
                and resume_state
                and resume_state.get("mode") == "cursor"
                and not restarted_expired_cursor
            ):
                # A provider may reject an old cursor after a long interruption.
                # Restart only this partition; globally merged papers remain.
                store.reset_discovery_task(task["task_id"])
                resume_state = {}
                restarted_expired_cursor = True
                continue

            store.mark_discovery_task_error(task["task_id"], exc)
            return {
                "task_id": task["task_id"],
                "source": source,
                "search_topic": topic,
                "query": query,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    current = store.get_discovery_task(task["task_id"])
    return {
        "task_id": task["task_id"],
        "source": source,
        "search_topic": topic,
        "query": query,
        "status": current["status"],
        "pages_committed": current["pages_committed"],
        "records_committed": current["records_committed"],
        "restarted_expired_cursor": restarted_expired_cursor,
    }


def discover_restartable(store, config, protocol, *, resume=True):
    """Run every source independently; persist successes even if others fail."""
    frozen = _freeze_config(config)
    plan = build_search_plan(frozen, protocol)
    payload = {
        "frozen_on": date.today().isoformat(),
        "plan": _plan_payload(plan),
        "search_protocol": protocol,
    }
    run_id = store.begin_discovery_run(payload, resume=resume)
    results = []

    for source, kwargs, queries in plan:
        if source == "arxiv":
            # Shared arXiv harvesting/matching is implemented separately so the
            # metadata window is downloaded once for all configured queries.
            from sources.arxiv_cache import run_shared_arxiv_discovery

            results.extend(
                run_shared_arxiv_discovery(
                    store,
                    run_id=run_id,
                    queries=queries,
                    kwargs=kwargs,
                    resume=resume,
                )
            )
            continue

        for topic, query in queries:
            results.append(
                _run_standard_task(
                    store,
                    run_id=run_id,
                    source=source,
                    topic=topic,
                    query=query,
                    kwargs=kwargs,
                    resume=resume,
                )
            )

    status = store.finish_discovery_run(run_id)
    return {
        "run_id": run_id,
        "status": status,
        "tasks": results,
        "plan_hash": hashlib.sha256(_json(payload).encode()).hexdigest(),
    }
