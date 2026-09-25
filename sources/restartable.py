"""Restartable discovery orchestration over persistent SQLite state."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
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
    today = date.today().isoformat()
    # Missing effective dates must not drift across a resumed run.
    if not frozen.get("to_publication_date"):
        frozen["to_publication_date"] = today
    if "arxiv" in frozen.get("sources", ["openalex"]):
        source_options = deepcopy(frozen.get("source_options", {}))
        arxiv_options = deepcopy(source_options.get("arxiv", {}))
        explicit_oai_window = bool(
            arxiv_options.get("oai_from_date")
            or arxiv_options.get("oai_until_date")
        )
        frozen["_arxiv_coverage_mode"] = (
            "oai_update_window"
            if explicit_oai_window
            else "historical_publication_window"
        )
        arxiv_options.setdefault("oai_until_date", today)
        source_options["arxiv"] = arxiv_options
        frozen["source_options"] = source_options
    return frozen


def _retrieval_protocol(protocol):
    if protocol.get("schema_version") == 2:
        return {
            "schema_version": 2,
            "searches": deepcopy(protocol.get("searches", [])),
        }
    return {
        key: deepcopy(value)
        for key, value in protocol.items()
        if key != "eligibility"
    }


def _discovery_resume_identity(config, protocol):
    keys = (
        "sources",
        "source_options",
        "max_results_per_query",
        "from_publication_date",
        "to_publication_date",
        "mailto",
        "sleep_seconds",
    )
    return {
        "discovery_config": {
            key: config.get(key)
            for key in keys
            if key in config
        },
        "search_protocol": _retrieval_protocol(protocol),
    }


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


def _cursor_error(exc):
    message = str(exc).casefold()
    return "cursor" in message and any(
        token in message
        for token in (
            "invalid",
            "expired",
            "ended early",
            "repeated",
            "rejected",
        )
    )


def _run_standard_task(
    store,
    *,
    run_id,
    source,
    topic,
    query,
    kwargs,
    resume,
    save_raw_responses=True,
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
                    raw_response=(
                        page.get("raw_response")
                        if save_raw_responses
                        else None
                    ),
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
            cursor_resume = (
                source == "openalex"
                and resume_state
                and resume_state.get("cursor")
            ) or (
                source == "scopus"
                and resume_state
                and resume_state.get("mode") == "cursor"
                and resume_state.get("cursor")
            )
            if (
                cursor_resume
                and _cursor_error(exc)
                and not restarted_expired_cursor
            ):
                # Provider cursors may expire after a long interruption.
                # Restart only this partition; globally merged papers remain.
                store.reset_discovery_task(task["task_id"])
                resume_state = {}
                restarted_expired_cursor = True
                continue

            if source == "scopus" and "5,000" in str(exc):
                return {
                    "task_id": task["task_id"],
                    "source": source,
                    "search_topic": topic,
                    "query": query,
                    "status": "partition_required",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

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


def _split_scopus_partition(kwargs):
    lower_text = kwargs.get("from_publication_date")
    upper_text = kwargs.get("to_publication_date")
    if not lower_text and kwargs.get("start_year") is not None:
        lower_text = f"{int(kwargs['start_year']):04d}-01-01"
    if not upper_text and kwargs.get("end_year") is not None:
        upper_text = f"{int(kwargs['end_year']):04d}-12-31"
    if not lower_text or not upper_text:
        return None

    lower = date.fromisoformat(str(lower_text))
    upper = date.fromisoformat(str(upper_text))
    if lower >= upper:
        return None

    midpoint = lower + timedelta(days=(upper - lower).days // 2)
    right_start = midpoint + timedelta(days=1)

    left = deepcopy(kwargs)
    right = deepcopy(kwargs)
    left["from_publication_date"] = lower.isoformat()
    left["to_publication_date"] = midpoint.isoformat()
    left["start_year"] = lower.year
    left["end_year"] = midpoint.year

    right["from_publication_date"] = right_start.isoformat()
    right["to_publication_date"] = upper.isoformat()
    right["start_year"] = right_start.year
    right["end_year"] = upper.year
    return left, right


def _run_scopus_partitioned(
    store,
    *,
    run_id,
    topic,
    query,
    kwargs,
    resume,
    save_raw_responses=True,
):
    result = _run_standard_task(
        store,
        run_id=run_id,
        source="scopus",
        topic=topic,
        query=query,
        kwargs=kwargs,
        resume=resume,
        save_raw_responses=save_raw_responses,
    )
    if result["status"] != "partition_required":
        return [result]

    children = _split_scopus_partition(kwargs)
    if children is None:
        exc = RuntimeError(
            "Scopus offset pagination still exceeds 5,000 source records "
            "at the minimum one-day partition; coverage remains incomplete."
        )
        store.mark_discovery_task_error(result["task_id"], exc)
        result.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        return [result]

    left, right = children
    left_results = _run_scopus_partitioned(
        store,
        run_id=run_id,
        topic=topic,
        query=query,
        kwargs=left,
        resume=resume,
        save_raw_responses=save_raw_responses,
    )
    right_results = _run_scopus_partitioned(
        store,
        run_id=run_id,
        topic=topic,
        query=query,
        kwargs=right,
        resume=resume,
        save_raw_responses=save_raw_responses,
    )
    child_ids = [
        left_results[0]["task_id"],
        right_results[0]["task_id"],
    ]
    store.mark_discovery_task_partitioned(result["task_id"], child_ids)
    result.update({
        "status": "partitioned",
        "children": child_ids,
        "partition_bounds": {
            "from": kwargs.get("from_publication_date"),
            "to": kwargs.get("to_publication_date"),
        },
    })
    return [result, *left_results, *right_results]


def discover_restartable(store, config, protocol, *, resume=True):
    """Run every source independently; persist successes even if others fail."""
    retrieval_protocol = _retrieval_protocol(protocol)
    resume_identity = _discovery_resume_identity(config, retrieval_protocol)
    frozen = _freeze_config(config)
    plan = build_search_plan(frozen, retrieval_protocol)
    payload = {
        "frozen_on": date.today().isoformat(),
        "frozen_config": frozen,
        "plan": _plan_payload(plan),
        "search_protocol": retrieval_protocol,
    }
    run_id = store.begin_discovery_run(
        payload,
        resume=resume,
        match_payload=resume_identity,
    )
    saved_run = store.get_discovery_run(run_id)
    saved_payload = saved_run["config"]
    if saved_payload != payload:
        frozen = saved_payload["frozen_config"]
        protocol = saved_payload["search_protocol"]
        plan = build_search_plan(frozen, protocol)
        payload = saved_payload
    results = []

    for source, kwargs, queries in plan:
        if not queries:
            task = store.ensure_discovery_task(
                run_id,
                source,
                "",
                "",
                "no_queries",
                {"source": source, "reason": "no_queries_configured"},
            )
            store.mark_discovery_task_incomplete(
                task["task_id"],
                "No queries configured for enabled source.",
            )
            results.append({
                "task_id": task["task_id"],
                "source": source,
                "search_topic": "",
                "query": "",
                "status": "incomplete",
                "reason": "No queries configured for enabled source.",
            })
            continue

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
                    coverage_mode=frozen.get(
                        "_arxiv_coverage_mode",
                        "historical_publication_window",
                    ),
                    save_raw_responses=frozen.get(
                        "save_raw_responses",
                        True,
                    ),
                )
            )
            continue

        for topic, query in queries:
            if source == "scopus":
                results.extend(
                    _run_scopus_partitioned(
                        store,
                        run_id=run_id,
                        topic=topic,
                        query=query,
                        kwargs=kwargs,
                        resume=resume,
                        save_raw_responses=frozen.get(
                            "save_raw_responses",
                            True,
                        ),
                    )
                )
            else:
                results.append(
                    _run_standard_task(
                        store,
                        run_id=run_id,
                        source=source,
                        topic=topic,
                        query=query,
                        kwargs=kwargs,
                        resume=resume,
                        save_raw_responses=frozen.get(
                            "save_raw_responses",
                            True,
                        ),
                    )
                )

    status = store.finish_discovery_run(run_id)
    return {
        "run_id": run_id,
        "status": status,
        "tasks": results,
        "plan_hash": hashlib.sha256(_json(payload).encode()).hexdigest(),
    }
