"""Persistent SQLite corpus for restartable discovery and screening."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

import pandas as pd

from utils.deduplication import decode, deduplicate_papers, identifiers, present
from utils.io import load_existing_table, save_table


HUMAN_FIELDS = ("manual_decision", "manual_reason", "notes")
PROCESS_PREFIXES = ("eligibility_", "screening_", "fulltext_", "pdf_")
DERIVED_FIELDS = {"effective_decision", "decision_origin"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _metadata_only(paper: dict) -> dict:
    return {
        key: value
        for key, value in paper.items()
        if key not in HUMAN_FIELDS
        and key not in DERIVED_FIELDS
        and not key.startswith(PROCESS_PREFIXES)
        and key != "corpus_id"
    }


def _processing_bundle(paper: dict) -> dict:
    return {
        key: value
        for key, value in paper.items()
        if key.startswith(PROCESS_PREFIXES)
    }


class CorpusStore:
    """Transactional working corpus. Excel remains an import/export interface."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @contextmanager
    def transaction(self):
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                corpus_id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS aliases (
                alias_key TEXT PRIMARY KEY,
                corpus_id TEXT NOT NULL REFERENCES papers(corpus_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS provenance (
                corpus_id TEXT NOT NULL REFERENCES papers(corpus_id) ON DELETE CASCADE,
                provenance_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (corpus_id, provenance_key)
            );
            CREATE TABLE IF NOT EXISTS human_review (
                corpus_id TEXT PRIMARY KEY REFERENCES papers(corpus_id) ON DELETE CASCADE,
                manual_decision TEXT NOT NULL DEFAULT '',
                manual_reason TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processing (
                corpus_id TEXT PRIMARY KEY REFERENCES papers(corpus_id) ON DELETE CASCADE,
                fulltext_status TEXT NOT NULL DEFAULT 'not_requested',
                pdf_sha256 TEXT NOT NULL DEFAULT '',
                screening_status TEXT NOT NULL DEFAULT 'not_screened',
                screening_signature TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL DEFAULT '{}',
                error_type TEXT NOT NULL DEFAULT '',
                error_stage TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                artifact_folder TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS discovery_runs (
                run_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS discovery_tasks (
                task_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES discovery_runs(run_id) ON DELETE CASCADE,
                source TEXT NOT NULL,
                search_topic TEXT NOT NULL,
                query TEXT NOT NULL,
                partition_key TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL DEFAULT '{}',
                pages_committed INTEGER NOT NULL DEFAULT 0,
                records_committed INTEGER NOT NULL DEFAULT 0,
                error_type TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_discovery_tasks_run
                ON discovery_tasks(run_id, status);
            CREATE TABLE IF NOT EXISTS discovery_pages (
                task_id TEXT NOT NULL REFERENCES discovery_tasks(task_id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                page_json TEXT NOT NULL,
                raw_response TEXT NOT NULL DEFAULT '',
                committed_at TEXT NOT NULL,
                PRIMARY KEY (task_id, page_number)
            );
            CREATE TABLE IF NOT EXISTS arxiv_harvests (
                harvest_key TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS arxiv_records (
                harvest_key TEXT NOT NULL REFERENCES arxiv_harvests(harvest_key) ON DELETE CASCADE,
                arxiv_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY (harvest_key, arxiv_id)
            );
            """
        )
        self.conn.commit()

    def _matching_ids(self, paper: dict) -> list[str]:
        keys = sorted(identifiers(paper))
        if not keys:
            return []
        marks = ",".join("?" for _ in keys)
        rows = self.conn.execute(
            f"SELECT DISTINCT corpus_id FROM aliases WHERE alias_key IN ({marks})",
            keys,
        ).fetchall()
        return [row["corpus_id"] for row in rows]

    def _load_metadata(self, corpus_id: str) -> dict:
        row = self.conn.execute(
            "SELECT metadata_json FROM papers WHERE corpus_id=?",
            (corpus_id,),
        ).fetchone()
        return json.loads(row["metadata_json"]) if row else {}

    def _merge_ids(self, target: str, others: list[str]):
        for source in others:
            if source == target:
                continue
            target_meta = self._load_metadata(target)
            source_meta = self._load_metadata(source)
            if source_meta:
                merged = deduplicate_papers([target_meta, source_meta])[0]
                merged = _metadata_only(merged)
                merged["corpus_id"] = target
                self.conn.execute(
                    "UPDATE papers SET metadata_json=?, updated_at=? WHERE corpus_id=?",
                    (_json(merged), utc_now(), target),
                )

            target_human = self.conn.execute(
                "SELECT * FROM human_review WHERE corpus_id=?", (target,)
            ).fetchone()
            source_human = self.conn.execute(
                "SELECT * FROM human_review WHERE corpus_id=?", (source,)
            ).fetchone()
            if source_human:
                if target_human:
                    t = dict(target_human)
                    s = dict(source_human)
                    if (
                        t["manual_decision"]
                        and s["manual_decision"]
                        and t["manual_decision"] != s["manual_decision"]
                    ):
                        raise ValueError(
                            "Cannot merge corpus records with conflicting human decisions."
                        )
                    values = [
                        t[field] or s[field]
                        for field in HUMAN_FIELDS
                    ]
                    self.conn.execute(
                        """UPDATE human_review
                           SET manual_decision=?, manual_reason=?, notes=?, updated_at=?
                           WHERE corpus_id=?""",
                        (*values, utc_now(), target),
                    )
                else:
                    self.conn.execute(
                        """INSERT INTO human_review
                           (corpus_id, manual_decision, manual_reason, notes, updated_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            target,
                            source_human["manual_decision"],
                            source_human["manual_reason"],
                            source_human["notes"],
                            source_human["updated_at"],
                        ),
                    )

            self.conn.execute(
                "UPDATE OR IGNORE aliases SET corpus_id=? WHERE corpus_id=?",
                (target, source),
            )
            rows = self.conn.execute(
                "SELECT provenance_key, payload_json FROM provenance WHERE corpus_id=?",
                (source,),
            ).fetchall()
            for row in rows:
                self.conn.execute(
                    """INSERT OR IGNORE INTO provenance
                       (corpus_id, provenance_key, payload_json)
                       VALUES (?, ?, ?)""",
                    (target, row["provenance_key"], row["payload_json"]),
                )

            target_processing = self.conn.execute(
                "SELECT * FROM processing WHERE corpus_id=?", (target,)
            ).fetchone()
            source_processing = self.conn.execute(
                "SELECT * FROM processing WHERE corpus_id=?", (source,)
            ).fetchone()
            if source_processing and not target_processing:
                columns = (
                    "fulltext_status", "pdf_sha256", "screening_status",
                    "screening_signature", "result_json", "error_type",
                    "error_stage", "error_message", "artifact_folder", "updated_at"
                )
                self.conn.execute(
                    """INSERT INTO processing
                       (corpus_id, fulltext_status, pdf_sha256, screening_status,
                        screening_signature, result_json, error_type, error_stage,
                        error_message, artifact_folder, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (target, *(source_processing[col] for col in columns)),
                )

            self.conn.execute("DELETE FROM papers WHERE corpus_id=?", (source,))

    def _insert_provenance(self, corpus_id: str, paper: dict):
        entries = decode(paper.get("provenance"), [])
        if not entries:
            payload = {
                key: paper.get(key)
                for key in (
                    "source", "paper_id", "query", "search_topic",
                    "retrieved_at", "run_id", "query_id"
                )
                if present(paper.get(key))
            }
            entries = [payload] if payload else []

        for payload in entries:
            key = hashlib.sha256(_json(payload).encode()).hexdigest()
            self.conn.execute(
                """INSERT OR IGNORE INTO provenance
                   (corpus_id, provenance_key, payload_json)
                   VALUES (?, ?, ?)""",
                (corpus_id, key, _json(payload)),
            )

    def _upsert_paper_locked(self, paper: dict) -> str:
        incoming = dict(paper)
        matched = self._matching_ids(incoming)
        if matched:
            corpus_id = matched[0]
            if len(matched) > 1:
                self._merge_ids(corpus_id, matched[1:])
            existing = self._load_metadata(corpus_id)
            merged = deduplicate_papers([existing, incoming])[0] if existing else incoming
        else:
            corpus_id = "p_" + uuid.uuid4().hex
            merged = incoming

        metadata = _metadata_only(merged)
        metadata["corpus_id"] = corpus_id
        timestamp = utc_now()
        self.conn.execute(
            """INSERT INTO papers(corpus_id, metadata_json, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(corpus_id) DO UPDATE SET
                 metadata_json=excluded.metadata_json,
                 updated_at=excluded.updated_at""",
            (corpus_id, _json(metadata), timestamp, timestamp),
        )

        for alias in sorted(identifiers(merged) | identifiers(incoming)):
            existing_alias = self.conn.execute(
                "SELECT corpus_id FROM aliases WHERE alias_key=?", (alias,)
            ).fetchone()
            if existing_alias and existing_alias["corpus_id"] != corpus_id:
                self._merge_ids(corpus_id, [existing_alias["corpus_id"]])
            self.conn.execute(
                "INSERT OR REPLACE INTO aliases(alias_key, corpus_id) VALUES (?, ?)",
                (alias, corpus_id),
            )

        self._insert_provenance(corpus_id, incoming)
        if any(present(incoming.get(field)) for field in HUMAN_FIELDS):
            self.set_human_review(
                corpus_id,
                incoming.get("manual_decision", ""),
                incoming.get("manual_reason", ""),
                incoming.get("notes", ""),
                commit=False,
            )
        if _processing_bundle(incoming):
            self.update_processing(corpus_id, incoming, commit=False)

        return corpus_id

    def upsert_paper(self, paper: dict, *, commit: bool = True) -> str:
        """Merge a record and all known aliases, returning its stable corpus ID."""
        if commit:
            with self.transaction():
                return self._upsert_paper_locked(paper)
        return self._upsert_paper_locked(paper)

    def set_human_review(
        self,
        corpus_id: str,
        manual_decision: str = "",
        manual_reason: str = "",
        notes: str = "",
        *,
        commit: bool = True,
    ):
        self.conn.execute(
            """INSERT INTO human_review
               (corpus_id, manual_decision, manual_reason, notes, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(corpus_id) DO UPDATE SET
                 manual_decision=excluded.manual_decision,
                 manual_reason=excluded.manual_reason,
                 notes=excluded.notes,
                 updated_at=excluded.updated_at""",
            (corpus_id, manual_decision or "", manual_reason or "", notes or "", utc_now()),
        )
        if commit:
            self.conn.commit()

    def update_processing(
        self,
        corpus_id: str,
        paper: dict,
        screening_signature: str | None = None,
        *,
        commit: bool = True,
    ):
        bundle = _processing_bundle(paper)
        status = paper.get("eligibility_status", "not_screened") or "not_screened"
        self.conn.execute(
            """INSERT INTO processing
               (corpus_id, fulltext_status, pdf_sha256, screening_status,
                screening_signature, result_json, error_type, error_stage,
                error_message, artifact_folder, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(corpus_id) DO UPDATE SET
                 fulltext_status=excluded.fulltext_status,
                 pdf_sha256=excluded.pdf_sha256,
                 screening_status=excluded.screening_status,
                 screening_signature=excluded.screening_signature,
                 result_json=excluded.result_json,
                 error_type=excluded.error_type,
                 error_stage=excluded.error_stage,
                 error_message=excluded.error_message,
                 artifact_folder=excluded.artifact_folder,
                 updated_at=excluded.updated_at""",
            (
                corpus_id,
                paper.get("fulltext_status", "not_requested") or "not_requested",
                paper.get("pdf_sha256", "") or "",
                status,
                screening_signature or paper.get("screening_signature", "") or "",
                _json(bundle),
                paper.get("eligibility_error_type", "") or "",
                paper.get("eligibility_error_stage", "") or "",
                paper.get("eligibility_error_message", "") or "",
                paper.get("fulltext_folder", "") or "",
                utc_now(),
            ),
        )
        if commit:
            self.conn.commit()

    def get_paper(self, corpus_id: str) -> dict | None:
        row = self.conn.execute(
            """SELECT p.metadata_json,
                      h.manual_decision, h.manual_reason, h.notes,
                      x.result_json, x.screening_signature
               FROM papers p
               LEFT JOIN human_review h USING(corpus_id)
               LEFT JOIN processing x USING(corpus_id)
               WHERE p.corpus_id=?""",
            (corpus_id,),
        ).fetchone()
        if not row:
            return None
        paper = json.loads(row["metadata_json"])
        paper["corpus_id"] = corpus_id
        result = json.loads(row["result_json"] or "{}")
        paper.update(result)
        for field in HUMAN_FIELDS:
            paper[field] = row[field] or ""
        paper["screening_signature"] = row["screening_signature"] or ""
        return paper

    def all_papers(self) -> list[dict]:
        ids = [
            row["corpus_id"]
            for row in self.conn.execute(
                "SELECT corpus_id FROM papers ORDER BY created_at, corpus_id"
            )
        ]
        return [self.get_paper(corpus_id) for corpus_id in ids]

    def begin_discovery_run(self, config_payload: dict, *, resume: bool = True) -> str:
        config_hash = hashlib.sha256(_json(config_payload).encode()).hexdigest()
        if resume:
            row = self.conn.execute(
                """SELECT run_id FROM discovery_runs
                   WHERE config_hash=? AND status IN ('running','incomplete')
                   ORDER BY started_at DESC LIMIT 1""",
                (config_hash,),
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE discovery_runs SET status='running', updated_at=? WHERE run_id=?",
                    (utc_now(), row["run_id"]),
                )
                self.conn.commit()
                return row["run_id"]

        run_id = "d_" + uuid.uuid4().hex
        now = utc_now()
        self.conn.execute(
            """INSERT INTO discovery_runs
               (run_id, config_hash, config_json, status, started_at, updated_at)
               VALUES (?, ?, ?, 'running', ?, ?)""",
            (run_id, config_hash, _json(config_payload), now, now),
        )
        self.conn.commit()
        return run_id

    def ensure_discovery_task(
        self,
        run_id: str,
        source: str,
        search_topic: str,
        query: str,
        partition_key: str,
        config_payload: dict,
    ) -> dict:
        config_hash = hashlib.sha256(_json(config_payload).encode()).hexdigest()
        task_seed = _json({
            "run_id": run_id,
            "source": source,
            "search_topic": search_topic,
            "query": query,
            "partition_key": partition_key,
            "config_hash": config_hash,
        })
        task_id = "t_" + hashlib.sha256(task_seed.encode()).hexdigest()[:24]
        now = utc_now()
        self.conn.execute(
            """INSERT OR IGNORE INTO discovery_tasks
               (task_id, run_id, source, search_topic, query, partition_key,
                config_hash, status, checkpoint_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', '{}', ?)""",
            (
                task_id, run_id, source, search_topic, query,
                partition_key, config_hash, now,
            ),
        )
        self.conn.commit()
        return self.get_discovery_task(task_id)

    def get_discovery_task(self, task_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM discovery_tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["checkpoint"] = json.loads(data.pop("checkpoint_json") or "{}")
        return data

    def reset_discovery_task(self, task_id: str):
        self.conn.execute(
            """UPDATE discovery_tasks
               SET status='pending', checkpoint_json='{}',
                   pages_committed=0, records_committed=0,
                   error_type='', error_message='', updated_at=?
               WHERE task_id=?""",
            (utc_now(), task_id),
        )
        self.conn.commit()

    def commit_discovery_page(
        self,
        task_id: str,
        papers: list[dict],
        checkpoint: dict,
        *,
        task_status: str,
        page_payload: dict | None = None,
        raw_response=None,
    ):
        """Atomically merge one page, archive it, and advance its checkpoint."""
        if task_status not in {"running", "complete", "incomplete"}:
            raise ValueError("Invalid discovery task status.")
        with self.transaction():
            row = self.conn.execute(
                "SELECT pages_committed FROM discovery_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Unknown discovery task: {task_id}")
            page_number = row["pages_committed"] + 1

            for paper in papers:
                self.upsert_paper(paper, commit=False)

            self.conn.execute(
                """INSERT INTO discovery_pages
                   (task_id, page_number, page_json, raw_response, committed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    task_id,
                    page_number,
                    _json(page_payload or {}),
                    (
                        raw_response
                        if isinstance(raw_response, str)
                        else _json(raw_response) if raw_response is not None else ""
                    ),
                    utc_now(),
                ),
            )
            self.conn.execute(
                """UPDATE discovery_tasks
                   SET status=?, checkpoint_json=?,
                       pages_committed=pages_committed+1,
                       records_committed=records_committed+?,
                       error_type='', error_message='', updated_at=?
                   WHERE task_id=?""",
                (
                    task_status,
                    _json(checkpoint or {}),
                    len(papers),
                    utc_now(),
                    task_id,
                ),
            )

    def mark_discovery_task_error(self, task_id: str, exc: Exception):
        self.conn.execute(
            """UPDATE discovery_tasks
               SET status='error', error_type=?, error_message=?, updated_at=?
               WHERE task_id=?""",
            (type(exc).__name__, str(exc), utc_now(), task_id),
        )
        self.conn.commit()

    def finish_discovery_run(self, run_id: str):
        rows = self.conn.execute(
            "SELECT status FROM discovery_tasks WHERE run_id=?",
            (run_id,),
        ).fetchall()
        statuses = [row["status"] for row in rows]
        status = (
            "complete"
            if statuses and all(value == "complete" for value in statuses)
            else "incomplete"
        )
        completed = utc_now() if status == "complete" else None
        self.conn.execute(
            """UPDATE discovery_runs
               SET status=?, updated_at=?, completed_at=?
               WHERE run_id=?""",
            (status, utc_now(), completed, run_id),
        )
        self.conn.commit()
        return status

    def discovery_status(self) -> dict:
        task_counts = {
            row["status"]: row["count"]
            for row in self.conn.execute(
                "SELECT status, COUNT(*) AS count FROM discovery_tasks GROUP BY status"
            )
        }
        run_counts = {
            row["status"]: row["count"]
            for row in self.conn.execute(
                "SELECT status, COUNT(*) AS count FROM discovery_runs GROUP BY status"
            )
        }
        return {"runs": run_counts, "tasks": task_counts}

    def import_workbook(self, path: str | Path) -> int:
        frame = load_existing_table(str(path))
        for paper in frame.to_dict("records"):
            self.upsert_paper(paper)
        return len(frame)

    def export_workbook(self, path: str | Path) -> int:
        papers = self.all_papers()
        save_table(pd.DataFrame(papers), str(path))
        return len(papers)

    def status(self) -> dict:
        papers = self.all_papers()
        return {
            "papers": len(papers),
            "screening_status": dict(
                Counter(p.get("eligibility_status", "not_screened") for p in papers)
            ),
            "fulltext_status": dict(
                Counter(p.get("fulltext_status", "not_requested") for p in papers)
            ),
            "human_decisions": dict(
                Counter(
                    p.get("manual_decision")
                    for p in papers
                    if p.get("manual_decision")
                )
            ),
        }
