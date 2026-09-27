import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from screening.candidate_selection import (
    build_candidate_selection,
    run_candidate_selection,
)
from screening.metadata_queue import select_for_metadata_screening
from screening.queue import select_for_screening
from utils.corpus_store import CorpusStore


def paper(
    corpus_id,
    *,
    year=2022,
    citations=0,
    topic="adversarial_vision",
    title="General paper",
    abstract="Computer vision research.",
    topics=None,
):
    result = {
        "corpus_id": corpus_id,
        "year": year,
        "citation_count": citations,
        "title": title,
        "abstract": abstract,
    }
    if topics is not None:
        result["search_topics"] = topics
    else:
        result["search_topic"] = topic
    return result


class CandidateSelectionTests(unittest.TestCase):
    def test_core_recent_and_historical_citation_selection(self):
        cfg = {
            "candidate_selection": {
                "enabled": True,
                "historical_start_year": 2020,
                "historical_through_year": 2024,
                "recent_from_year": 2025,
                "top_n_per_year_per_stream": 2,
                "include_ties": True,
            },
        }
        papers = [
            # Separate historical streams are ranked independently.
            paper("a1", citations=100, topic="adversarial_vision"),
            paper("a2", citations=50, topic="adversarial_vision"),
            paper("a3", citations=50, topic="adversarial_vision"),
            paper("a4", citations=10, topic="adversarial_vision"),
            paper("v1", citations=80, topic="visual_forgery_detection"),
            paper("v2", citations=40, topic="visual_forgery_detection"),
            paper("v3", citations=5, topic="visual_forgery_detection"),
            # Both-stream retrieval is always protected.
            paper(
                "both",
                citations=0,
                topics=["visual_forgery_detection", "adversarial_vision"],
            ),
            # Lexical core protection bypasses citation rank.
            paper(
                "core",
                citations=0,
                topic="visual_forgery_detection",
                title="Adversarial attacks on deepfake detectors",
            ),
            # Recent records bypass citation filtering.
            paper(
                "recent",
                year=2026,
                citations=0,
                topic="adversarial_vision",
            ),
            # Missing historical citations are unresolved, not treated as zero.
            paper(
                "missing",
                citations=None,
                topic="adversarial_vision",
            ),
        ]

        rows, _, summary = build_candidate_selection(papers, cfg)
        by_id = {row["corpus_id"]: row for row in rows}

        self.assertEqual(by_id["a1"]["candidate_selection_status"], "selected")
        self.assertEqual(by_id["a2"]["candidate_selection_status"], "selected")
        self.assertEqual(by_id["a3"]["candidate_selection_status"], "selected")
        self.assertEqual(
            by_id["a3"]["candidate_selection_citation_rank"],
            2,
        )
        self.assertEqual(
            by_id["a4"]["candidate_selection_status"],
            "not_selected",
        )
        self.assertEqual(by_id["v1"]["candidate_selection_status"], "selected")
        self.assertEqual(by_id["v2"]["candidate_selection_status"], "selected")
        self.assertEqual(
            by_id["v3"]["candidate_selection_status"],
            "not_selected",
        )
        self.assertTrue(
            by_id["both"]["candidate_selection_core_protected"]
        )
        self.assertTrue(
            by_id["core"]["candidate_selection_core_protected"]
        )
        self.assertEqual(
            by_id["recent"]["candidate_selection_reason"],
            "recent_unfiltered",
        )
        self.assertEqual(
            by_id["missing"]["candidate_selection_status"],
            "unresolved",
        )
        self.assertEqual(
            by_id["missing"]["candidate_selection_reason"],
            "citation_unavailable",
        )
        self.assertEqual(summary.citation_selected, 5)
        self.assertEqual(summary.core_protected, 2)
        self.assertEqual(summary.recent_unfiltered, 1)

    def test_generative_adversarial_network_does_not_trigger_core(self):
        cfg = {
            "candidate_selection": {
                "enabled": True,
                "historical_start_year": 2020,
                "historical_through_year": 2024,
                "recent_from_year": 2025,
                "top_n_per_year_per_stream": 1,
            },
        }
        papers = [
            paper(
                "gan",
                year=2022,
                citations=1,
                topic="visual_forgery_detection",
                title="Deepfake generation with generative adversarial networks",
                abstract="We use generative adversarial networks for synthesis.",
            ),
            paper(
                "adv",
                year=2022,
                citations=0,
                topic="visual_forgery_detection",
                title="Adversarial attacks against deepfake detection",
                abstract="We evaluate adversarial examples.",
            ),
        ]
        rows, _, _ = build_candidate_selection(papers, cfg)
        by_id = {row["corpus_id"]: row for row in rows}
        self.assertFalse(
            by_id["gan"]["candidate_selection_core_protected"]
        )
        self.assertTrue(
            by_id["adv"]["candidate_selection_core_protected"]
        )

    def test_selection_persists_and_gates_metadata_queue(self):
        protocol = {
            "eligibility": {
                "criteria": [{
                    "id": "scope",
                    "kind": "inclusion",
                    "description": "Visual research.",
                }],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            cfg = {
                "output_dir": folder,
                "candidate_selection": {
                    "enabled": True,
                    "historical_start_year": 2020,
                    "historical_through_year": 2024,
                    "recent_from_year": 2025,
                    "top_n_per_year_per_stream": 1,
                    "manifest_dir": str(Path(folder) / "selection"),
                },
                "metadata_screening": {
                    "require_candidate_selection": True,
                },
                "screening": {"model": "qwen3.5:9b"},
            }
            with CorpusStore(Path(folder) / "corpus.sqlite3") as store:
                selected_id = store.upsert_paper({
                    "paper_id": "openalex:S1",
                    "title": "Paper one",
                    "abstract": "Visual research.",
                    "year": 2022,
                    "citation_count": 100,
                    "source": "openalex",
                    "search_topic": "adversarial_vision",
                })
                rejected_id = store.upsert_paper({
                    "paper_id": "openalex:S2",
                    "title": "Paper two",
                    "abstract": "Visual research.",
                    "year": 2022,
                    "citation_count": 1,
                    "source": "openalex",
                    "search_topic": "adversarial_vision",
                })

                summary = run_candidate_selection(store, cfg)
                self.assertEqual(summary.selected, 1)
                self.assertEqual(
                    store.get_paper(selected_id)["candidate_selection_status"],
                    "selected",
                )
                self.assertEqual(
                    store.get_paper(rejected_id)["candidate_selection_status"],
                    "not_selected",
                )

                with patch(
                    "screening.metadata_queue.get_model_digest",
                    return_value="model-a",
                ):
                    queue, queue_summary = select_for_metadata_screening(
                        store,
                        protocol,
                        cfg,
                    )

                self.assertEqual(
                    [row[0]["corpus_id"] for row in queue],
                    [selected_id],
                )
                self.assertEqual(
                    queue_summary.skipped_candidate_not_selected,
                    1,
                )

    def test_active_selection_also_gates_fulltext_queue(self):
        protocol = {
            "eligibility": {
                "criteria": [{
                    "id": "scope",
                    "kind": "inclusion",
                    "description": "Visual research.",
                }],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            cfg = {
                "output_dir": folder,
                "candidate_selection": {
                    "enabled": True,
                    "historical_start_year": 2020,
                    "historical_through_year": 2024,
                    "recent_from_year": 2025,
                    "top_n_per_year_per_stream": 1,
                    "manifest_dir": str(Path(folder) / "selection"),
                },
                "screening": {
                    "model": "qwen3.5:9b",
                    "require_metadata_screening": False,
                },
            }
            with CorpusStore(Path(folder) / "corpus.sqlite3") as store:
                keep = store.upsert_paper({
                    "paper_id": "openalex:F1",
                    "title": "Keep",
                    "abstract": "Visual research.",
                    "year": 2022,
                    "citation_count": 100,
                    "source": "openalex",
                    "search_topic": "adversarial_vision",
                })
                store.upsert_paper({
                    "paper_id": "openalex:F2",
                    "title": "Drop",
                    "abstract": "Visual research.",
                    "year": 2022,
                    "citation_count": 1,
                    "source": "openalex",
                    "search_topic": "adversarial_vision",
                })
                run_candidate_selection(store, cfg)

                with patch(
                    "screening.queue.get_model_digest",
                    return_value="model-a",
                ):
                    queue, queue_summary = select_for_screening(
                        store,
                        protocol,
                        cfg,
                    )

                self.assertEqual(
                    [row[0]["corpus_id"] for row in queue],
                    [keep],
                )
                self.assertEqual(
                    queue_summary.skipped_candidate_not_selected,
                    1,
                )

    def test_metadata_reset_archives_active_machine_state(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "corpus.sqlite3"
            with CorpusStore(db) as store:
                corpus_id = store.upsert_paper({
                    "paper_id": "openalex:R1",
                    "title": "Paper",
                    "year": 2022,
                    "source": "openalex",
                    "search_topic": "visual_forgery_detection",
                })
                paper_row = store.get_paper(corpus_id)
                paper_row.update({
                    "metadata_screening_status": "screened",
                    "metadata_screening_decision": "include",
                    "metadata_screening_reason": "old run",
                })
                store.update_metadata_screening(
                    corpus_id,
                    paper_row,
                    metadata_screening_signature="sig-old",
                )

                result = store.reset_metadata_screening(
                    artifact_archive="/tmp/archive",
                    reason="change selection policy",
                )

                self.assertEqual(result["archived_rows"], 1)
                reset_id = result["reset_id"]
                self.assertIsNone(
                    store.get_paper(corpus_id).get(
                        "metadata_screening_status"
                    )
                )
                archived = store.conn.execute(
                    """SELECT COUNT(*) AS n
                       FROM metadata_screening_archive
                       WHERE reset_id=?""",
                    (reset_id,),
                ).fetchone()["n"]
                self.assertEqual(archived, 1)


if __name__ == "__main__":
    unittest.main()
