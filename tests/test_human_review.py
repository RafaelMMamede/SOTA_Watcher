import tempfile
import unittest
from pathlib import Path

from screening.human_review import (
    format_human_screen_paper,
    run_human_screening,
    select_for_human_screening,
)
from screening.queue import select_for_screening
from utils.corpus_store import CorpusStore


PROTOCOL = {
    "eligibility": {
        "criteria": [{
            "id": "scope",
            "kind": "inclusion",
            "description": "Visual research.",
        }],
    },
}


class HumanReviewTests(unittest.TestCase):
    def _selected_store(self, folder):
        store = CorpusStore(Path(folder) / "corpus.sqlite3")
        ids = []
        for index in range(3):
            corpus_id = store.upsert_paper({
                "paper_id": f"openalex:H{index}",
                "title": f"Paper {index}",
                "abstract": f"Abstract {index}",
                "year": 2022 + index,
                "citation_count": 100 - index,
                "source": "openalex",
                "search_topic": "adversarial_vision",
            })
            ids.append(corpus_id)

        policy = {
            "top_n_per_year_per_stream": 100,
        }
        records = []
        for index, corpus_id in enumerate(ids):
            records.append({
                "corpus_id": corpus_id,
                "candidate_selection_title": f"Paper {index}",
                "candidate_selection_search_topics": ["adversarial_vision"],
                "candidate_selection_status": "selected",
                "candidate_selection_stream": "adversarial",
                "candidate_selection_reason": "citation_top_n",
                "candidate_selection_year": 2022 + index,
                "candidate_selection_citation_count": 100 - index,
                "candidate_selection_citation_rank": index + 1,
                "candidate_selection_citation_quota": 100,
                "candidate_selection_core_protected": False,
                "candidate_selection_core_reason": "",
                "candidate_selection_citation_metric": "test",
            })
        store.replace_candidate_selection(
            policy,
            records,
            {"selected": 3},
        )
        return store, ids

    def test_core_only_filters_to_core_protected_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            store, ids = self._selected_store(folder)
            try:
                records = []
                for index, corpus_id in enumerate(ids):
                    is_core = index == 0
                    records.append({
                        "corpus_id": corpus_id,
                        "candidate_selection_title": f"Paper {index}",
                        "candidate_selection_search_topics": ["adversarial_vision"],
                        "candidate_selection_status": "selected",
                        "candidate_selection_stream": "adversarial",
                        "candidate_selection_reason": (
                            "core_protected" if is_core else "citation_top_n"
                        ),
                        "candidate_selection_year": 2022 + index,
                        "candidate_selection_citation_count": 100 - index,
                        "candidate_selection_citation_rank": (
                            None if is_core else index + 1
                        ),
                        "candidate_selection_citation_quota": 100,
                        "candidate_selection_core_protected": is_core,
                        "candidate_selection_core_reason": (
                            "deepfake_and_adversarial_concepts" if is_core else ""
                        ),
                        "candidate_selection_citation_metric": "test",
                    })
                store.replace_candidate_selection(
                    {"top_n_per_year_per_stream": 100},
                    records,
                    {"selected": 3},
                )

                pending, summary = select_for_human_screening(
                    store,
                    core_only=True,
                )
                self.assertEqual(
                    [paper["corpus_id"] for paper in pending],
                    [ids[0]],
                )
                self.assertEqual(summary.eligible_pool, 1)
                self.assertEqual(summary.remaining, 1)
            finally:
                store.close()

    def test_human_screen_is_resumable_and_separate_from_final_review(self):
        with tempfile.TemporaryDirectory() as folder:
            store, ids = self._selected_store(folder)
            try:
                answers = iter(["y", "n", "q"])
                summary = run_human_screening(
                    store,
                    input_fn=lambda _: next(answers),
                    output_fn=lambda _: None,
                )

                self.assertEqual(summary.session_reviewed, 2)
                self.assertEqual(summary.included, 1)
                self.assertEqual(summary.excluded, 1)
                self.assertTrue(summary.quit)

                first = store.get_paper(ids[0])
                second = store.get_paper(ids[1])
                self.assertEqual(
                    first["human_metadata_screening_decision"],
                    "include",
                )
                self.assertEqual(
                    second["human_metadata_screening_decision"],
                    "exclude",
                )
                self.assertEqual(first["manual_decision"], "")
                self.assertEqual(second["manual_decision"], "")

                pending, resumed = select_for_human_screening(store)
                self.assertEqual(
                    [paper["corpus_id"] for paper in pending],
                    [ids[2]],
                )
                self.assertEqual(resumed.already_reviewed, 2)
                self.assertEqual(resumed.remaining, 1)
            finally:
                store.close()

    def test_human_metadata_include_advances_and_exclude_blocks_fulltext(self):
        with tempfile.TemporaryDirectory() as folder:
            store, ids = self._selected_store(folder)
            try:
                store.set_human_metadata_screening(ids[0], "include")
                store.set_human_metadata_screening(ids[1], "exclude")

                config = {
                    "candidate_selection": {"enabled": True},
                    "metadata_screening": {
                        "require_candidate_selection": True,
                    },
                    "screening": {
                        "model": "qwen3.5:9b",
                        "require_metadata_screening": True,
                    },
                }

                from unittest.mock import patch
                with patch(
                    "screening.queue.get_model_digest",
                    return_value="model-a",
                ):
                    selected, summary = select_for_screening(
                        store,
                        PROTOCOL,
                        config,
                    )

                self.assertIn(ids[0], [row[0]["corpus_id"] for row in selected])
                self.assertNotIn(ids[1], [row[0]["corpus_id"] for row in selected])
                self.assertEqual(summary.skipped_metadata_excluded, 1)
            finally:
                store.close()

    def test_render_includes_useful_metadata_and_model_decision(self):
        paper = {
            "corpus_id": "p_test",
            "title": "Adversarial Deepfake Detection",
            "abstract": "We study robust deepfake detection.",
            "year": 2024,
            "citation_count": 42,
            "sources": ["openalex", "scopus"],
            "search_topics": ["visual_forgery_detection"],
            "candidate_selection_stream": "visual",
            "candidate_selection_reason": "citation_top_n",
            "candidate_selection_citation_rank": 10,
            "candidate_selection_citation_quota": 100,
            "metadata_screening_status": "screened",
            "metadata_screening_decision": "include",
            "metadata_screening_reason": "In scope.",
        }
        rendered = format_human_screen_paper(
            paper,
            index=1,
            total=10,
            reviewed=0,
        )
        self.assertIn("Adversarial Deepfake Detection", rendered)
        self.assertIn("We study robust deepfake detection.", rendered)
        self.assertIn("42", rendered)
        self.assertIn("10 / top 100", rendered)
        self.assertIn("Decision:        include", rendered)


if __name__ == "__main__":
    unittest.main()
