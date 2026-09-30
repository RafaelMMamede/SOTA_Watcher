import tempfile
import unittest
from pathlib import Path

from utils.corpus_deduplication import (
    find_duplicate_clusters,
    normalized_title,
    run_corpus_deduplication,
)
from utils.corpus_store import CorpusStore


class CorpusDeduplicationTests(unittest.TestCase):
    def test_normalized_title_ignores_case_and_punctuation(self):
        self.assertEqual(
            normalized_title({"title": "Deepfake Detection: A Survey"}),
            normalized_title({"title": "deepfake detection — a survey"}),
        )

    def test_exact_title_requires_compatible_authors_and_year(self):
        papers = [
            {
                "corpus_id": "p_a",
                "title": "Adversarial Attacks Against Deepfake Detectors",
                "authors": "Alice Smith, Bob Jones",
                "year": 2023,
            },
            {
                "corpus_id": "p_b",
                "title": "Adversarial attacks against deepfake detectors",
                "authors": "Smith, Alice, Jones, Bob",
                "year": 2024,
            },
            {
                "corpus_id": "p_c",
                "title": "Adversarial attacks against deepfake detectors",
                "authors": "Carla Other, Daniel Person",
                "year": 2023,
            },
        ]
        clusters = find_duplicate_clusters(papers, include_fuzzy=False)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            set(clusters[0]["member_ids"]),
            {"p_a", "p_b"},
        )

    def test_fuzzy_high_confidence_title_variant_is_detected(self):
        papers = [
            {
                "corpus_id": "p_a",
                "title": (
                    "What Makes Adversarial Examples Transfer Across "
                    "Deepfake Detectors?"
                ),
                "authors": "Rafael M. Mamede, Pedro C. Neto, Ana F. Sequeira",
                "year": 2026,
            },
            {
                "corpus_id": "p_b",
                "title": (
                    "What Makes Adversarial Examples Transfer Across "
                    "Deepfake Detectors"
                ),
                "authors": "Mamede Rafael, Neto Pedro, Sequeira Ana",
                "year": 2026,
            },
        ]
        clusters = find_duplicate_clusters(papers, include_fuzzy=True)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            set(clusters[0]["member_ids"]),
            {"p_a", "p_b"},
        )

    def test_apply_preserves_reviews_and_conflict_resolves_to_include(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = {
                "output_dir": str(folder),
                "deduplication": {
                    "manifest_dir": str(folder / "dedup"),
                },
            }
            with CorpusStore(folder / "corpus.sqlite3") as store:
                preprint = store.upsert_paper({
                    "paper_id": "arxiv:1234.5678",
                    "arxiv_id": "1234.5678",
                    "title": "Robust Deepfake Detection Under Adversarial Attacks",
                    "authors": "Alice Smith, Bob Jones",
                    "year": 2023,
                    "venue": "arXiv",
                    "venue_type": "repository",
                    "abstract": "A longer preprint abstract.",
                })
                published = store.upsert_paper({
                    "paper_id": "scopus:999",
                    "scopus_id": "999",
                    "doi": "10.1234/example",
                    "title": "Robust Deepfake Detection Under Adversarial Attacks",
                    "authors": "Smith, Alice, Jones, Bob",
                    "year": 2024,
                    "venue": "Example Conference",
                    "venue_type": "conference",
                    "publication_type": "Conference Paper",
                    "abstract": "Published abstract.",
                })

                store.set_human_metadata_screening(
                    preprint,
                    "include",
                    reason="Human include.",
                )
                store.set_human_metadata_screening(
                    published,
                    "exclude",
                    reason="Human exclude.",
                )

                preprint_row = store.get_paper(preprint)
                preprint_row.update({
                    "metadata_screening_status": "screened",
                    "metadata_screening_decision": "include",
                    "metadata_screening_reason": "Valid model result.",
                })
                store.update_metadata_screening(
                    preprint,
                    preprint_row,
                    metadata_screening_signature="screened-sig",
                )

                published_row = store.get_paper(published)
                published_row.update({
                    "metadata_screening_status": "error",
                    "metadata_screening_decision": "uncertain",
                    "metadata_screening_reason": "Model error.",
                    "metadata_screening_error_type": "ValueError",
                })
                store.update_metadata_screening(
                    published,
                    published_row,
                    metadata_screening_signature="error-sig",
                )

                preview = run_corpus_deduplication(
                    store,
                    config,
                    apply=False,
                    include_fuzzy=False,
                )
                self.assertEqual(preview.duplicate_clusters, 1)
                self.assertEqual(preview.human_metadata_conflicts, 1)
                self.assertEqual(len(store.all_papers()), 2)

                applied = run_corpus_deduplication(
                    store,
                    config,
                    apply=True,
                    include_fuzzy=False,
                )
                self.assertEqual(applied.applied_clusters, 1)
                self.assertEqual(applied.total_after, 1)
                self.assertTrue(Path(applied.backup_path).exists())

                papers = store.all_papers()
                self.assertEqual(len(papers), 1)
                merged = papers[0]

                # Published version is canonical, but the reviewed preprint's
                # state is transferred into it.
                self.assertEqual(merged["corpus_id"], published)
                self.assertEqual(
                    merged["human_metadata_screening_decision"],
                    "include",
                )
                self.assertIn(
                    "conflicting human title/abstract screening decisions",
                    merged["human_metadata_screening_reason"],
                )

                # A successful automated screen beats a newer/alternate error.
                self.assertEqual(
                    merged["metadata_screening_status"],
                    "screened",
                )
                self.assertEqual(
                    merged["metadata_screening_decision"],
                    "include",
                )

    def test_final_manual_decision_conflict_blocks_auto_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = {"output_dir": str(folder)}
            with CorpusStore(folder / "corpus.sqlite3") as store:
                first = store.upsert_paper({
                    "paper_id": "openalex:W1",
                    "title": "Same Study Title for Duplicate Records",
                    "authors": "Alice Smith, Bob Jones",
                    "year": 2024,
                })
                second = store.upsert_paper({
                    "paper_id": "scopus:2",
                    "title": "Same Study Title for Duplicate Records",
                    "authors": "Alice Smith, Bob Jones",
                    "year": 2024,
                })
                store.set_human_review(first, "include")
                store.set_human_review(second, "exclude")

                result = run_corpus_deduplication(
                    store,
                    config,
                    apply=True,
                    include_fuzzy=False,
                )
                self.assertEqual(result.blocked_manual_conflicts, 1)
                self.assertEqual(result.applied_clusters, 0)
                self.assertEqual(len(store.all_papers()), 2)


if __name__ == "__main__":
    unittest.main()
