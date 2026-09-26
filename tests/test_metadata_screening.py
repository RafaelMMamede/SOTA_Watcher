import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from screening.metadata import (
    metadata_decision,
    metadata_screening_signature,
    screen_metadata_paper,
    validate_metadata_result,
)
from screening.metadata_queue import (
    metadata_config,
    screen_saved_metadata,
    select_for_metadata_screening,
)
from screening.pipeline import _criteria_for_paper
from screening.queue import select_for_screening
from screening.ollama import get_model_digest
from utils.corpus_store import CorpusStore


PROTOCOL = {
    "schema_version": 2,
    "searches": [
        {
            "id": "visual_forgery_detection",
            "queries": {"openalex": ["q"]},
        },
    ],
    "eligibility": {
        "criteria": [
            {
                "id": "visual_scope",
                "kind": "inclusion",
                "description": "Visual research.",
            },
            {
                "id": "non_visual_only",
                "kind": "exclusion",
                "description": "Exclusively non-visual research.",
            },
        ],
    },
}


def metadata_result(decision="include"):
    if decision == "include":
        return {
            "metadata_screening_status": "screened",
            "metadata_screening_decision": "include",
            "metadata_screening_reason": "test",
            "metadata_screening_criteria": [],
            "metadata_screening_evidence": [],
            "metadata_screening_artifact": "test",
        }
    if decision == "exclude":
        return {
            "metadata_screening_status": "screened",
            "metadata_screening_decision": "exclude",
            "metadata_screening_reason": "test",
            "metadata_screening_criteria": [],
            "metadata_screening_evidence": [],
            "metadata_screening_artifact": "test",
        }
    return {
        "metadata_screening_status": "screened",
        "metadata_screening_decision": "uncertain",
        "metadata_screening_reason": "test",
        "metadata_screening_criteria": [],
        "metadata_screening_evidence": [],
        "metadata_screening_artifact": "test",
    }


class MetadataScreeningTests(unittest.TestCase):
    def test_metadata_validation_accepts_safe_typography_only(self):
        criteria = [PROTOCOL["eligibility"]["criteria"][0]]
        paper = {
            "title": "Deepfake Detection",
            "abstract": "This section’s goal is visual detection.",
        }
        result = {
            "criteria": [{
                "id": "visual_scope",
                "assessment": "met",
                "reason": "Visual detection is explicit.",
                "evidence": [{
                    "source": "abstract",
                    "quote": "This section's goal is visual detection.",
                }],
            }],
        }
        rows = validate_metadata_result(result, criteria, paper)
        self.assertEqual(
            rows[0]["evidence"][0]["quote"],
            "This section’s goal is visual detection.",
        )

        bad = {
            "criteria": [{
                "id": "visual_scope",
                "assessment": "met",
                "reason": "Changed punctuation.",
                "evidence": [{
                    "source": "abstract",
                    "quote": "This sections goal is visual detection.",
                }],
            }],
        }
        with self.assertRaisesRegex(ValueError, "title/abstract"):
            validate_metadata_result(bad, criteria, paper)

    def test_metadata_uncertainty_is_not_exclusion(self):
        criteria = PROTOCOL["eligibility"]["criteria"]
        rows = [
            {
                "id": "visual_scope",
                "assessment": "met",
                "reason": "Visual scope explicit.",
                "evidence": [{"source": "title", "quote": "Visual"}],
            },
            {
                "id": "non_visual_only",
                "assessment": "uncertain",
                "reason": "Abstract does not settle exclusivity.",
                "evidence": [],
            },
        ]
        decision, _ = metadata_decision(rows, criteria)
        self.assertEqual(decision, "uncertain")

    def test_two_metadata_batches_advance(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = {
                "output_dir": folder,
                "screening": {"model": "qwen3.5:9b"},
                "metadata_screening": {
                    "artifacts_dir": str(Path(folder) / "metadata"),
                },
            }
            with CorpusStore(Path(folder) / "corpus.sqlite3") as store:
                for index in range(4):
                    store.upsert_paper({
                        "paper_id": f"openalex:W{index}",
                        "openalex_id": f"W{index}",
                        "title": f"Visual paper {index}",
                        "abstract": "We study image detection.",
                        "year": 2026,
                        "source": "openalex",
                        "search_topic": "visual_forgery_detection",
                    })

                with patch(
                    "screening.metadata_queue.get_model_digest",
                    return_value="model-a",
                ), patch(
                    "screening.metadata_queue.screen_metadata_paper",
                    side_effect=lambda *args, **kwargs: metadata_result("include"),
                ):
                    first = screen_saved_metadata(
                        store, PROTOCOL, cfg, limit=2
                    )
                    second = screen_saved_metadata(
                        store, PROTOCOL, cfg, limit=2
                    )
                    third = screen_saved_metadata(
                        store, PROTOCOL, cfg, limit=2
                    )

                self.assertEqual(first.selected, 2)
                self.assertEqual(second.selected, 2)
                self.assertEqual(third.selected, 0)
                self.assertEqual(first.included + second.included, 4)

    def test_metadata_exclude_blocks_fulltext_but_uncertain_advances(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = {
                "screening": {
                    "model": "qwen3.5:9b",
                    "require_metadata_screening": True,
                },
            }
            with CorpusStore(Path(folder) / "corpus.sqlite3") as store:
                ids = []
                for index, decision in enumerate(("exclude", "uncertain")):
                    corpus_id = store.upsert_paper({
                        "paper_id": f"openalex:G{index}",
                        "openalex_id": f"G{index}",
                        "title": f"Paper {index}",
                        "abstract": "We study visual research.",
                        "year": 2026,
                        "source": "openalex",
                        "search_topic": "visual_forgery_detection",
                    })
                    paper = store.get_paper(corpus_id)
                    active, _, _ = _criteria_for_paper(
                        PROTOCOL["eligibility"]["criteria"],
                        paper,
                    )
                    metadata_cfg = metadata_config(cfg)
                    signature = metadata_screening_signature(
                        paper,
                        active,
                        metadata_cfg,
                        "model-a",
                    )
                    paper.update(metadata_result(decision))
                    store.update_metadata_screening(
                        corpus_id,
                        paper,
                        metadata_screening_signature=signature,
                    )
                    ids.append(corpus_id)

                with patch(
                    "screening.queue.get_model_digest",
                    return_value="model-a",
                ):
                    selected, summary = select_for_screening(
                        store,
                        PROTOCOL,
                        cfg,
                    )

                self.assertEqual(
                    [paper["corpus_id"] for paper, _, _ in selected],
                    [ids[1]],
                )
                self.assertEqual(summary.skipped_metadata_excluded, 1)

    def test_new_search_family_requeues_metadata_if_criteria_change(self):
        protocol = {
            "schema_version": 2,
            "searches": [
                {"id": "visual", "queries": {"openalex": ["v"]}},
                {"id": "adversarial", "queries": {"openalex": ["a"]}},
            ],
            "eligibility": {
                "criteria": [
                    {
                        "id": "scope",
                        "kind": "inclusion",
                        "description": "Visual research.",
                    },
                    {
                        "id": "adv_only",
                        "kind": "exclusion",
                        "description": "Adversarial-only false positive.",
                        "applies_to_search_topics": ["adversarial"],
                        "skip_if_search_topics": ["visual"],
                    },
                ],
            },
        }
        cfg = {"screening": {"model": "qwen3.5:9b"}}

        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / "corpus.sqlite3") as store:
                corpus_id = store.upsert_paper({
                    "paper_id": "openalex:W1",
                    "openalex_id": "W1",
                    "title": "Paper",
                    "abstract": "Visual research.",
                    "year": 2026,
                    "source": "openalex",
                    "search_topic": "adversarial",
                })
                paper = store.get_paper(corpus_id)
                active, _, _ = _criteria_for_paper(
                    protocol["eligibility"]["criteria"], paper
                )
                signature = metadata_screening_signature(
                    paper,
                    active,
                    metadata_config(cfg),
                    "model-a",
                )
                paper.update(metadata_result("uncertain"))
                store.update_metadata_screening(
                    corpus_id,
                    paper,
                    metadata_screening_signature=signature,
                )

                store.upsert_paper({
                    "openalex_id": "W1",
                    "title": "Paper",
                    "abstract": "Visual research.",
                    "year": 2026,
                    "source": "openalex",
                    "search_topic": "visual",
                })

                with patch(
                    "screening.metadata_queue.get_model_digest",
                    return_value="model-a",
                ):
                    selected, summary = select_for_metadata_screening(
                        store,
                        protocol,
                        cfg,
                        limit=1,
                    )

                self.assertEqual(len(selected), 1)
                self.assertEqual(summary.stale, 1)

    @patch("screening.metadata.requests.post")
    def test_length_fallback_uses_no_thinking_structured_repair(self, post):
        criteria = [PROTOCOL["eligibility"]["criteria"][0]]
        paper = {
            "title": "Deepfake detection",
            "abstract": "We study visual research for image detection.",
        }

        def response(payload):
            mock = Mock()
            mock.raise_for_status = Mock()
            mock.json = Mock(return_value=payload)
            return mock

        valid = {
            "criteria": [{
                "id": "visual_scope",
                "assessment": "met",
                "reason": "Visual scope is explicit.",
                "evidence": [{
                    "source": "abstract",
                    "quote": "visual research for image detection",
                }],
            }],
        }
        post.side_effect = [
            response({
                "done": True,
                "done_reason": "stop",
                "message": {"content": "not json"},
            }),
            response({
                "done": True,
                "done_reason": "length",
                "eval_count": 3072,
                "message": {"content": ""},
            }),
            response({
                "done": True,
                "done_reason": "stop",
                "message": {"content": __import__("json").dumps(valid)},
            }),
        ]

        with tempfile.TemporaryDirectory() as folder:
            result = screen_metadata_paper(
                paper,
                criteria,
                {
                    "model": "qwen3.5:9b",
                    "base_url": "http://localhost:11434",
                    "num_ctx": 8192,
                    "fast_num_predict": 1200,
                    "fallback_enabled": True,
                    "fallback_think": "low",
                    "fallback_num_predict": 3072,
                    "repair_num_predict": 1600,
                },
                Path(folder),
                "model-a",
            )

        self.assertEqual(result["metadata_screening_mode"], "repair")
        self.assertEqual(result["metadata_screening_decision"], "include")
        self.assertEqual(post.call_count, 3)
        repair_payload = post.call_args_list[2].kwargs["json"]
        self.assertIs(repair_payload["think"], False)
        self.assertIn("format", repair_payload)

    @patch("screening.ollama.requests.get")
    def test_model_digest_lookup_queries_ollama(self, get):
        get.return_value = Mock(
            json=lambda: {
                "models": [{
                    "name": "qwen3.5:9b",
                    "digest": "sha256:model-a",
                }],
            },
        )
        get.return_value.raise_for_status = Mock()
        digest = get_model_digest({
            "model": "qwen3.5:9b",
            "base_url": "http://localhost:11434",
            "timeout_seconds": 10,
        })
        self.assertEqual(digest, "sha256:model-a")
        get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=10,
        )


if __name__ == "__main__":
    unittest.main()
