import sqlite3
import tempfile
import unittest
from pathlib import Path

from docx import Document

from screening.human_export import export_human_included_docx, load_human_included


class StubStore:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE candidate_selection (corpus_id TEXT, status TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE human_metadata_screening (corpus_id TEXT, decision TEXT)"
        )
        self.papers = {}
        self.run = {"run_id": "test_active"}

    def add(self, corpus_id, status, decision, **kwargs):
        self.conn.execute(
            "INSERT INTO candidate_selection VALUES (?, ?)", (corpus_id, status)
        )
        if decision is not None:
            self.conn.execute(
                "INSERT INTO human_metadata_screening VALUES (?, ?)",
                (corpus_id, decision),
            )
        self.papers[corpus_id] = {
            "corpus_id": corpus_id,
            "title": f"Title of {corpus_id}",
            "abstract": f"Abstract of {corpus_id}",
            "year": 2026,
            "candidate_selection_core_protected": False,
            "candidate_selection_reason": "citation_top_n",
            "candidate_selection_stream": "visual",
            **kwargs,
        }
        self.conn.commit()

    def latest_candidate_selection_run(self):
        return self.run

    def get_paper(self, corpus_id):
        return self.papers[corpus_id]


class HumanExportTests(unittest.TestCase):
    def setUp(self):
        self.store = StubStore()
        self.store.add(
            "core", "selected", "include", year=2026,
            candidate_selection_core_protected=True,
            candidate_selection_reason="core_protected",
            authors=[{"name": "Author One"}, {"name": "Author Two"}],
            doi="10.1000/core",
        )
        self.store.add("support", "selected", "include", year=2025)
        self.store.add("rejected", "selected", "exclude")
        self.store.add("pending", "selected", None)
        self.store.add("dropped", "not_selected", "include")

    def tearDown(self):
        self.store.conn.close()

    def test_export_contains_only_active_human_includes(self):
        papers, counts, run = load_human_included(self.store)
        self.assertEqual([p["corpus_id"] for p in papers], ["core", "support"])
        self.assertEqual(counts, {"include": 2, "exclude": 1, "pending": 1})
        self.assertEqual(run["run_id"], "test_active")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subfolder" / "included.docx"
            result = export_human_included_docx(self.store, path)
            self.assertEqual(result["human_included"], 2)
            self.assertEqual(result["human_excluded"], 1)
            self.assertEqual(result["selected"], 4)
            self.assertTrue(path.exists())

            document = Document(path)
            text = "\n".join(p.text for p in document.paragraphs)
            self.assertIn("Title of core", text)
            self.assertIn("Title of support", text)
            self.assertIn("Author One, Author Two", text)
            self.assertIn("Abstract of core", text)
            self.assertIn("10.1000/core", text)
            self.assertNotIn("Title of rejected", text)
            self.assertNotIn("Title of pending", text)
            self.assertNotIn("Title of dropped", text)
            self.assertIn("full-text review", text.lower())

    def test_compact_export_omits_abstracts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.docx"
            result = export_human_included_docx(
                self.store, path, include_abstracts=False
            )
            document = Document(path)
            text = "\n".join(p.text for p in document.paragraphs)
            self.assertIn("Title of core", text)
            self.assertNotIn("Abstract of core", text)
            self.assertFalse(result["abstracts_included"])

    def test_markdown_export_is_chat_readable(self):
        from screening.human_export import export_human_included_markdown

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "included.md"
            result = export_human_included_markdown(self.store, path)
            self.assertEqual(result["human_included"], 2)
            content = path.read_text(encoding="utf-8")
            self.assertIn("# SOTA_Watcher", content)
            self.assertIn("Title of core", content)
            self.assertIn("Title of support", content)
            self.assertIn("Abstract of core", content)
            self.assertIn("10.1000/core", content)
            self.assertNotIn("Title of rejected", content)
            self.assertNotIn("Title of dropped", content)
            self.assertIn("## Included papers", content)
            self.assertGreater(content.count("\\n"), 15)

    def test_requires_active_candidate_selection(self):
        self.store.run = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "should_not_exist.docx"
            with self.assertRaisesRegex(ValueError, "No active"):
                export_human_included_docx(self.store, path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
