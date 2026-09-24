import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf

from fulltext import resolve_paper, fetch_pdf, extract_pdf
from fulltext._common import sha256


def json_response(data=None, status=200, url="https://api.example/result"):
    response = Mock(status_code=status, headers={}, url=url)
    response.json.return_value = data
    return response


def pdf_response(data, url="https://repo.example/paper.pdf"):
    response = Mock(status_code=200, headers={}, url=url)
    response.iter_content.return_value = [data]
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    return response


def openalex_work(pdf_url=None):
    best = None
    if pdf_url:
        best = {
            "is_oa": True,
            "pdf_url": pdf_url,
            "landing_page_url": "https://repo.example/item",
            "version": "acceptedVersion",
            "license": "cc-by",
            "source": {
                "display_name": "Example Repository",
                "type": "repository",
            },
        }
    return {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1234/example",
        "title": "Example Paper",
        "publication_year": 2025,
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
        ],
        "best_oa_location": best,
        "locations": [best] if best else [],
    }


class FulltextTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.pdf"
        self.make_pdf("First document")
        self.output = self.root / "paper"

    def make_pdf(self, text):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), text)
            doc.save(self.source)

    def test_resolution(self):
        r = resolve_paper({"url": "https://arxiv.org/abs/2609.10002v1"})
        self.assertEqual(r["url"], "https://arxiv.org/pdf/2609.10002v1")
        self.assertTrue(r["version_pinned"])

        r = resolve_paper({"arxiv_id": "hep-th/9901001v2"})
        self.assertEqual(r["arxiv_id"], "hep-th/9901001v2")

        self.assertEqual(
            resolve_paper(
                {"pdf_url": "https://publisher.example/file.pdf"},
                resolver_config={"enabled": False},
            )["status"],
            "unavailable",
        )
        self.assertEqual(
            resolve_paper(
                {"arxiv_id": "../../secret"},
                resolver_config={"enabled": False},
            )["status"],
            "unavailable",
        )

    def test_openalex_resolves_doi_to_open_repository_pdf(self):
        session = Mock()
        session.get.return_value = json_response(
            openalex_work("https://repo.example/paper.pdf")
        )

        resolution = resolve_paper(
            {
                "doi": "https://doi.org/10.1234/example",
                "title": "Example Paper",
                "year": 2025,
            },
            resolver_config={"email": "researcher@example.org"},
            session=session,
        )

        self.assertEqual(resolution["status"], "resolved")
        self.assertEqual(resolution["resolver"], "openalex")
        self.assertEqual(resolution["kind"], "remote")
        self.assertTrue(resolution["is_oa"])
        self.assertEqual(
            resolution["url"],
            "https://repo.example/paper.pdf",
        )
        self.assertEqual(
            resolution["fulltext_source"],
            "Example Repository",
        )
        self.assertEqual(resolution["resolved_via"], "doi")

    def test_environment_email_overrides_yaml_placeholder(self):
        session = Mock()
        session.get.return_value = json_response({"is_oa": False})

        with patch.dict(
            "os.environ",
            {"UNPAYWALL_EMAIL": "real@example.org"},
            clear=False,
        ):
            resolution = resolve_paper(
                {"doi": "10.1234/example"},
                resolver_config={
                    "email": "your_email@example.com",
                    "openalex": False,
                    "semantic_scholar": False,
                    "crossref_metadata": False,
                },
                session=session,
            )

        self.assertEqual(resolution["status"], "unavailable")
        self.assertEqual(
            session.get.call_args.kwargs["params"]["email"],
            "real@example.org",
        )

    def test_unpaywall_fallback_after_openalex_has_no_pdf(self):
        session = Mock()
        session.get.side_effect = [
            json_response(openalex_work()),
            json_response(
                {
                    "is_oa": True,
                    "best_oa_location": {
                        "url_for_pdf": "https://green.example/paper.pdf",
                        "url_for_landing_page": "https://green.example/item",
                        "version": "acceptedVersion",
                        "license": "cc-by",
                        "host_type": "repository",
                    },
                    "oa_locations": [],
                }
            ),
        ]

        resolution = resolve_paper(
            {"doi": "10.1234/example"},
            resolver_config={
                "email": "researcher@example.org",
                "semantic_scholar": False,
                "crossref_metadata": False,
            },
            session=session,
        )

        self.assertEqual(resolution["resolver"], "unpaywall")
        self.assertEqual(
            resolution["url"],
            "https://green.example/paper.pdf",
        )
        self.assertEqual(
            [a["status"] for a in resolution["resolver_attempts"]],
            ["no_open_pdf", "resolved"],
        )

    def test_semantic_scholar_fallback(self):
        session = Mock()
        session.get.side_effect = [
            json_response(openalex_work()),
            json_response(status=404),
            json_response(
                {
                    "paperId": "S2",
                    "url": "https://www.semanticscholar.org/paper/S2",
                    "openAccessPdf": {
                        "url": "https://s2.example/paper.pdf",
                    },
                }
            ),
        ]

        resolution = resolve_paper(
            {"doi": "10.1234/example"},
            resolver_config={
                "email": "researcher@example.org",
                "crossref_metadata": False,
            },
            session=session,
        )

        self.assertEqual(resolution["resolver"], "semantic_scholar")
        self.assertEqual(
            resolution["url"],
            "https://s2.example/paper.pdf",
        )

    def test_exact_title_fallback_can_enrich_and_resolve(self):
        work = openalex_work("https://repo.example/title-match.pdf")
        session = Mock()
        session.get.return_value = json_response({"results": [work]})

        resolution = resolve_paper(
            {
                "title": "Example Paper",
                "year": 2025,
                "authors": "Alice Smith",
            },
            resolver_config={
                "email": "researcher@example.org",
                "unpaywall": False,
                "semantic_scholar": False,
                "crossref_metadata": False,
            },
            session=session,
        )

        self.assertEqual(resolution["resolver"], "openalex")
        self.assertEqual(
            resolution["resolved_via"],
            "title_year_author",
        )
        self.assertEqual(
            resolution["resolved_doi"],
            "10.1234/example",
        )

    def test_crossref_links_are_manual_candidates_not_auto_downloads(self):
        session = Mock()
        session.get.side_effect = [
            json_response(openalex_work()),
            json_response({"is_oa": False}),
            json_response(status=404),
            json_response(
                {
                    "message": {
                        "link": [
                            {
                                "URL": "https://publisher.example/tdm.pdf",
                                "content-type": "application/pdf",
                                "content-version": "vor",
                                "intended-application": "text-mining",
                            }
                        ]
                    }
                }
            ),
        ]

        resolution = resolve_paper(
            {"doi": "10.1234/example"},
            resolver_config={"email": "researcher@example.org"},
            session=session,
        )

        self.assertEqual(resolution["status"], "unavailable")
        self.assertEqual(len(resolution["manual_candidates"]), 1)
        self.assertFalse(
            resolution["manual_candidates"][0]["auto_download"]
        )

    def test_local_cache_invalidates_when_source_changes(self):
        resolution = resolve_paper(local_pdf=self.source)
        first = fetch_pdf(resolution, self.output)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(fetch_pdf(resolution, self.output)["cache_hit"])

        self.make_pdf("Changed document")
        changed = fetch_pdf(resolution, self.output)
        self.assertFalse(changed["cache_hit"])
        self.assertNotEqual(first["pdf_sha256"], changed["pdf_sha256"])

    def test_failed_download_preserves_existing_pdf(self):
        fetch_pdf(resolve_paper(local_pdf=self.source), self.output)
        old_hash = sha256(self.output / "paper.pdf")

        response = Mock(
            status_code=200,
            url="https://arxiv.org/pdf/2609.10002v1",
        )
        response.iter_content.return_value = [
            b"<html>access denied</html>"
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)

        session = Mock()
        session.get.return_value = response

        with self.assertRaises(ValueError):
            fetch_pdf(
                resolve_paper({"arxiv_id": "2609.10002v1"}),
                self.output,
                session=session,
            )

        self.assertEqual(
            sha256(self.output / "paper.pdf"),
            old_hash,
        )
        self.assertTrue(
            (self.output / "download_error.json").is_file()
        )

    def test_http_success_and_cache(self):
        resolution = resolve_paper({"arxiv_id": "2609.10002v1"})
        response = pdf_response(
            self.source.read_bytes(),
            resolution["url"],
        )
        session = Mock()
        session.get.return_value = response

        result = fetch_pdf(
            resolution,
            self.output,
            session=session,
        )
        self.assertEqual(
            result["pdf_sha256"],
            sha256(self.source),
        )
        self.assertTrue(
            fetch_pdf(
                resolution,
                self.output,
                session=session,
            )["cache_hit"]
        )
        self.assertEqual(session.get.call_count, 1)

    def test_trusted_remote_oa_pdf_download(self):
        resolution = {
            "status": "resolved",
            "kind": "remote",
            "resolver": "openalex",
            "is_oa": True,
            "url": "https://repo.example/paper.pdf",
        }
        session = Mock()
        session.get.return_value = pdf_response(
            self.source.read_bytes(),
            resolution["url"],
        )

        result = fetch_pdf(
            resolution,
            self.output,
            session=session,
        )
        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(
            result["pdf_sha256"],
            sha256(self.source),
        )

    def test_untrusted_remote_candidate_is_rejected(self):
        resolution = {
            "status": "resolved",
            "kind": "remote",
            "resolver": "crossref",
            "is_oa": True,
            "url": "https://publisher.example/paper.pdf",
        }
        with self.assertRaisesRegex(ValueError, "trusted OA resolver"):
            fetch_pdf(resolution, self.output)

    def test_size_limit(self):
        with self.assertRaises(ValueError):
            fetch_pdf(
                resolve_paper(local_pdf=self.source),
                self.output,
                max_bytes=10,
            )
        self.assertFalse((self.output / "paper.pdf").exists())

    def test_extraction_cache_and_markdown_recovery(self):
        fetch_pdf(resolve_paper(local_pdf=self.source), self.output)
        pdf = self.output / "paper.pdf"

        first = extract_pdf(pdf)
        self.assertEqual(first["page_count"], 1)
        self.assertEqual(first["pages"][0]["page"], 1)
        self.assertIn("First document", first["pages"][0]["text"])
        self.assertTrue(extract_pdf(pdf)["cache_hit"])

        (self.output / "paper_layout.md").write_text("damaged")
        self.assertFalse(extract_pdf(pdf)["cache_hit"])
        self.assertIn(
            "PDF PAGE 1",
            (self.output / "paper_layout.md").read_text(),
        )

        self.make_pdf("New PDF content")
        fetch_pdf(resolve_paper(local_pdf=self.source), self.output)
        changed = extract_pdf(pdf)
        self.assertFalse(changed["cache_hit"])
        self.assertIn(
            "New PDF content",
            changed["pages"][0]["text"],
        )


if __name__ == "__main__":
    unittest.main()
