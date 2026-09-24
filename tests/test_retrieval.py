import json
import unittest
from unittest.mock import Mock, patch
from datetime import date
import xml.etree.ElementTree as ET

from sources.openalex_source import iter_openalex_pages
from sources.arxiv_source import iter_arxiv_pages, _matches_query
from sources.scopus_source import iter_scopus_pages


def oai(records="", token=""):
    return ET.fromstring(
        f'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        f"<ListRecords>{records}<resumptionToken>{token}</resumptionToken>"
        f"</ListRecords></OAI-PMH>"
    )


def record(
    identifier="2301.00001",
    created="Sun, 01 Jan 2023 00:00:00 GMT",
):
    return f'''<record><header><identifier>oai:arXiv.org:{identifier}</identifier><datestamp>2026-09-24</datestamp></header>
    <metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/"><id>{identifier}</id><title>deepfake detection</title><abstract>adversarial</abstract><version version="v1"><date>{created}</date></version><version version="v2"><date>Thu, 24 Sep 2026 00:00:00 GMT</date></version></arXivRaw></metadata></record>'''


def scopus_entry(identifier, cover_date):
    return {
        "dc:identifier": f"SCOPUS_ID:{identifier}",
        "eid": f"2-s2.0-{identifier}",
        "dc:title": f"Paper {identifier}",
        "prism:coverDate": cover_date,
        "prism:aggregationType": "Journal",
    }


class RetrievalTests(unittest.TestCase):
    def response(self, ids, total, cursor):
        return {
            "meta": {"count": total, "next_cursor": cursor},
            "results": [{"id": x, "title": x} for x in ids],
        }

    @patch("sources.openalex_source.time.sleep")
    def test_openalex_cursor_and_cap(self, sleep):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.headers = {}
        session.get.return_value.json.side_effect = [
            self.response(["W1", "W2"], 3, "next"),
            self.response(["W3"], 3, None),
        ]
        pages = list(iter_openalex_pages("q", page_size=2, session=session))
        self.assertTrue(pages[-1]["complete"])
        self.assertEqual(pages[-1]["retrieved_count"], 3)

        session.get.return_value.json.side_effect = [
            self.response(["W1", "W2"], 3, "next")
        ]
        self.assertEqual(
            list(
                iter_openalex_pages(
                    "q",
                    max_results=2,
                    page_size=2,
                    session=session,
                )
            )[0]["stop_reason"],
            "max_results",
        )

    def test_openalex_uses_title_abstract_oqo_and_exact_dates(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.headers = {}
        session.get.return_value.json.return_value = self.response(
            ["W1"],
            1,
            None,
        )

        pages = list(
            iter_openalex_pages(
                '(deepfake OR "face forgery") AND detection',
                max_results=1,
                from_publication_date="2023-01-01",
                to_publication_date="2026-09-24",
                session=session,
            )
        )
        params = session.get.call_args.kwargs["params"]
        oqo = json.loads(params["oqo"])

        self.assertNotIn("search", params)
        self.assertEqual(pages[0]["search_scope"], "title_abstract")

        serialized = json.dumps(oqo)
        self.assertIn("title_and_abstract.search", serialized)
        self.assertIn("title_and_abstract.search.exact", serialized)
        self.assertNotIn("fulltext.search", serialized)

        date_filters = [
            node
            for node in oqo["filter_rows"]
            if node.get("column_id") == "publication_date"
        ]
        self.assertEqual(
            {(node["operator"], node["value"]) for node in date_filters},
            {(">=", "2023-01-01"), ("<=", "2026-09-24")},
        )

    def test_openalex_broken_cursor(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = self.response(
            ["W1"],
            3,
            "*",
        )
        with self.assertRaisesRegex(RuntimeError, "pagination"):
            list(iter_openalex_pages("q", session=session))

    def test_scopus_exact_date_cutoff(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.headers = {}
        session.get.return_value.json.return_value = {
            "search-results": {
                "opensearch:totalResults": "3",
                "entry": [
                    scopus_entry("1", "2026-12-01"),
                    scopus_entry("2", "2026-09-24"),
                    scopus_entry("3", "2025-01-01"),
                ],
            }
        }

        pages = list(
            iter_scopus_pages(
                "TITLE-ABS-KEY(deepfake)",
                api_key="secret",
                page_size=3,
                from_publication_date="2023-01-01",
                to_publication_date="2026-09-24",
                session=session,
            )
        )
        page = pages[-1]

        self.assertTrue(page["complete"])
        self.assertEqual(page["locally_excluded_count"], 1)
        self.assertEqual(page["exact_date_total"], 2)
        self.assertEqual(
            [paper["published_date"] for paper in page["papers"]],
            ["2026-09-24", "2025-01-01"],
        )
        self.assertEqual(
            page["exact_date_bounds"],
            {"from": "2023-01-01", "to": "2026-09-24"},
        )

        effective_query = session.get.call_args.kwargs["params"]["query"]
        self.assertIn("PUBYEAR > 2022", effective_query)
        self.assertIn("PUBYEAR < 2027", effective_query)

    @patch("sources.arxiv_source.time.sleep")
    @patch("sources.arxiv_source._request_oai")
    def test_arxiv_historical_window_includes_later_updates(
        self,
        request,
        sleep,
    ):
        request.side_effect = [
            oai(record(), "next"),
            oai(record("2301.00002")),
        ]
        pages = list(
            iter_arxiv_pages(
                "deepfake",
                from_publication_date="2023-01-01",
                to_publication_date="2023-12-31",
            )
        )
        self.assertEqual(
            request.call_args_list[0].args[0]["until"],
            date.today().isoformat(),
        )
        self.assertEqual(pages[0]["papers"][0]["published_date"], "2023-01-01")
        self.assertEqual(
            pages[0]["papers"][0]["oai_datestamp"],
            "2026-09-24",
        )
        self.assertTrue(pages[-1]["historical_scope_complete"])
        self.assertEqual(
            request.call_args_list[1].args[0],
            {"verb": "ListRecords", "resumptionToken": "next"},
        )

    @patch("sources.arxiv_source._request_oai")
    def test_arxiv_cap_and_update_scope(self, request):
        request.return_value = oai(
            record() + record("2301.00002")
        )
        page = list(
            iter_arxiv_pages(
                "deepfake",
                max_results=1,
                oai_from_date="2026-09-01",
            )
        )[0]
        self.assertFalse(page["complete"])
        self.assertEqual(page["coverage_scope"], "oai_update_window")

    @patch("sources.arxiv_source._request_oai")
    def test_arxiv_rejects_unsupported_syntax_before_request(self, request):
        for query in ("cat:cs.CV", "deepfake*", "deepfake AND"):
            with self.assertRaises(ValueError):
                list(
                    iter_arxiv_pages(
                        query,
                        oai_from_date="2026-09-01",
                    )
                )
        request.assert_not_called()

    def test_local_boolean(self):
        self.assertTrue(
            _matches_query(
                "Deepfake detection",
                "",
                "(deepfake OR synthetic) AND detection",
            )
        )
        self.assertFalse(
            _matches_query(
                "Deepfake detection",
                "",
                "deepfake ANDNOT detection",
            )
        )


if __name__ == "__main__":
    unittest.main()
