"""Offline contract tests. Run: python -m unittest discover -s tests -v"""
import unittest
from unittest.mock import Mock, patch

import requests

from sources.ieee_source import iter_ieee_pages, search_ieee
from sources.scopus_source import iter_scopus_pages, search_scopus


def response(data=None, status=200, headers=None):
    obj = Mock(status_code=status, headers=headers or {})
    obj.json.return_value = data
    return obj


def ieee_page(ids, total):
    return {"total_records": total, "articles": [
        {"article_number": str(i), "title": f"Paper {i}", "abstract": "Some evidence",
         "authors": {"authors": [{"full_name": "A. Researcher"}]},
         "publication_year": 2026, "content_type": "Conferences",
         "doi": f"10.1234/TEST{i}", "pdf_url": "https://example.org/paper.pdf",
         "accessType": "Locked"} for i in ids]}


def scopus_page(ids, total):
    return {"search-results": {"opensearch:totalResults": str(total), "entry": [
        {"dc:identifier": f"SCOPUS_ID:{i}", "eid": f"2-s2.0-{i}",
         "dc:title": f"Paper {i}", "dc:creator": "A. Researcher",
         "prism:doi": f"10.1234/TEST{i}", "prism:coverDate": "2026-01-01",
         "prism:aggregationType": "Journal"} for i in ids]}}


class SourcesTest(unittest.TestCase):
    def client(self, *responses):
        client = Mock()
        # Capture params before the next page mutates the adapter's dictionary.
        client.calls = []
        items = iter(responses)
        def get(url, **kwargs):
            client.calls.append((url, {**kwargs, "params": dict(kwargs["params"])}))
            return next(items)
        client.get.side_effect = get
        return client

    def test_ieee_pagination_metadata_and_secret_free_provenance(self):
        client = self.client(response(ieee_page([1, 2], 3)), response(ieee_page([3], 3)))
        pages = list(iter_ieee_pages('deepfake AND adversarial', api_key="secret", page_size=2,
                                    start_year=2020, end_year=2026, session=client, sleep_seconds=0))
        self.assertEqual([c[1]["params"]["start_record"] for c in client.calls], [1, 3])
        self.assertFalse(pages[0]["complete"])
        self.assertTrue(pages[-1]["complete"])
        self.assertEqual(pages[-1]["retrieved_count"], 3)
        paper = pages[0]["papers"][0]
        self.assertEqual(paper["doi"], "https://doi.org/10.1234/test1")
        self.assertEqual(paper["venue_type"], "conference")
        self.assertFalse(paper["is_open_access"])
        self.assertNotIn("secret", repr(pages))
        client.close.assert_not_called()

    def test_scopus_headers_native_query_and_missing_abstract(self):
        client = self.client(response(scopus_page([1], 2)), response(scopus_page([2], 2)))
        query = 'TITLE-ABS-KEY(deepfake AND adversarial)'
        pages = list(iter_scopus_pages(query, api_key="secret", insttoken="token", page_size=1,
                                      start_year=2020, end_year=2026, session=client, sleep_seconds=0))
        self.assertEqual([c[1]["params"]["start"] for c in client.calls], [0, 1])
        call = client.calls[0][1]
        self.assertEqual(call["headers"]["X-ELS-APIKey"], "secret")
        self.assertEqual(call["headers"]["X-ELS-Insttoken"], "token")
        self.assertEqual(call["params"]["query"], f"({query}) AND PUBYEAR > 2019 AND PUBYEAR < 2027")
        self.assertFalse(pages[0]["papers"][0]["has_abstract"])
        self.assertTrue(pages[-1]["complete"])
        self.assertNotIn("secret", repr(pages))

    def test_complete_view_extracts_abstract_and_authors(self):
        data = scopus_page([1], 1)
        data["search-results"]["entry"][0].update({"dc:description": "Full abstract", "author": [{"authname": "First"}, {"authname": "Second"}]})
        paper = search_scopus("deepfake", api_key="key", view="COMPLETE", session=self.client(response(data)))[0]
        self.assertEqual(paper["abstract"], "Full abstract")
        self.assertEqual(paper["authors"], "First, Second")

    def test_empty_results(self):
        for iterator, data in [(iter_ieee_pages, ieee_page([], 0)),
                               (iter_scopus_pages, {"search-results": {"opensearch:totalResults": "0", "entry": [{"error": "No results"}]}})]:
            with self.subTest(source=iterator.__name__):
                page = list(iterator("deepfake", api_key="key", session=self.client(response(data))))[0]
                self.assertTrue(page["complete"])
                self.assertEqual(page["papers"], [])

    def test_caps_warn_and_mark_incomplete(self):
        for search, iterator, data in [(search_ieee, iter_ieee_pages, ieee_page([1], 10)),
                                      (search_scopus, iter_scopus_pages, scopus_page([1], 10))]:
            with self.subTest(source=search.__name__):
                page = list(iterator("deepfake", api_key="key", max_results=1, session=self.client(response(data))))[0]
                self.assertFalse(page["complete"])
                self.assertEqual(page["stop_reason"], "max_results")
                with self.assertWarns(RuntimeWarning):
                    self.assertEqual(len(search("deepfake", api_key="key", max_results=1, session=self.client(response(data)))), 1)

    def test_scopus_limit_does_not_silently_truncate(self):
        with self.assertRaisesRegex(RuntimeError, "5,000"):
            search_scopus("deepfake", api_key="key", session=self.client(response(scopus_page([1], 5001))))

    def test_repeated_and_premature_empty_pages_fail(self):
        for search, make in [(search_ieee, ieee_page), (search_scopus, scopus_page)]:
            for second in ([1], []):
                with self.subTest(source=search.__name__, second=second):
                    client = self.client(response(make([1], 2)), response(make(second, 2)))
                    with self.assertRaises(RuntimeError):
                        search("deepfake", api_key="key", page_size=1, sleep_seconds=0, session=client)

    def test_bad_payload_is_not_empty_success(self):
        for search in (search_ieee, search_scopus):
            with self.subTest(source=search.__name__), self.assertRaises(RuntimeError):
                search("deepfake", api_key="key", session=self.client(response({"error": "failed"})))

    def test_api_errors_and_network_errors_do_not_expose_keys(self):
        for status in (401, 403, 400):
            client = self.client(response(status=status))
            with self.assertRaises(RuntimeError) as caught:
                search_ieee("deepfake", api_key="secret", session=client)
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(len(client.calls), 1)
        client = Mock()
        client.get.side_effect = requests.ConnectionError("https://example.org?apikey=secret")
        with self.assertRaises(RuntimeError) as caught:
            search_ieee("deepfake", api_key="secret", session=client, max_retries=0)
        self.assertNotIn("secret", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    @patch("sources._api_common.time.sleep")
    def test_retry_after(self, sleep):
        client = self.client(response(status=429, headers={"Retry-After": "2"}), response(ieee_page([], 0)))
        self.assertEqual(search_ieee("deepfake", api_key="key", session=client), [])
        sleep.assert_called_once_with(2)

    @patch("sources._api_common.time.sleep")
    def test_long_retry_after_stops_without_retrying_early(self, sleep):
        client = self.client(response(status=429, headers={"Retry-After": "3600"}))
        with self.assertRaisesRegex(RuntimeError, "rerun later"):
            search_ieee("deepfake", api_key="key", session=client)
        sleep.assert_not_called()

    def test_environment_keys_and_validation(self):
        with patch.dict("os.environ", {"IEEE_API_KEY": "envkey", "SCOPUS_API_KEY": "envkey"}):
            self.assertEqual(search_ieee("deepfake", session=self.client(response(ieee_page([], 0)))), [])
            self.assertEqual(search_scopus("deepfake", session=self.client(response(scopus_page([], 0)))), [])
        for call in (lambda: search_ieee("deep* OR face* OR attack*", api_key="key"),
                     lambda: search_scopus("x", api_key="key", view="COMPLETE", page_size=200),
                     lambda: search_ieee("x", api_key="key", start_year=2026, end_year=2020),
                     lambda: search_ieee("x", api_key="key", max_results=0)):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
