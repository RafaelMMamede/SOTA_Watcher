import unittest
from contextlib import ExitStack
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from sources.discovery import fetch_papers
from sources.arxiv_source import search_arxiv
import feedparser


class DiscoveryTests(unittest.TestCase):
    def adapters(self, stack):
        mocks = {}
        for name in ('openalex', 'arxiv', 'ieee', 'scopus'):
            mock = stack.enter_context(patch(f'sources.discovery.iter_{name}_pages', autospec=True))
            mock.return_value = [{'papers':[{'paper_id':name,'title':'Example','query':'q'}], 'complete':True, 'stop_reason':'exhausted', 'total_results':1}]
            mocks[name] = mock
        return mocks

    def test_default_and_disabled_sources(self):
        with ExitStack() as stack:
            mocks = self.adapters(stack)
            rows = fetch_papers({}, {'topics': {'topic': {'queries': ['deepfake']}}})
            self.assertEqual(rows[0]['search_topic'], 'topic')
            mocks['openalex'].assert_called_once_with(query='deepfake', max_results=None)
            for name in ('arxiv', 'ieee', 'scopus'):
                mocks[name].assert_not_called()

    def test_all_sources_options_queries_and_dates(self):
        config = {'sources': ['openalex', 'arxiv', 'ieee', 'scopus'],
                  'max_results_per_query': 10, 'sleep_seconds': 1,
                  'from_publication_date': '2025-06-01',
                  'source_options': {'scopus': {'view': 'STANDARD', 'max_results': 2},
                                     'arxiv': {'native_query': True}}}
        terms = {'topics': {'topic': {'queries': ['deepfake'], 'source_queries': {
            'arxiv': ['ti:deepfake'], 'scopus': ['TITLE-ABS-KEY(deepfake)']}}}}
        with ExitStack() as stack:
            mocks = self.adapters(stack)
            with self.assertWarns(UserWarning):
                rows = fetch_papers(config, terms)
            self.assertEqual(len(rows), 4)
            self.assertEqual(mocks['scopus'].call_args.kwargs['query'], 'TITLE-ABS-KEY(deepfake)')
            self.assertEqual(mocks['scopus'].call_args.kwargs['max_results'], 2)
            self.assertEqual(mocks['ieee'].call_args.kwargs['start_year'], 2025)
            self.assertEqual(mocks['arxiv'].call_args.kwargs['sleep_seconds'], 3)
            self.assertTrue(mocks['arxiv'].call_args.kwargs['native_query'])

    def test_invalid_plan_fails_before_network(self):
        terms = {'topics': {'topic': {'queries': ['deepfake']}}}
        bad_configs = [{'sources': ['typo']}, {'sources': ['openalex', 'scopus']},
                       {'sources': 'arxiv'}, {'sources': ['arxiv', 'arxiv']},
                       {'source_options': {'openalex': {'view': 'STANDARD'}}},
                       {'max_results_per_query': 0}]
        with ExitStack() as stack:
            mocks = self.adapters(stack)
            for config in bad_configs:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    fetch_papers(config, terms)
            for mock in mocks.values():
                mock.assert_not_called()

    def test_empty_override_skips_source(self):
        with ExitStack() as stack:
            mocks = self.adapters(stack)
            fetch_papers({'sources': ['scopus']}, {'topics': {'t': {
                'queries': ['q'], 'source_queries': {'scopus': []}}}})
            mocks['scopus'].assert_not_called()

    def test_failures_propagate(self):
        with ExitStack() as stack:
            mocks = self.adapters(stack)
            mocks['openalex'].side_effect = RuntimeError('Service unavailable')
            with self.assertRaisesRegex(RuntimeError, 'Service unavailable'):
                fetch_papers({}, {'topics': {'t': {'queries': ['q']}}})

