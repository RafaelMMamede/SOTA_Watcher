import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sources.restartable import discover_restartable
from utils.corpus_store import CorpusStore


def protocol(searches):
    return {
        'schema_version': 2,
        'searches': searches,
        'eligibility': {
            'criteria': [
                {
                    'id': 'scope',
                    'kind': 'inclusion',
                    'description': 'Relevant visual research.',
                },
            ],
        },
    }


class RestartableDiscoveryTests(unittest.TestCase):
    def test_interrupted_query_resumes_from_committed_checkpoint(self):
        cfg = {
            'sources': ['openalex'],
            'from_publication_date': '2026-01-01',
            'to_publication_date': '2026-09-25',
        }
        terms = protocol([
            {'id': 'visual', 'queries': {'openalex': ['deepfake']}},
        ])
        resume_values = []

        def interrupted(*, query, resume=None, **kwargs):
            resume_values.append(dict(resume or {}))
            yield {
                'papers': [
                    {
                        'paper_id': 'openalex:W1',
                        'openalex_id': 'W1',
                        'title': 'First',
                        'year': 2026,
                    },
                ],
                'complete': False,
                'stop_reason': 'more_pages',
                'checkpoint': {
                    'cursor': 'next',
                    'retrieved': 1,
                    'total': 2,
                    'seen_ids': ['openalex:W1'],
                    'seen_cursors': ['*', 'next'],
                },
            }
            raise RuntimeError('temporary failure')

        def resumed(*, query, resume=None, **kwargs):
            resume_values.append(dict(resume or {}))
            self.assertEqual(resume['cursor'], 'next')
            yield {
                'papers': [
                    {
                        'paper_id': 'openalex:W2',
                        'openalex_id': 'W2',
                        'title': 'Second',
                        'year': 2026,
                    },
                ],
                'complete': True,
                'stop_reason': 'exhausted',
                'checkpoint': {
                    'cursor': None,
                    'retrieved': 2,
                    'total': 2,
                    'seen_ids': ['openalex:W1', 'openalex:W2'],
                    'seen_cursors': ['*', 'next'],
                },
            }

        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                with patch.dict(
                    'sources.restartable.ITERATORS',
                    {'openalex': interrupted},
                    clear=False,
                ):
                    first = discover_restartable(
                        store, cfg, terms, resume=True
                    )
                self.assertEqual(first['status'], 'incomplete')
                self.assertEqual(len(store.all_papers()), 1)

                with patch.dict(
                    'sources.restartable.ITERATORS',
                    {'openalex': resumed},
                    clear=False,
                ):
                    second = discover_restartable(
                        store, cfg, terms, resume=True
                    )

                self.assertEqual(second['run_id'], first['run_id'])
                self.assertEqual(second['status'], 'complete')
                self.assertEqual(len(store.all_papers()), 2)
                self.assertEqual(resume_values[0], {})
                self.assertEqual(resume_values[1]['cursor'], 'next')

    def test_completed_run_is_reused_with_resume(self):
        cfg = {
            'sources': ['openalex'],
            'from_publication_date': '2026-01-01',
            'to_publication_date': '2026-09-25',
        }
        terms = protocol([
            {'id': 'visual', 'queries': {'openalex': ['deepfake']}},
        ])

        def pages(*, query, resume=None, **kwargs):
            yield {
                'papers': [{
                    'paper_id': 'openalex:W1',
                    'openalex_id': 'W1',
                    'title': 'Saved',
                    'year': 2026,
                }],
                'complete': True,
                'stop_reason': 'exhausted',
                'checkpoint': {'retrieved': 1},
            }

        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                with patch.dict(
                    'sources.restartable.ITERATORS',
                    {'openalex': pages},
                    clear=False,
                ) as patched:
                    first = discover_restartable(
                        store, cfg, terms, resume=True
                    )
                    second = discover_restartable(
                        store, cfg, terms, resume=True
                    )

                self.assertEqual(first['status'], 'complete')
                self.assertEqual(second['status'], 'complete')
                self.assertEqual(second['run_id'], first['run_id'])
                self.assertTrue(second['tasks'][0]['skipped'])
                self.assertEqual(len(store.all_papers()), 1)

    def test_eligibility_change_does_not_restart_completed_discovery(self):
        cfg = {
            'sources': ['openalex'],
            'from_publication_date': '2026-01-01',
            'to_publication_date': '2026-09-25',
        }
        terms = protocol([
            {'id': 'visual', 'queries': {'openalex': ['deepfake']}},
        ])

        def pages(*, query, resume=None, **kwargs):
            yield {
                'papers': [],
                'complete': True,
                'stop_reason': 'exhausted',
                'checkpoint': {'retrieved': 0},
            }

        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                with patch.dict(
                    'sources.restartable.ITERATORS',
                    {'openalex': pages},
                    clear=False,
                ):
                    first = discover_restartable(
                        store, cfg, terms, resume=True
                    )
                    changed = dict(terms)
                    changed['eligibility'] = {
                        'criteria': [{
                            'id': 'different',
                            'kind': 'inclusion',
                            'description': 'Changed screening rule.',
                        }],
                    }
                    second = discover_restartable(
                        store, cfg, changed, resume=True
                    )

                self.assertEqual(second['run_id'], first['run_id'])
                self.assertTrue(second['tasks'][0]['skipped'])

    def test_one_provider_failure_preserves_other_results(self):
        cfg = {
            'sources': ['openalex', 'ieee'],
            'from_publication_date': '2026-01-01',
            'to_publication_date': '2026-09-25',
        }
        terms = protocol([
            {
                'id': 'visual',
                'queries': {
                    'openalex': ['deepfake'],
                    'ieee': ['deepfake'],
                },
            },
        ])

        def good(*, query, resume=None, **kwargs):
            yield {
                'papers': [{
                    'paper_id': 'openalex:W1',
                    'openalex_id': 'W1',
                    'title': 'Saved',
                    'year': 2026,
                }],
                'complete': True,
                'stop_reason': 'exhausted',
                'checkpoint': {'retrieved': 1},
            }

        def bad(*, query, resume=None, **kwargs):
            raise RuntimeError('provider down')
            yield

        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                with patch.dict(
                    'sources.restartable.ITERATORS',
                    {'openalex': good, 'ieee': bad},
                    clear=False,
                ):
                    result = discover_restartable(
                        store, cfg, terms, resume=True
                    )
                self.assertEqual(result['status'], 'incomplete')
                self.assertEqual(len(store.all_papers()), 1)
                statuses = {item['source']: item['status'] for item in result['tasks']}
                self.assertEqual(statuses['openalex'], 'complete')
                self.assertEqual(statuses['ieee'], 'error')

    def test_scopus_offset_limit_is_partitioned(self):
        cfg = {
            'sources': ['scopus'],
            'from_publication_date': '2026-01-01',
            'to_publication_date': '2026-01-04',
            'source_options': {
                'scopus': {'pagination_mode': 'auto'},
            },
        }
        terms = protocol([
            {
                'id': 'visual',
                'queries': {
                    'scopus': ['TITLE-ABS-KEY(deepfake)'],
                },
            },
        ])

        def pages(*, query, resume=None, **kwargs):
            lower = kwargs['from_publication_date']
            upper = kwargs['to_publication_date']
            if lower == '2026-01-01' and upper == '2026-01-04':
                raise RuntimeError(
                    'Scopus: offset fallback exceeds 5,000 source records.'
                )
            yield {
                'papers': [],
                'complete': True,
                'stop_reason': 'exhausted',
                'checkpoint': {
                    'mode': 'offset',
                    'source_retrieved': 0,
                    'exact_retrieved': 0,
                },
            }

        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                with patch.dict(
                    'sources.restartable.ITERATORS',
                    {'scopus': pages},
                    clear=False,
                ):
                    result = discover_restartable(
                        store, cfg, terms, resume=True
                    )
                self.assertEqual(result['status'], 'complete')
                statuses = [item['status'] for item in result['tasks']]
                self.assertIn('partitioned', statuses)
                self.assertEqual(statuses.count('complete'), 2)

    @patch('sources.arxiv_cache.iter_arxiv_harvest_pages')
    def test_multiple_arxiv_queries_share_completed_harvest(self, harvest):
        harvest.return_value = iter([{
            'records': [
                {
                    'paper_id': 'arxiv:2601.00001',
                    'arxiv_id': '2601.00001',
                    'title': 'Deepfake detection',
                    'abstract': 'Adversarial robustness for vision.',
                    'published_date': '2026-01-10',
                    'year': 2026,
                    'source': 'arxiv',
                },
            ],
            'complete': True,
            'stop_reason': 'exhausted',
            'checkpoint': {'resumption_token': '', 'pages': 1},
        }])
        cfg = {
            'sources': ['arxiv'],
            'from_publication_date': '2026-01-01',
            'to_publication_date': '2026-09-25',
            'source_options': {'arxiv': {'sleep_seconds': 3.0}},
        }
        first_terms = protocol([
            {
                'id': 'visual',
                'queries': {
                    'arxiv': ['ti:deepfake', 'abs:"adversarial robustness"'],
                },
            },
        ])

        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                first = discover_restartable(
                    store, cfg, first_terms, resume=False
                )
                self.assertEqual(first['status'], 'complete')
                self.assertEqual(harvest.call_count, 1)

                second_terms = protocol([
                    {
                        'id': 'visual',
                        'queries': {
                            'arxiv': ['all:vision'],
                        },
                    },
                ])
                second = discover_restartable(
                    store, cfg, second_terms, resume=False
                )
                self.assertEqual(second['status'], 'complete')
                self.assertEqual(
                    harvest.call_count,
                    1,
                    'revised query should reuse the completed metadata harvest',
                )


if __name__ == '__main__':
    unittest.main()
