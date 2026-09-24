import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sources.discovery import fetch_papers
from utils.discovery_log import DiscoveryLog
from sota_watcher import run_pipeline


class ArchiveTests(unittest.TestCase):
    def events(self, audit):
        return [json.loads(line) for line in (audit.path / 'events.jsonl').read_text().splitlines()]

    def test_later_failure_preserves_prior_query_and_zero_results(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError), DiscoveryLog(folder) as audit:
                with patch('sources.discovery.search_openalex', autospec=True) as search:
                    search.side_effect = [[{'title':'Saved'}], [], RuntimeError('secret-url')]
                    fetch_papers({}, {'topics': {'t': {'queries': ['a', 'b', 'c']}}}, audit=audit)
            events = self.events(audit)
            successes = [e for e in events if e['event'] == 'query_succeeded']
            self.assertEqual(successes[0]['records'][0]['title'], 'Saved')
            self.assertEqual(successes[1]['retrieved_count'], 0)
            self.assertEqual(events[-1]['event'], 'run_failed')
            self.assertNotIn('secret-url', (audit.path / 'events.jsonl').read_text())

    def test_page_survives_later_page_failure(self):
        def pages(*args, **kwargs):
            yield {'papers':[{'title':'First page'}], 'request_params':{}, 'total_results':2,
                   'complete':False, 'stop_reason':'more_pages'}
            raise RuntimeError('failed page')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError), DiscoveryLog(folder) as audit:
                with patch('sources.discovery.iter_ieee_pages', autospec=True, side_effect=pages):
                    fetch_papers({'sources':['ieee']}, {'topics':{'t':{'queries':['q']}}}, audit=audit)
            event = next(e for e in self.events(audit) if e['event'] == 'query_page')
            self.assertEqual(event['records'][0]['title'], 'First page')
            self.assertFalse(event['complete'])

    def test_excluded_paper_is_archived_before_filtering(self):
        with tempfile.TemporaryDirectory() as folder, DiscoveryLog(folder) as audit:
            with patch('sources.discovery.search_openalex', autospec=True, return_value=[{'title':'Unrelated', 'paper_id':'x'}]):
                run_pipeline({'min_triage_score':100}, {'topics':{'t':{'queries':['q']}}}, audit)
            self.assertEqual(len(json.loads((audit.path / 'discovered.json').read_text())), 1)
            row = json.loads((audit.path / 'screening.json').read_text())[0]
            self.assertEqual(row['screening_reason'], 'below_min_triage_score')
            self.assertNotIn('screening_status', json.loads((audit.path / 'discovered.json').read_text())[0])

    def test_run_directories_are_unique(self):
        with tempfile.TemporaryDirectory() as folder:
            with DiscoveryLog(folder) as first, DiscoveryLog(folder) as second:
                self.assertNotEqual(first.path, second.path)
