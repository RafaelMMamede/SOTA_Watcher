import unittest
from unittest.mock import Mock, patch

from sources.arxiv_source import _fetch_arxiv_feed


class ArxivTransportTests(unittest.TestCase):
    @patch('sources.arxiv_source.time.sleep')
    @patch('sources.arxiv_source.requests.get')
    def test_rejected_requests_fail_without_retry(self, get, sleep):
        for status in (400, 401, 403, 404, 406):
            with self.subTest(status=status):
                get.reset_mock()
                get.return_value = Mock(status_code=status)
                with self.assertRaisesRegex(RuntimeError, f'HTTP {status}'):
                    _fetch_arxiv_feed('https://export.arxiv.org/api/query')
                get.assert_called_once()
                sleep.assert_not_called()

    @patch('sources.arxiv_source.time.sleep')
    @patch('sources.arxiv_source.requests.get')
    def test_rate_limit_still_retries(self, get, sleep):
        get.side_effect = [Mock(status_code=429, headers={'Retry-After':'3'}),
                           Mock(status_code=200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>')]
        result = _fetch_arxiv_feed('https://export.arxiv.org/api/query')
        self.assertEqual(len(result.entries), 0)
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(3)
