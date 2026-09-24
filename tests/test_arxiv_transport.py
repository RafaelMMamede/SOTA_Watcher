import unittest
from unittest.mock import Mock, patch
from sources.arxiv_source import _request_oai

class ArxivTransportTests(unittest.TestCase):
    @patch('sources.arxiv_source.time.sleep')
    @patch('sources.arxiv_source.requests.get')
    def test_406_is_not_retried(self, get, sleep):
        get.return_value=Mock(status_code=406)
        with self.assertRaisesRegex(RuntimeError, '406'):
            _request_oai({'verb':'ListRecords'})
        get.assert_called_once()
        sleep.assert_not_called()
    @patch('sources.arxiv_source.time.sleep')
    @patch('sources.arxiv_source.requests.get')
    def test_429_retries(self, get, sleep):
        get.side_effect=[Mock(status_code=429,headers={'Retry-After':'3'}),
                         Mock(status_code=200,content=b'<OAI-PMH/>')]
        _request_oai({'verb':'ListRecords'})
        sleep.assert_called_once_with(3)
