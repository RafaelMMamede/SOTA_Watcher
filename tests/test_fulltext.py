import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import pymupdf
from fulltext import resolve_paper, fetch_pdf, extract_pdf
from fulltext._common import sha256


class FulltextTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.pdf'
        self.make_pdf('First document')
        self.output = self.root / 'paper'

    def make_pdf(self, text):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), text)
            doc.save(self.source)

    def test_resolution(self):
        r = resolve_paper({'url': 'https://arxiv.org/abs/2609.10002v1'})
        self.assertEqual(r['url'], 'https://arxiv.org/pdf/2609.10002v1')
        self.assertTrue(r['version_pinned'])
        r = resolve_paper({'arxiv_id': 'hep-th/9901001v2'})
        self.assertEqual(r['arxiv_id'], 'hep-th/9901001v2')
        self.assertEqual(resolve_paper({'pdf_url': 'https://publisher.example/file.pdf'})['status'], 'unavailable')
        self.assertEqual(resolve_paper({'arxiv_id': '../../secret'})['status'], 'unavailable')

    def test_local_cache_invalidates_when_source_changes(self):
        resolution = resolve_paper(local_pdf=self.source)
        first = fetch_pdf(resolution, self.output)
        self.assertFalse(first['cache_hit'])
        self.assertTrue(fetch_pdf(resolution, self.output)['cache_hit'])
        self.make_pdf('Changed document')
        changed = fetch_pdf(resolution, self.output)
        self.assertFalse(changed['cache_hit'])
        self.assertNotEqual(first['pdf_sha256'], changed['pdf_sha256'])

    def test_failed_download_preserves_existing_pdf(self):
        fetch_pdf(resolve_paper(local_pdf=self.source), self.output)
        old_hash = sha256(self.output / 'paper.pdf')
        response = Mock(status_code=200, url='https://arxiv.org/pdf/2609.10002v1')
        response.iter_content.return_value = [b'<html>access denied</html>']
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        session = Mock()
        session.get.return_value = response
        with self.assertRaises(ValueError):
            fetch_pdf(resolve_paper({'arxiv_id': '2609.10002v1'}), self.output, session=session)
        self.assertEqual(sha256(self.output / 'paper.pdf'), old_hash)
        self.assertTrue((self.output / 'download_error.json').is_file())

    def test_http_success_and_cache(self):
        resolution = resolve_paper({'arxiv_id': '2609.10002v1'})
        response = Mock(status_code=200, url=resolution['url'])
        response.iter_content.return_value = [self.source.read_bytes()]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        session = Mock()
        session.get.return_value = response
        result = fetch_pdf(resolution, self.output, session=session)
        self.assertEqual(result['pdf_sha256'], sha256(self.source))
        self.assertTrue(fetch_pdf(resolution, self.output, session=session)['cache_hit'])
        self.assertEqual(session.get.call_count, 1)

    def test_size_limit(self):
        with self.assertRaises(ValueError):
            fetch_pdf(resolve_paper(local_pdf=self.source), self.output, max_bytes=10)
        self.assertFalse((self.output / 'paper.pdf').exists())

    def test_extraction_cache_and_markdown_recovery(self):
        fetch_pdf(resolve_paper(local_pdf=self.source), self.output)
        pdf = self.output / 'paper.pdf'
        first = extract_pdf(pdf)
        self.assertEqual(first['page_count'], 1)
        self.assertEqual(first['pages'][0]['page'], 1)
        self.assertIn('First document', first['pages'][0]['text'])
        self.assertTrue(extract_pdf(pdf)['cache_hit'])
        (self.output / 'paper_layout.md').write_text('damaged')
        self.assertFalse(extract_pdf(pdf)['cache_hit'])
        self.assertIn('PDF PAGE 1', (self.output / 'paper_layout.md').read_text())
        self.make_pdf('New PDF content')
        fetch_pdf(resolve_paper(local_pdf=self.source), self.output)
        changed = extract_pdf(pdf)
        self.assertFalse(changed['cache_hit'])
        self.assertIn('New PDF content', changed['pages'][0]['text'])


if __name__ == '__main__':
    unittest.main()
