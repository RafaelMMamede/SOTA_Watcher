import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from screening.queue import screen_saved_corpus
from utils.corpus_store import CorpusStore


PROTOCOL = {
    'schema_version': 2,
    'searches': [
        {'id': 'visual_forgery_detection', 'queries': {'openalex': ['q']}},
    ],
    'eligibility': {
        'criteria': [
            {
                'id': 'visual_scope',
                'kind': 'inclusion',
                'description': 'Visual research.',
            },
        ],
    },
}


class CorpusStoreTests(unittest.TestCase):
    def test_two_screening_batches_advance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'corpus.sqlite3'
            with CorpusStore(path) as store:
                for index in range(4):
                    store.upsert_paper({
                        'paper_id': f'openalex:W{index}',
                        'openalex_id': f'W{index}',
                        'title': f'Paper {index}',
                        'year': 2026,
                        'source': 'openalex',
                        'search_topic': 'visual_forgery_detection',
                    })

                def fake_screen(papers, protocol, config, audit=None):
                    paper = papers[0]
                    paper.update({
                        'eligibility_status': 'screened',
                        'eligibility_decision': 'include',
                        'eligibility_reason': 'test',
                        'eligibility_criteria': [],
                        'eligibility_evidence': [],
                        'fulltext_status': 'extracted',
                        'pdf_sha256': 'pdf-' + paper['corpus_id'],
                    })
                    return papers

                cfg = {
                    'screening': {
                        'model': 'qwen3.5:9b',
                        'papers_dir': str(Path(folder) / 'papers'),
                    },
                }
                with patch('screening.queue.get_model_digest', return_value='model-a'),                      patch('screening.queue.screen_papers', side_effect=fake_screen):
                    first = screen_saved_corpus(store, PROTOCOL, cfg, limit=2)
                    second = screen_saved_corpus(store, PROTOCOL, cfg, limit=2)
                    third = screen_saved_corpus(store, PROTOCOL, cfg, limit=2)

                self.assertEqual(first.selected, 2)
                self.assertEqual(first.completed, 2)
                self.assertEqual(second.selected, 2)
                self.assertEqual(second.completed, 2)
                self.assertEqual(third.selected, 0)
                self.assertEqual(
                    sum(
                        p.get('eligibility_status') == 'screened'
                        for p in store.all_papers()
                    ),
                    4,
                )

    def test_workbook_import_preserves_human_review(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'review.xlsx'
            target = folder / 'roundtrip.xlsx'

            import pandas as pd
            from utils.io import save_table, load_existing_table

            save_table(
                pd.DataFrame([{
                    'doi': '10.1/example',
                    'title': 'Reviewed paper',
                    'year': 2026,
                    'manual_decision': 'exclude',
                    'manual_reason': 'Human scope decision',
                    'notes': 'Keep this note',
                }]),
                str(source),
            )

            with CorpusStore(folder / 'corpus.sqlite3') as store:
                self.assertEqual(store.import_workbook(source), 1)
                store.upsert_paper({
                    'doi': 'https://doi.org/10.1/example',
                    'title': 'Reviewed paper',
                    'year': 2026,
                    'source': 'scopus',
                    'search_topic': 'visual_forgery_detection',
                })
                self.assertEqual(store.export_workbook(target), 1)

            row = load_existing_table(str(target)).iloc[0]
            self.assertEqual(row['manual_decision'], 'exclude')
            self.assertEqual(row['manual_reason'], 'Human scope decision')
            self.assertEqual(row['notes'], 'Keep this note')

    def test_later_identifier_enrichment_keeps_stable_corpus_id(self):
        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                first = store.upsert_paper({
                    'paper_id': 'scopus:1',
                    'scopus_id': '1',
                    'title': 'Same paper',
                    'year': 2026,
                })
                enriched = store.upsert_paper({
                    'scopus_id': '1',
                    'doi': '10.1/same',
                    'title': 'Same paper',
                    'year': 2026,
                })
                by_doi = store.upsert_paper({
                    'doi': 'https://doi.org/10.1/same',
                    'title': 'Same paper',
                    'year': 2026,
                })

                self.assertEqual(first, enriched)
                self.assertEqual(first, by_doi)
                self.assertEqual(len(store.all_papers()), 1)


if __name__ == '__main__':
    unittest.main()
