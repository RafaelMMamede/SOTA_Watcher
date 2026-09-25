import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from screening.queue import screen_saved_corpus, select_for_screening
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

    def test_new_search_family_match_requeues_when_criteria_change(self):
        scoped_protocol = {
            'schema_version': 2,
            'searches': [
                {'id': 'visual_forgery_detection', 'queries': {'openalex': ['v']}},
                {'id': 'adversarial_vision', 'queries': {'openalex': ['a']}},
            ],
            'eligibility': {
                'criteria': [
                    {
                        'id': 'visual_scope',
                        'kind': 'inclusion',
                        'description': 'Visual research.',
                    },
                    {
                        'id': 'gan_only',
                        'kind': 'exclusion',
                        'description': 'GAN-only adversarial terminology.',
                        'applies_to_search_topics': ['adversarial_vision'],
                        'skip_if_search_topics': ['visual_forgery_detection'],
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                store.upsert_paper({
                    'paper_id': 'openalex:W1',
                    'openalex_id': 'W1',
                    'title': 'Paper',
                    'year': 2026,
                    'source': 'openalex',
                    'search_topic': 'adversarial_vision',
                })

                def fake_screen(papers, protocol, config, audit=None):
                    papers[0].update({
                        'eligibility_status': 'screened',
                        'eligibility_decision': 'include',
                        'eligibility_reason': 'test',
                        'eligibility_criteria': [],
                        'eligibility_evidence': [],
                        'fulltext_status': 'extracted',
                        'pdf_sha256': 'stable-pdf',
                    })
                    return papers

                cfg = {'screening': {'model': 'qwen3.5:9b'}}
                with patch('screening.queue.get_model_digest', return_value='model-a'), \
                     patch('screening.queue.screen_papers', side_effect=fake_screen):
                    first = screen_saved_corpus(
                        store,
                        scoped_protocol,
                        cfg,
                        limit=1,
                    )
                self.assertEqual(first.completed, 1)

                # Same paper is later discovered through the visual-forgery
                # family. That makes the conditional exclusion inactive.
                store.upsert_paper({
                    'openalex_id': 'W1',
                    'source': 'openalex',
                    'search_topic': 'visual_forgery_detection',
                    'title': 'Paper',
                    'year': 2026,
                })

                with patch('screening.queue.get_model_digest', return_value='model-a'):
                    selected, summary = select_for_screening(
                        store,
                        scoped_protocol,
                        cfg,
                        limit=1,
                    )
                self.assertEqual(len(selected), 1)
                self.assertEqual(summary.stale, 1)

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

    def test_human_workbook_sync_does_not_overwrite_newer_screening(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            workbook = folder / 'review.xlsx'
            import pandas as pd
            from utils.io import save_table

            with CorpusStore(folder / 'corpus.sqlite3') as store:
                corpus_id = store.upsert_paper({
                    'doi': '10.1/current',
                    'title': 'Current paper',
                    'year': 2026,
                })
                current = store.get_paper(corpus_id)
                current.update({
                    'eligibility_status': 'screened',
                    'eligibility_decision': 'include',
                    'eligibility_reason': 'new model result',
                    'fulltext_status': 'extracted',
                    'pdf_sha256': 'new-pdf',
                })
                store.update_processing(
                    corpus_id,
                    current,
                    screening_signature='new-signature',
                )

                save_table(
                    pd.DataFrame([{
                        'corpus_id': corpus_id,
                        'doi': '10.1/current',
                        'title': 'Current paper',
                        'year': 2026,
                        'eligibility_status': 'screened',
                        'eligibility_decision': 'exclude',
                        'eligibility_reason': 'stale workbook result',
                        'pdf_sha256': 'old-pdf',
                        'manual_decision': 'exclude',
                        'manual_reason': 'Human decision',
                        'notes': 'Edited note',
                    }]),
                    str(workbook),
                )

                store.import_workbook(
                    workbook,
                    include_processing=False,
                )
                row = store.get_paper(corpus_id)

                self.assertEqual(row['eligibility_decision'], 'include')
                self.assertEqual(row['pdf_sha256'], 'new-pdf')
                self.assertEqual(row['screening_signature'], 'new-signature')
                self.assertEqual(row['manual_decision'], 'exclude')
                self.assertEqual(row['manual_reason'], 'Human decision')
                self.assertEqual(row['notes'], 'Edited note')

    def test_model_digest_change_requeues_completed_screening(self):
        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                corpus_id = store.upsert_paper({
                    'paper_id': 'openalex:W1',
                    'openalex_id': 'W1',
                    'title': 'Paper',
                    'year': 2026,
                    'source': 'openalex',
                    'search_topic': 'visual_forgery_detection',
                })
                paper = store.get_paper(corpus_id)
                paper.update({
                    'eligibility_status': 'screened',
                    'eligibility_decision': 'include',
                    'eligibility_reason': 'test',
                    'fulltext_status': 'extracted',
                    'pdf_sha256': 'stable-pdf',
                })

                from screening.ollama import screening_signature
                cfg = {'screening': {'model': 'qwen3.5:9b'}}
                old_signature = screening_signature(
                    PROTOCOL['eligibility']['criteria'],
                    cfg['screening'],
                    'stable-pdf',
                    'model-a',
                )
                store.update_processing(
                    corpus_id,
                    paper,
                    screening_signature=old_signature,
                )

                with patch(
                    'screening.queue.get_model_digest',
                    return_value='model-b',
                ):
                    selected, summary = select_for_screening(
                        store,
                        PROTOCOL,
                        cfg,
                        limit=1,
                    )

                self.assertEqual(len(selected), 1)
                self.assertEqual(summary.stale, 1)

    def test_stage_specific_retry_selects_only_matching_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            with CorpusStore(Path(folder) / 'corpus.sqlite3') as store:
                ids = []
                for index, stage in enumerate(('screening', 'extraction')):
                    corpus_id = store.upsert_paper({
                        'paper_id': f'openalex:E{index}',
                        'openalex_id': f'E{index}',
                        'title': f'Error {index}',
                        'year': 2026,
                        'source': 'openalex',
                        'search_topic': 'visual_forgery_detection',
                    })
                    paper = store.get_paper(corpus_id)
                    paper.update({
                        'eligibility_status': 'error',
                        'eligibility_decision': 'uncertain',
                        'eligibility_error_stage': stage,
                        'eligibility_error_type': 'ValueError',
                        'eligibility_error_message': 'test',
                    })
                    store.update_processing(corpus_id, paper)
                    ids.append(corpus_id)

                cfg = {'screening': {'model': 'qwen3.5:9b'}}
                with patch(
                    'screening.queue.get_model_digest',
                    return_value='model-a',
                ):
                    selected, summary = select_for_screening(
                        store,
                        PROTOCOL,
                        cfg,
                        retry=['screening'],
                    )

                self.assertEqual(
                    [paper['corpus_id'] for paper, _, _ in selected],
                    [ids[0]],
                )
                self.assertEqual(summary.skipped_retry_required, 1)

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
