import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from screening.ollama import validate_result, aggregate, parse_structured_content, screen_extraction
from screening.pipeline import screen_papers
from utils.deduplication import merge_with_existing
import pandas as pd

CRITERIA=[{'id':'scope','kind':'inclusion','description':'Visual research.'}]
PAGES=[{'page':1,'text':'We study visual research.'},{'page':2,'text':'The evaluation uses images.'}]


def result(page=1,quote='We study visual research.',assessment='met'):
    return {'criteria':[{'id':'scope','assessment':assessment,'reason':'Scope evidence.',
                         'evidence':[{'page':page,'quote':quote}]}]}


class ScreeningTests(unittest.TestCase):
    def test_reject_hallucinated_quote_page_and_missing_criterion(self):
        for value in (result(9),result(1,'Invented quote'),{'criteria':[]}):
            with self.assertRaises(ValueError):
                validate_result(value,CRITERIA,PAGES)
    def test_structured_parser_accepts_only_plain_or_fenced_json(self):
        payload = result()
        plain = json.dumps(payload)
        fenced = "~~~json\n" + plain + "\n~~~"
        fenced = fenced.replace("~~~", "```")
        self.assertEqual(parse_structured_content(plain), payload)
        self.assertEqual(parse_structured_content(fenced), payload)
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_structured_content("Here is the result: " + plain)

    def test_conflicting_parts_are_uncertain(self):
        rows=[result()['criteria'],result(assessment='not_met')['criteria']]
        self.assertEqual(aggregate(rows,CRITERIA)['eligibility_decision'],'uncertain')
    @patch('screening.ollama.requests.post')
    @patch('screening.ollama.requests.get')
    def test_full_page_coverage_and_cache(self,get,post):
        get.return_value.json.return_value={'models':[{'name':'qwen3.5:9b','digest':'abc'}]}
        post.side_effect=[Mock(json=lambda r=r:{'done':True,'done_reason':'stop','message':{'content':json.dumps(r)}})
                          for r in (result(), result(2,'The evaluation uses images.'))]
        extraction={'status':'extracted','empty_pages':[], 'pages':PAGES,'pdf_sha256':'hash'}
        with tempfile.TemporaryDirectory() as folder:
            first=screen_extraction(extraction,CRITERIA,{},folder)
            second=screen_extraction(extraction,CRITERIA,{},folder)
            self.assertEqual(first['screened_pages'],2)
            self.assertEqual(first['eligibility_decision'],'include')
            self.assertTrue(second['screening_cache_hit'])
            self.assertEqual(post.call_count,2)
    @patch('screening.ollama.requests.post')
    @patch('screening.ollama.requests.get')
    def test_incomplete_generation_reports_budget(self,get,post):
        get.return_value.json.return_value={
            'models':[{'name':'qwen3.5:9b','digest':'abc'}]
        }
        post.return_value=Mock(
            json=lambda:{
                'done':True,
                'done_reason':'length',
                'eval_count':2048,
                'message':{'content':'{'},
            }
        )
        extraction={
            'status':'extracted',
            'empty_pages':[],
            'pages':[PAGES[0]],
            'pdf_sha256':'hash',
        }
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(
                ValueError,
                "done_reason='length'.*num_predict=2048",
            ):
                screen_extraction(
                    extraction,
                    CRITERIA,
                    {'num_predict':2048},
                    folder,
                )

    @patch('screening.ollama.requests.post')
    def test_empty_extraction_never_calls_model(self,post):
        with self.assertRaises(ValueError):
            screen_extraction({'status':'needs_review','empty_pages':[1]},CRITERIA,{},'unused')
        post.assert_not_called()
    def test_missing_pdf_stays_uncertain_and_limit_defers(self):
        with tempfile.TemporaryDirectory() as folder:
            papers=[{'paper_id':'one'},{'paper_id':'two'}]
            screen_papers(papers,{'eligibility':{'criteria':CRITERIA}},
                          {'screening':{'enabled':True,'papers_dir':folder,'max_papers_per_run':1}})
            self.assertEqual(papers[0]['eligibility_status'],'fulltext_unavailable')
            self.assertEqual(papers[0]['eligibility_decision'],'uncertain')
            self.assertEqual(papers[1]['eligibility_status'],'deferred')
    @patch('screening.pipeline.extract_pdf')
    @patch('screening.pipeline.fetch_pdf')
    @patch('screening.pipeline.resolve_paper')
    @patch('screening.pipeline.screen_extraction')
    def test_pipeline_preserves_screening_error_detail(
        self,
        screen,
        resolve,
        fetch,
        extract,
    ):
        resolve.return_value = {
            'status':'resolved',
            'kind':'local',
            'resolver':'local',
            'path':'unused.pdf',
        }
        fetch.return_value = {'status':'downloaded','cache_hit':True}
        extract.return_value = {
            'status':'extracted',
            'empty_pages':[],
            'pages':PAGES,
            'pdf_sha256':'hash',
        }
        screen.side_effect = ValueError(
            'Ollama part 1: structured output validation failed: '
            'Evidence quote/page does not match supplied PDF text.'
        )

        with tempfile.TemporaryDirectory() as folder:
            papers=[{'paper_id':'one'}]
            screen_papers(
                papers,
                {'eligibility':{'criteria':CRITERIA}},
                {
                    'screening':{
                        'enabled':True,
                        'papers_dir':folder,
                        'max_papers_per_run':1,
                    },
                    'local_pdfs':{'one':'unused.pdf'},
                },
            )

        self.assertEqual(papers[0]['eligibility_status'],'error')
        self.assertEqual(papers[0]['eligibility_error_stage'],'screening')
        self.assertIn(
            'Evidence quote/page does not match',
            papers[0]['eligibility_error_message'],
        )
        self.assertIn(
            'during screening',
            papers[0]['eligibility_reason'],
        )

    def test_new_screening_replaces_old_bundle_but_not_human(self):
        old={'doi':'10.1/a','manual_decision':'include','notes':'keep',
             'eligibility_decision':'include','eligibility_status':'screened','eligibility_evidence':[{'page':1}]}
        new={'doi':'10.1/a','eligibility_decision':'uncertain','eligibility_status':'error','eligibility_evidence':[]}
        merged=merge_with_existing(pd.DataFrame([old]),[new]).iloc[0]
        self.assertEqual(merged['manual_decision'],'include')
        self.assertEqual(merged['eligibility_evidence'],[])
        self.assertEqual(merged['eligibility_status'],'error')
