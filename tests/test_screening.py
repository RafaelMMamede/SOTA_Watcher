import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from screening.ollama import _pack_pages, validate_result, aggregate, parse_structured_content, screen_extraction
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
    def test_boundary_ellipses_are_canonicalized_but_internal_omissions_fail(self):
        value=result(quote='...We study visual research....')
        rows=validate_result(value,CRITERIA,[PAGES[0]])
        self.assertEqual(
            rows[0]['evidence'][0]['quote'],
            'We study visual research.',
        )

        with self.assertRaisesRegex(ValueError, 'Evidence quote/page'):
            validate_result(
                result(quote='We study ... research.'),
                CRITERIA,
                [PAGES[0]],
            )

    def test_unique_internal_ellipsis_resolves_to_exact_source_span(self):
        pages=[{
            'page':1,
            'text':(
                'It outlines key deepfake generation models, such as GANs, '
                'autoencoders, neural rendering, and diffusion systems, while '
                'also explaining how adversarial methods enhance realism and '
                'challenge existing detectors.'
            ),
        }]
        value=result(
            quote=(
                'It outlines key deepfake generation models... while also '
                'explaining how adversarial methods enhance realism and '
                'challenge existing detectors.'
            )
        )
        rows=validate_result(value,CRITERIA,pages)
        self.assertEqual(rows[0]['evidence'][0]['quote'],pages[0]['text'])

    def test_internal_ellipsis_must_be_unique_and_bounded(self):
        ambiguous=[{
            'page':1,
            'text':(
                'Alpha evidence phrase middle one omega evidence phrase. '
                'Alpha evidence phrase middle two omega evidence phrase.'
            ),
        }]
        with self.assertRaisesRegex(ValueError, 'Evidence quote/page'):
            validate_result(
                result(
                    quote='Alpha evidence phrase... omega evidence phrase.'
                ),
                CRITERIA,
                ambiguous,
            )

        with self.assertRaisesRegex(ValueError, 'Evidence quote/page'):
            validate_result(
                result(quote='We study ... visual ... research.'),
                CRITERIA,
                [PAGES[0]],
            )

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
    def test_page_packer_combines_pages_and_splits_only_oversized_page(self):
        packed=_pack_pages(PAGES,1000)
        self.assertEqual(len(packed),1)
        self.assertEqual([p['page'] for p in packed[0]],[1,2])

        long_text='abcdefghij' * 100
        split=_pack_pages([{'page':7,'text':long_text}],180)
        self.assertGreater(len(split),1)
        self.assertEqual(
            ''.join(part[0]['text'] for part in split),
            long_text,
        )
        self.assertTrue(all(part[0]['page']==7 for part in split))

    @patch('screening.ollama.requests.post')
    @patch('screening.ollama.requests.get')
    def test_full_page_coverage_and_cache(self,get,post):
        get.return_value.json.return_value={'models':[{'name':'qwen3.5:9b','digest':'abc'}]}
        post.return_value=Mock(
            json=lambda:{
                'done':True,
                'done_reason':'stop',
                'message':{'content':json.dumps(result())},
            }
        )
        extraction={'status':'extracted','empty_pages':[], 'pages':PAGES,'pdf_sha256':'hash'}
        with tempfile.TemporaryDirectory() as folder:
            first=screen_extraction(extraction,CRITERIA,{},folder)
            second=screen_extraction(extraction,CRITERIA,{},folder)
            self.assertEqual(first['screened_pages'],2)
            self.assertEqual(first['screening_parts'],1)
            self.assertEqual(first['eligibility_decision'],'include')
            self.assertTrue(second['screening_cache_hit'])
            self.assertEqual(post.call_count,1)
            sent_pages=post.call_args.kwargs['json']['messages'][1]['content']
            sent_pages=json.loads(sent_pages)['pdf_pages']
            self.assertEqual(sent_pages,PAGES)
    @patch('screening.ollama.requests.post')
    @patch('screening.ollama.requests.get')
    def test_fallback_length_reports_budget_when_split_disabled(self,get,post):
        get.return_value.json.return_value={
            'models':[{'name':'qwen3.5:9b','digest':'abc'}]
        }
        post.side_effect=[
            Mock(json=lambda:{
                'done':True,
                'done_reason':'stop',
                'message':{'content':'not json'},
            }),
            Mock(json=lambda:{
                'done':True,
                'done_reason':'length',
                'eval_count':8192,
                'message':{'content':'{'},
            }),
        ]
        extraction={
            'status':'extracted',
            'empty_pages':[],
            'pages':[PAGES[0]],
            'pdf_sha256':'hash',
        }
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(
                ValueError,
                "Reasoning fallback exhausted num_predict=8192.*done_reason='length'",
            ):
                screen_extraction(
                    extraction,
                    CRITERIA,
                    {'max_split_depth':0},
                    folder,
                )

    @patch('screening.ollama.requests.post')
    @patch('screening.ollama.requests.get')
    def test_fast_primary_then_reasoning_fallback(self,get,post):
        get.return_value.json.return_value={
            'models':[{'name':'qwen3.5:9b','digest':'abc'}]
        }
        post.side_effect=[
            Mock(json=lambda:{
                'done':True,
                'done_reason':'stop',
                'message':{'content':'not json'},
            }),
            Mock(json=lambda:{
                'done':True,
                'done_reason':'stop',
                'eval_count':150,
                'message':{'content':json.dumps(result())},
            }),
        ]
        extraction={
            'status':'extracted',
            'empty_pages':[],
            'pages':[PAGES[0]],
            'pdf_sha256':'hash',
        }
        with tempfile.TemporaryDirectory() as folder:
            screened=screen_extraction(extraction,CRITERIA,{},folder)

        self.assertEqual(screened['eligibility_decision'],'include')
        self.assertEqual(post.call_count,2)

        fast_payload=post.call_args_list[0].kwargs['json']
        self.assertFalse(fast_payload['think'])
        self.assertNotIn('format',fast_payload)
        self.assertEqual(fast_payload['options']['num_predict'],2048)

        fallback_payload=post.call_args_list[1].kwargs['json']
        self.assertEqual(fallback_payload['think'],'low')
        self.assertIn('format',fallback_payload)
        self.assertEqual(fallback_payload['options']['num_predict'],8192)

    @patch('screening.ollama.requests.post')
    @patch('screening.ollama.requests.get')
    def test_bad_fast_evidence_uses_reasoning_fallback(self,get,post):
        get.return_value.json.return_value={
            'models':[{'name':'qwen3.5:9b','digest':'abc'}]
        }
        bad=result(quote='Paraphrased evidence')
        post.side_effect=[
            Mock(json=lambda:{
                'done':True,
                'done_reason':'stop',
                'message':{'content':json.dumps(bad)},
            }),
            Mock(json=lambda:{
                'done':True,
                'done_reason':'stop',
                'message':{'content':json.dumps(result())},
            }),
        ]
        extraction={
            'status':'extracted',
            'empty_pages':[],
            'pages':[PAGES[0]],
            'pdf_sha256':'hash',
        }
        with tempfile.TemporaryDirectory() as folder:
            screened=screen_extraction(extraction,CRITERIA,{},folder)

        self.assertEqual(screened['eligibility_decision'],'include')
        self.assertEqual(post.call_count,2)

    @patch('screening.ollama.requests.post')
    @patch('screening.ollama.requests.get')
    def test_fallback_length_adaptively_splits_multi_page_part(self,get,post):
        get.return_value.json.return_value={
            'models':[{'name':'qwen3.5:9b','digest':'abc'}]
        }
        post.side_effect=[
            Mock(json=lambda:{
                'done':True,
                'done_reason':'stop',
                'message':{'content':'not json'},
            }),
            Mock(json=lambda:{
                'done':True,
                'done_reason':'length',
                'eval_count':8192,
                'message':{'content':'{'},
            }),
            Mock(json=lambda:{
                'done':True,
                'done_reason':'stop',
                'message':{'content':json.dumps(result())},
            }),
            Mock(json=lambda:{
                'done':True,
                'done_reason':'stop',
                'message':{
                    'content':json.dumps(
                        result(2,'The evaluation uses images.')
                    )
                },
            }),
        ]
        extraction={
            'status':'extracted',
            'empty_pages':[],
            'pages':PAGES,
            'pdf_sha256':'hash',
        }
        with tempfile.TemporaryDirectory() as folder:
            screened=screen_extraction(extraction,CRITERIA,{},folder)

        self.assertEqual(screened['eligibility_decision'],'include')
        self.assertEqual(screened['screening_initial_parts'],1)
        self.assertEqual(screened['screening_parts'],2)
        self.assertEqual(screened['screening_adaptive_splits'],1)
        self.assertEqual(post.call_count,4)

        first_child=json.loads(
            post.call_args_list[2].kwargs['json']['messages'][1]['content']
        )['pdf_pages']
        second_child=json.loads(
            post.call_args_list[3].kwargs['json']['messages'][1]['content']
        )['pdf_pages']
        self.assertEqual(first_child,[PAGES[0]])
        self.assertEqual(second_child,[PAGES[1]])

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
