import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import pymupdf
from utils.discovery_log import DiscoveryLog
from utils.io import load_existing_table, save_table
from utils.reporting import review_counts, summary_markdown
from sota_watcher import run_pipeline


class ReviewReportingTests(unittest.TestCase):
    def test_counts_distinguish_model_human_and_incomplete_retrieval(self):
        papers=[{'eligibility_decision':'exclude','manual_decision':'include','eligibility_status':'screened'},
                {'eligibility_decision':'uncertain','eligibility_status':'fulltext_unavailable'}]
        counts=review_counts([{}, {}, {}],papers,[{'complete':False}])
        self.assertEqual(counts['duplicate_records_removed'],1)
        self.assertEqual(counts['human_decisions']['include'],1)
        self.assertEqual(counts['model_decisions']['exclude'],1)
        self.assertEqual(counts['awaiting_human_decision'],1)
        self.assertFalse(counts['all_configured_queries_exhausted'])
        self.assertIn('include** (human)',summary_markdown(papers,counts))

    @patch('screening.ollama.requests.get')
    @patch('screening.ollama.requests.post')
    @patch('sources.discovery.iter_openalex_pages', autospec=True)
    def test_discovery_pdf_model_excel_summary_and_resume(self,discover,post,get):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            pdf=folder/'local.pdf'
            doc=pymupdf.open()
            doc.new_page().insert_text((72,72),'We evaluate deepfake detection with visual experiments.')
            doc.save(pdf)
            doc.close()
            records=[{'paper_id':'openalex:W1','doi':'10.1/test','source':'openalex',
                      'title':'Visual experiments','abstract':'Research'},
                     {'paper_id':'openalex:W2','doi':'https://doi.org/10.1/test','source':'openalex',
                      'title':'Visual experiments','abstract':'Longer research abstract'}]
            discover.return_value=[{'papers':records,'complete':True,'stop_reason':'exhausted','total_results':2}]
            get.return_value.json.return_value={'models':[{'name':'qwen3.5:9b','digest':'stable-model'}]}
            def respond(*args,**kwargs):
                page=json.loads(kwargs['json']['messages'][1]['content'])['pdf_pages'][0]
                result={'criteria':[{'id':'visual','assessment':'met','reason':'Explicit experiment.',
                                     'evidence':[{'page':page['page'],'quote':page['text'].strip()}]}]}
                return Mock(json=lambda:{'done':True,'done_reason':'stop','message':{'content':json.dumps(result)}})
            post.side_effect=respond
            cfg={'sources':['openalex'],'sota_table_path':str(folder/'table.xlsx'),
                 'local_pdfs':{'openalex:W1':str(pdf)},
                 'screening':{'enabled':True,'papers_dir':str(folder/'papers')}}
            protocol={'schema_version':2,'searches':[{'id':'visual','queries':{'openalex':['deepfake']}}],
                      'eligibility':{'criteria':[{'id':'visual','kind':'inclusion','description':'Visual experiments.'}]}}
            with DiscoveryLog(folder/'runs') as audit:
                run_pipeline(cfg,protocol,audit)
            counts=json.loads((audit.path/'review_counts.json').read_text())
            self.assertEqual(counts['records_identified'],2)
            self.assertEqual(counts['unique_candidates'],1)
            self.assertEqual(counts['model_decisions']['include'],1)
            table=load_existing_table(cfg['sota_table_path'])
            self.assertEqual(table.iloc[0]['eligibility_evidence'][0]['page'],1)
            table.loc[0,'manual_decision']='exclude'
            table.loc[0,'manual_reason']='Human scope decision'
            save_table(table,cfg['sota_table_path'])
            with DiscoveryLog(folder/'runs') as second:
                run_pipeline(cfg,protocol,second)
            self.assertEqual(post.call_count,1)
            second_counts=json.loads((second.path/'review_counts.json').read_text())
            self.assertEqual(second_counts['human_decisions']['exclude'],1)
            table=load_existing_table(cfg['sota_table_path'])
            self.assertEqual(table.iloc[0]['effective_decision'],'exclude')
            self.assertEqual(table.iloc[0]['decision_origin'],'human')
            self.assertEqual(table.iloc[0]['manual_reason'],'Human scope decision')
            self.assertIn('PDF page 1', (second.path/'review_summary.md').read_text())
