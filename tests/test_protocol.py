import unittest
from utils.protocol import validate_protocol, queries_for
from utils.eligibility import initialize_eligibility, final_decision
from sota_watcher import filter_papers

class ProtocolTests(unittest.TestCase):
    def test_source_queries_and_no_score_exclusion(self):
        p = {'schema_version':2, 'searches':[{'id':'s', 'queries':{'openalex':['q'], 'scopus':[]}}]}
        validate_protocol(p)
        self.assertEqual(queries_for(p, 'openalex'), [('s','q')])
        self.assertEqual(queries_for(p, 'ieee'), [])
        rows = [{'triage_score':0}]
        self.assertEqual(filter_papers(rows, {'min_triage_score':100}), rows)
    def test_invalid_ids(self):
        with self.assertRaises(ValueError):
            validate_protocol({'schema_version':2, 'searches':[{'id':'s','queries':{'typo':['q']}}]})

    def test_criterion_search_topic_scope_is_validated(self):
        protocol={
            'schema_version':2,
            'searches':[
                {'id':'visual','queries':{'openalex':['q']}},
                {'id':'adversarial','queries':{'openalex':['r']}},
            ],
            'eligibility':{
                'criteria':[
                    {
                        'id':'conditional',
                        'kind':'exclusion',
                        'description':'Scoped criterion.',
                        'applies_to_search_topics':['adversarial'],
                    },
                ],
            },
        }
        self.assertIs(validate_protocol(protocol),protocol)
        protocol['eligibility']['criteria'][0][
            'applies_to_search_topics'
        ]=['typo']
        with self.assertRaisesRegex(ValueError,'unknown applies_to_search_topics'):
            validate_protocol(protocol)
    def test_gan_only_exclusion_requires_adversarial_scope(self):
        protocol={
            'schema_version':2,
            'searches':[
                {'id':'visual_forgery_detection','queries':{'openalex':['q']}},
                {'id':'adversarial_vision','queries':{'openalex':['r']}},
            ],
            'eligibility':{
                'criteria':[
                    {
                        'id':'generative_adversarial_only',
                        'kind':'exclusion',
                        'description':'GAN-only false positive.',
                    },
                ],
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            'must declare applies_to_search_topics',
        ):
            validate_protocol(protocol)

    def test_manual_precedence(self):
        row=initialize_eligibility({'manual_decision':'include', 'eligibility_decision':'exclude'})
        self.assertEqual(final_decision(row),'include')
