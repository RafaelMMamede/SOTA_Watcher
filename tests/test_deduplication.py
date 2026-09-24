import tempfile
from pathlib import Path
import unittest
import pandas as pd
from utils.deduplication import deduplicate_papers, merge_with_existing
from utils.io import load_existing_table, save_table


class DedupTests(unittest.TestCase):
    def test_bridge_identifiers_and_richer_metadata(self):
        records = [
            {'doi':'HTTPS://DOI.ORG/10.1/ABC', 'source':'ieee','query':'q1','abstract':'Short', 'citation_count':2},
            {'arxiv_id':'2609.10002v1','source':'arxiv','query':'q2','abstract':'A more complete abstract'},
            {'doi':'doi:10.1/abc','arxiv_id':'2609.10002v2','source':'openalex','query':'q3','citation_count':5}]
        merged = deduplicate_papers(records)
        self.assertEqual(len(merged), 1)
        row = merged[0]
        self.assertEqual(row['abstract'], 'A more complete abstract')
        self.assertEqual(row['citation_count'], 5)
        self.assertEqual(row['sources'], ['ieee','arxiv','openalex'])
        self.assertEqual(len(row['provenance']), 3)
        self.assertEqual(len(row['metadata_variants']['abstract']), 2)
        self.assertEqual(records[0]['abstract'], 'Short')

    def test_sparse_nan_and_title_collision(self):
        rows = [{'title':'Same', 'doi':'10.1/a'}, {'title':'Same','doi':'10.1/b'},
                {'doi':float('nan')}, {}, {'title':'Other','year':2025}, {'title':'OTHER','year':2025.0}]
        self.assertEqual(len(deduplicate_papers(rows)), 5)

    def test_excel_roundtrip_preserves_decisions_and_provenance(self):
        old = {'doi':'10.1/a','source':'ieee','query':'one','notes':'My note','manual_decision':'include','abstract':''}
        new = {'doi':'https://doi.org/10.1/a','source':'scopus','query':'two','notes':'','manual_decision':'','abstract':'Full abstract'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'table.xlsx'
            save_table(merge_with_existing(pd.DataFrame([old]), [new]), str(path))
            loaded = load_existing_table(str(path))
            merged = merge_with_existing(loaded, [new]).iloc[0]
            self.assertEqual(merged['notes'], 'My note')
            self.assertEqual(merged['manual_decision'], 'include')
            self.assertEqual(merged['abstract'], 'Full abstract')
            self.assertEqual(merged['sources'], ['ieee','scopus'])
            self.assertEqual(len(merged['provenance']), 2)

    def test_alternate_identifiers_survive_repeat_merge(self):
        first = deduplicate_papers([{'doi':'10.1/a','openalex_id':'W1'}, {'doi':'10.1/a','openalex_id':'W2'}])
        second = deduplicate_papers(first + [{'openalex_id':'https://openalex.org/W2', 'abstract':'New'}])
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]['abstract'], 'New')
