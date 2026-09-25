"""Merge connected stable identifiers; retain metadata conflicts and provenance."""
from __future__ import annotations

import json
import re
from urllib.parse import unquote
import pandas as pd

LIST_FIELDS = ('sources', 'queries', 'search_topics', 'provenance')
STRUCTURED_FIELDS = (*LIST_FIELDS, 'metadata_variants', 'eligibility_evidence', 'eligibility_criteria', 'screening_search_topics', 'screening_inactive_criteria', 'fulltext_manual_candidates', 'fulltext_resolver_attempts')


def present(value):
    if value is None:
        return False
    if isinstance(value, (list, dict)):
        return bool(value)
    return not bool(pd.isna(value)) and str(value).strip() != ''


def decode(value, default):
    if not present(value):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value


def identifiers(paper):
    keys = set()
    # Retain alternate identifiers across subsequent runs and Excel reloads.
    for field, values in decode(paper.get('metadata_variants'), {}).items():
        if field in ('doi', 'arxiv_id', 'openalex_id', 'semantic_scholar_id', 'scopus_id', 'ieee_id', 'paper_id', 'url'):
            for value in values:
                keys |= identifiers({field: value})
    for field in ('doi', 'arxiv_id', 'openalex_id', 'semantic_scholar_id', 'scopus_id', 'ieee_id'):
        if not present(paper.get(field)):
            continue
        value = unquote(str(paper[field])).strip().lower()
        if field == 'doi':
            value = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', value)
        elif field == 'arxiv_id':
            value = re.sub(r'^(?:https?://(?:export\.|www\.)?arxiv.org/(?:abs|pdf)/|arxiv:)', '', value)
            value = re.sub(r'v\d+$', '', value.removesuffix('.pdf'))
        elif field == 'openalex_id':
            value = value.rstrip('/').split('/')[-1]
        keys.add(f'{field}:{value}')
    url = str(paper.get('url') or '') if present(paper.get('url')) else ''
    if re.match(r'https?://(?:dx\.)?doi.org/', url, re.I):
        keys |= identifiers({'doi': url})
    elif re.match(r'https?://(?:export\.|www\.)?arxiv.org/(abs|pdf)/', url, re.I):
        keys |= identifiers({'arxiv_id': url})
    # Stable provider ID helps records with sparse metadata.
    if present(paper.get('paper_id')):
        keys.add('paper_id:' + str(paper['paper_id']).strip().lower())
    if not keys and url:
        keys.add('url:' + url.rstrip('/'))
    if not keys and present(paper.get('title')) and present(paper.get('year')):
        title = re.sub(r'\W+', ' ', str(paper['title']).casefold()).strip()
        keys.add(f'title_year:{title}:{str(paper["year"]).removesuffix(".0")}')
    return keys


def get_dedup_key(paper):
    return next(iter(sorted(identifiers(paper))), '')


def unique(values):
    result = []
    for value in values:
        if present(value) and value not in result:
            result.append(value)
    return result


def merge_group(records):
    merged = {}
    variants = {}
    for paper in records:
        for key, values in decode(paper.get('metadata_variants'), {}).items():
            variants[key] = unique(variants.get(key, []) + values)
        for key, value in paper.items():
            if key in STRUCTURED_FIELDS or not present(value):
                continue
            variants[key] = unique(variants.get(key, []) + [value])
            if not present(merged.get(key)):
                merged[key] = value
            elif key in ('abstract', 'authors', 'title') and len(str(value)) > len(str(merged[key])):
                merged[key] = value
            elif key == 'citation_count':
                try:
                    merged[key] = max(float(value), float(merged[key]))
                except (TypeError, ValueError):
                    pass
    for plural, singular in (('sources', 'source'), ('queries', 'query'), ('search_topics', 'search_topic')):
        merged[plural] = unique([v for p in records for v in [*decode(p.get(plural), []), p.get(singular)]])
    provenance = []
    for paper in records:
        existing = decode(paper.get('provenance'), [])
        provenance.extend(existing or [{k: paper[k] for k in ('source', 'paper_id', 'query', 'search_topic', 'retrieved_at', 'run_id', 'query_id') if present(paper.get(k))}])
    merged['provenance'] = unique(provenance)
    merged['metadata_variants'] = {k: v for k, v in variants.items() if len(v) > 1}
    # Latest screening is an atomic assessment bundle, never mix old evidence
    # with a new decision. Preserve human fields independently.
    latest = next((p for p in reversed(records) if present(p.get('eligibility_status')) and p.get('eligibility_status') not in {'not_screened', 'deferred'}), None)
    if latest is None:
        latest = next((p for p in reversed(records) if present(p.get('eligibility_status'))), None)
    if latest is not None:
        for key in list(merged):
            if key.startswith(('eligibility_', 'screening_', 'fulltext_', 'pdf_')):
                del merged[key]
        merged.update({k:v for k,v in latest.items() if k.startswith(('eligibility_', 'screening_', 'fulltext_', 'pdf_'))})
    merged['has_abstract'] = present(merged.get('abstract'))
    return merged


def deduplicate_papers(papers):
    parents = list(range(len(papers)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, paper in enumerate(papers):
        for key in identifiers(paper):
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, paper in enumerate(papers):
        groups.setdefault(root(i), []).append(paper)
    return [merge_group(records) for records in groups.values()]


def merge_with_existing(existing_df, new_papers):
    # Existing nonempty manual decisions and notes win because old records come first.
    return pd.DataFrame(deduplicate_papers(existing_df.to_dict('records') + new_papers))
