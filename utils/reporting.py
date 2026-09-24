"""Run-level review counts; model proposals never masquerade as human decisions."""
from collections import Counter
from .eligibility import DECISIONS, final_decision
from .deduplication import present, decode


def decorate(paper):
    manual=paper.get('manual_decision')
    paper['effective_decision']=final_decision(paper)
    paper['decision_origin']='human' if manual in DECISIONS else 'model' if paper.get('eligibility_status')=='screened' else 'unreviewed'
    return paper


def review_counts(raw, unique, outcomes):
    model=Counter(p.get('eligibility_decision','uncertain') for p in unique)
    humans=Counter(p['manual_decision'] for p in unique if p.get('manual_decision') in DECISIONS)
    return {
        'scope':'Current run only; these are paper records, not unique studies.',
        'records_identified':len(raw), 'duplicate_records_removed':len(raw)-len(unique),
        'unique_candidates':len(unique),
        'query_outcomes':outcomes,
        'all_configured_queries_exhausted':bool(outcomes) and all(o.get('complete') is True for o in outcomes),
        'historical_scope_complete':bool(outcomes) and all(o.get('complete') is True and o.get('historical_scope_complete',True) for o in outcomes),
        'fulltext_status_counts':dict(Counter(p.get('fulltext_status','not_requested') for p in unique)),
        'fulltext_resolver_counts':dict(Counter(p.get('fulltext_resolver','unresolved') or 'unresolved' for p in unique)),
        'fulltext_manual_candidate_records':sum(bool(decode(p.get('fulltext_manual_candidates'),[])) for p in unique),
        'screening_status_counts':dict(Counter(p.get('eligibility_status','not_screened') for p in unique)),
        'model_decisions':{d:model[d] for d in DECISIONS},
        'human_decisions':{d:humans[d] for d in DECISIONS},
        'awaiting_human_decision':len(unique)-sum(humans.values()),
        'model_exclusion_reasons':dict(Counter(
            c['id'] for p in unique if p.get('eligibility_decision')=='exclude'
            for c in decode(p.get('eligibility_criteria'),[])
            if (c['kind']=='inclusion' and c['assessment']=='not_met') or
               (c['kind']=='exclusion' and c['assessment']=='met'))),
        'human_exclusion_reasons':dict(Counter(str(p.get('manual_reason') or 'reason_missing')
            for p in unique if p.get('manual_decision')=='exclude')),
    }


def summary_markdown(papers, counts=None):
    lines=['# Systematic review working summary','',
           'Model decisions are provisional. Missing text and errors are not exclusions.','']
    if counts:
        for field in ('records_identified','duplicate_records_removed','unique_candidates',
                      'all_configured_queries_exhausted','historical_scope_complete','awaiting_human_decision'):
            lines.append(f'- {field}: {counts[field]}')
        lines.append('')
    for index,p in enumerate(papers,1):
        decorate(p)
        lines.extend([f"## {index}. {p.get('title') or '[Untitled record]'}",'',
                      f"Decision: **{p['effective_decision']}** ({p['decision_origin']})",'',
                      f"Screening: {p.get('eligibility_status','not_screened')}; full text: {p.get('fulltext_status','not_requested')}",'',
                      f"Full-text resolver: {p.get('fulltext_resolver','') or 'none'}",''])
        for key in ('doi','url','manual_reason','notes','eligibility_reason','screening_artifact'):
            if present(p.get(key)):
                lines.extend([f"{key}: {p[key]}",''])
        for criterion in decode(p.get('eligibility_criteria'),[]):
            lines.extend([f"- {criterion['id']}: {criterion['assessment']} — {criterion['reason']}"])
            for evidence in criterion.get('evidence',[]):
                quote=' '.join(evidence['quote'].split())
                lines.append(f"  - PDF page {evidence['page']}: “{quote}”")
        lines.append('')
    return '\n'.join(lines)
