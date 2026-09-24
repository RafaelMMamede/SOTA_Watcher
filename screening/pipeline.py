"""Resumable PDF retrieval and provisional eligibility for every candidate."""
import hashlib
from pathlib import Path
import time
from fulltext import resolve_paper, fetch_pdf, extract_pdf
from utils.deduplication import get_dedup_key
from utils.eligibility import initialize_eligibility
from fulltext._common import write_json
from .ollama import screen_extraction


def screen_papers(papers, protocol, config, audit=None):
    cfg=config.get('screening',{})
    criteria=protocol.get('eligibility',{}).get('criteria',[])
    enabled=cfg.get('enabled',False)
    limit=cfg.get('max_papers_per_run')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('screening.max_papers_per_run must be positive or null.')
    for index,paper in enumerate(papers):
        initialize_eligibility(paper)
        if not enabled or (limit is not None and index >= limit):
            paper['eligibility_status']='not_screened' if not enabled else 'deferred'
            paper['eligibility_reason']='Screening disabled.' if not enabled else 'Per-run screening limit reached.'
            continue
        key=get_dedup_key(paper) or paper.get('title') or str(index)
        folder=Path(cfg.get('papers_dir','output/papers'))/hashlib.sha256(key.encode()).hexdigest()[:24]
        paper['fulltext_folder']=str(folder)
        try:
            local=config.get('local_pdfs',{}).get(paper.get('paper_id')) or config.get('local_pdfs',{}).get(paper.get('doi'))
            resolution=resolve_paper(paper,local_pdf=local)
            download=fetch_pdf(resolution,folder,refresh=cfg.get('refresh_pdf',False))
            paper['fulltext_status']=download['status']
            if download['status'] != 'downloaded':
                paper['eligibility_status']='fulltext_unavailable'
                paper['eligibility_reason']=download.get('reason','Full text unavailable.')
            else:
                if resolution['kind']=='arxiv' and not download.get('cache_hit'):
                    time.sleep(3)
                extraction=extract_pdf(folder/'paper.pdf',folder)
                paper['pdf_sha256']=extraction['pdf_sha256']
                paper['fulltext_status']=extraction['status']
                paper.update(screen_extraction(extraction,criteria,cfg,folder))
        except Exception as exc:
            paper['eligibility_decision']='uncertain'
            paper['eligibility_status']='error'
            paper['eligibility_reason']=f'{type(exc).__name__}: full-text retrieval/extraction/screening failed; inspect artifacts and retry.'
            paper['eligibility_error_type']=type(exc).__name__
        folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'latest_eligibility.json',{k:v for k,v in paper.items() if k.startswith(('eligibility_','fulltext_','screening_','pdf_'))})
        if audit:
            audit.event('eligibility_result',paper_id=paper.get('paper_id'),
                        decision=paper['eligibility_decision'], status=paper['eligibility_status'], folder=str(folder))
    return papers
