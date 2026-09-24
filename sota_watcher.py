from pathlib import Path

from sources.discovery import fetch_papers, get_queries_from_search_terms
from utils.config import load_config, load_search_terms
from utils.io import make_output_dirs, load_existing_table, save_table
from utils.deduplication import deduplicate_papers, merge_with_existing
from utils.discovery_log import DiscoveryLog
from utils.eligibility import initialize_eligibility
from utils.protocol import validate_protocol
from screening import screen_papers


def fetch_papers_from_openalex(config, search_terms):
    return fetch_papers({**config, 'sources': ['openalex']}, search_terms)


def filter_papers(papers, config):
    """Compatibility shim: scores and repository type never remove candidates."""
    return papers


def drop_low_score_existing_rows(df, config):
    return df


def run_pipeline(config, search_terms, audit):
    validate_protocol(search_terms)
    all_papers = fetch_papers(config, search_terms, audit=audit)
    audit.snapshot('discovered', all_papers)
    papers = deduplicate_papers(all_papers)
    audit.snapshot('deduplicated', papers)
    for paper in papers:
        initialize_eligibility(paper)
    papers = screen_papers(papers, search_terms, config, audit)
    audit.snapshot('screening', papers)
    existing = load_existing_table(config['sota_table_path'])
    final = merge_with_existing(existing, papers)
    audit.snapshot('merged_table_before_filter', final.to_dict('records'))
    Path(config['sota_table_path']).parent.mkdir(parents=True, exist_ok=True)
    save_table(final, config['sota_table_path'])
    print(f"Retrieved {len(all_papers)} records; {len(papers)} unique candidates. No score exclusions.")
    print(f"Saved {len(final)} rows to {config['sota_table_path']}")


def main():
    from dotenv import load_dotenv
    load_dotenv('.env', override=False)
    config = load_config('config.yaml')
    terms = load_search_terms(config.get('search_terms_path', 'search_terms.yaml'))
    validate_protocol(terms)
    make_output_dirs(config)
    root = config.get('discovery_log_dir', str(config.get('output_dir', 'output')) + '/discovery_runs')
    with DiscoveryLog(root) as audit:
        print(f'Discovery archive: {audit.path}')
        audit.snapshot('search_terms', terms)
        run_pipeline(config, terms, audit)


if __name__ == '__main__':
    main()
