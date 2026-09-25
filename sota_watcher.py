from pathlib import Path
import argparse
import json
import sys
import pandas as pd

from sources.discovery import fetch_papers, get_queries_from_search_terms
from sources.restartable import discover_restartable
from utils.config import load_config, load_search_terms
from utils.io import make_output_dirs, load_existing_table, save_table
from utils.deduplication import deduplicate_papers, merge_with_existing, identifiers
from utils.discovery_log import DiscoveryLog
from utils.eligibility import initialize_eligibility
from utils.protocol import validate_protocol
from utils.corpus_store import CorpusStore
from screening import screen_papers
from screening.queue import screen_saved_corpus
from utils.reporting import decorate, review_counts, summary_markdown
from fulltext._common import read_json


def fetch_papers_from_openalex(config, search_terms):
    return fetch_papers({**config, 'sources': ['openalex']}, search_terms)


def filter_papers(papers, config):
    """Compatibility shim: scores and repository type never remove candidates."""
    return papers


def drop_low_score_existing_rows(df, config):
    return df


def run_pipeline(config, search_terms, audit):
    """Legacy single-run workflow retained for compatibility and tests."""
    validate_protocol(search_terms)
    if config.get('screening', {}).get('enabled') and not search_terms.get('eligibility', {}).get('criteria'):
        raise ValueError('Enable full-text screening only after configuring eligibility.criteria.')
    all_papers = fetch_papers(config, search_terms, audit=audit)
    audit.snapshot('discovered', all_papers)
    papers = deduplicate_papers(all_papers)
    audit.snapshot('deduplicated', papers)
    for paper in papers:
        initialize_eligibility(paper)
    papers = screen_papers(papers, search_terms, config, audit)
    existing = load_existing_table(config['sota_table_path'])
    final = merge_with_existing(existing, papers)
    final = final.apply(lambda row: pd.Series(decorate(row.to_dict())), axis=1) if not final.empty else final
    audit.snapshot('merged_table_before_filter', final.to_dict('records'))
    outcomes = read_json(audit.path / 'retrieval_summary.json') or []
    by_identifier = {key: row for row in final.to_dict('records') for key in identifiers(row)}
    for paper in papers:
        old = next((by_identifier[key] for key in identifiers(paper) if key in by_identifier), {})
        for key in ('manual_decision', 'manual_reason', 'notes'):
            if key in old:
                paper[key] = old[key]
        decorate(paper)
    audit.snapshot('screening', papers)
    counts = review_counts(all_papers, papers, outcomes)
    audit.snapshot('review_counts', counts)
    (audit.path / 'review_summary.md').write_text(summary_markdown(papers, counts), encoding='utf-8')
    summary_path = Path(config.get('review_summary_path', str(audit.path / 'table_summary.md')))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_markdown(final.to_dict('records')), encoding='utf-8')
    Path(config['sota_table_path']).parent.mkdir(parents=True, exist_ok=True)
    save_table(final, config['sota_table_path'])
    print(f"Retrieved {len(all_papers)} records; {len(papers)} unique candidates. No score exclusions.")
    print(f"Saved {len(final)} rows to {config['sota_table_path']}")


def _load_runtime():
    from dotenv import load_dotenv
    load_dotenv('.env', override=False)
    config = load_config('config.yaml')
    terms = load_search_terms(config.get('search_terms_path', 'search_terms.yaml'))
    validate_protocol(terms)
    make_output_dirs(config)
    return config, terms


def _store_path(config):
    return config.get(
        'corpus_db_path',
        str(Path(config.get('output_dir', 'output')) / 'sota_corpus.sqlite3'),
    )


def command_import_workbook(args, config):
    path = args.path or config.get('sota_table_path', 'output/sota_table.xlsx')
    with CorpusStore(_store_path(config)) as store:
        count = store.import_workbook(path)
    print(f"Imported {count} workbook rows into the persistent corpus.")


def command_export(args, config):
    path = args.path or config.get('sota_table_path', 'output/sota_table.xlsx')
    with CorpusStore(_store_path(config)) as store:
        # Pull human edits back in before replacing the workbook.
        if Path(path).exists():
            store.import_workbook(path, include_processing=False)
        count = store.export_workbook(path)
    print(f"Exported {count} corpus rows to {path}.")


def command_status(config):
    with CorpusStore(_store_path(config)) as store:
        status = store.status()
        status["discovery"] = store.discovery_status()
    print(json.dumps(status, indent=2, sort_keys=True))


def command_screen(args, config, terms):
    retry = []
    if args.retry:
        retry = [value.strip() for value in args.retry.split(',') if value.strip()]
    limit = args.limit
    if limit is None:
        limit = config.get('screening', {}).get('max_papers_per_run')

    with CorpusStore(_store_path(config)) as store:
        workbook = Path(config.get('sota_table_path', 'output/sota_table.xlsx'))
        if workbook.exists():
            store.import_workbook(workbook, include_processing=False)
        summary = screen_saved_corpus(
            store,
            terms,
            config,
            limit=limit,
            retry=retry,
        )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


def command_discover(args, config, terms):
    with CorpusStore(_store_path(config)) as store:
        workbook = Path(config.get('sota_table_path', 'output/sota_table.xlsx'))
        if workbook.exists() and not store.all_papers():
            store.import_workbook(workbook)
        result = discover_restartable(
            store,
            config,
            terms,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(
        description='Restartable systematic discovery and full-text screening.'
    )
    sub = parser.add_subparsers(dest='command')

    discover = sub.add_parser('discover', help='Run discovery into the persistent corpus.')
    discover.add_argument('--resume', action='store_true', help='Resume matching saved discovery work.')

    screen = sub.add_parser('screen', help='Screen the next pending saved candidates.')
    screen.add_argument('--limit', type=int, default=None)
    screen.add_argument(
        '--retry',
        default='',
        help=(
            'Comma-separated retries: error, fulltext_unavailable, '
            'resolution, download, extraction, screening.'
        ),
    )

    sub.add_parser('status', help='Show persistent corpus processing counts.')

    export = sub.add_parser('export', help='Export persistent corpus to Excel.')
    export.add_argument('path', nargs='?', default=None)

    import_workbook = sub.add_parser(
        'import-workbook',
        help='Import an existing Excel review table without losing human decisions.',
    )
    import_workbook.add_argument('path', nargs='?', default=None)

    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config, terms = _load_runtime()

    if not argv:
        # Preserve the pre-persistent behavior for existing scripts.
        root = config.get(
            'discovery_log_dir',
            str(config.get('output_dir', 'output')) + '/discovery_runs',
        )
        with DiscoveryLog(root) as audit:
            print(f'Discovery archive: {audit.path}')
            audit.snapshot('search_terms', terms)
            run_pipeline(config, terms, audit)
        return

    args = build_parser().parse_args(argv)
    if args.command == 'discover':
        command_discover(args, config, terms)
    elif args.command == 'screen':
        command_screen(args, config, terms)
    elif args.command == 'status':
        command_status(config)
    elif args.command == 'export':
        command_export(args, config)
    elif args.command == 'import-workbook':
        command_import_workbook(args, config)
    else:
        build_parser().print_help()


if __name__ == '__main__':
    main()
