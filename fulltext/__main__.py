"""python -m fulltext --arxiv 2609.10002v1 --output output/papers/2609.10002v1"""
import argparse
from . import resolve_paper, fetch_pdf, extract_pdf


def main():
    parser = argparse.ArgumentParser(description='Fetch/import and extract one paper. No LLM calls.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--arxiv', help='arXiv ID or abs/PDF URL; prefer a versioned ID')
    source.add_argument('--pdf', help='Local PDF path')
    parser.add_argument('--output', required=True, help='Dedicated directory for this paper/version')
    parser.add_argument('--refresh', action='store_true', help='Fetch PDF again')
    parser.add_argument('--force-extract', action='store_true')
    args = parser.parse_args()
    from pathlib import Path
    try:
        resolution = resolve_paper({'arxiv_id': args.arxiv}, local_pdf=args.pdf)
        download = fetch_pdf(resolution, args.output, refresh=args.refresh)
        if download['status'] == 'unavailable':
            parser.exit(2, download['reason'] + '\n')
        result = extract_pdf(Path(args.output) / 'paper.pdf', force=args.force_extract)
    except Exception as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')
    print(f"PDF cached: {download['cache_hit']}; extraction cached: {result['cache_hit']}")
    print(f"Pages: {result['page_count']}; empty pages: {result['empty_pages']}")
    print(f"Saved to {Path(args.output).resolve()}")


if __name__ == '__main__':
    main()
