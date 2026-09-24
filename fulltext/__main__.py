"""Standalone full-text resolver/downloader; never invokes an LLM."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from . import resolve_paper, fetch_pdf, extract_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Resolve, fetch/import, and extract one paper. No LLM calls."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--arxiv",
        help="arXiv ID or abs/PDF URL; prefer a versioned ID",
    )
    source.add_argument(
        "--doi",
        help="DOI or https://doi.org/... identifier",
    )
    source.add_argument(
        "--openalex",
        help="OpenAlex work ID or URL",
    )
    source.add_argument(
        "--title",
        help="Exact paper title for conservative OpenAlex fallback",
    )
    source.add_argument("--pdf", help="Local PDF path")

    parser.add_argument("--year", type=int, help="Publication year for --title")
    parser.add_argument(
        "--authors",
        help="Author string used to validate --title matches",
    )
    parser.add_argument(
        "--email",
        help="Contact email for resolver APIs; otherwise UNPAYWALL_EMAIL is used",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Dedicated directory for this paper/version",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Print resolver result without downloading/extracting",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch PDF again",
    )
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env", override=False)

    paper = {}
    if args.arxiv:
        paper["arxiv_id"] = args.arxiv
    if args.doi:
        paper["doi"] = args.doi
    if args.openalex:
        paper["openalex_id"] = args.openalex
    if args.title:
        paper["title"] = args.title
        if args.year:
            paper["year"] = args.year
        if args.authors:
            paper["authors"] = args.authors

    try:
        resolution = resolve_paper(
            paper,
            local_pdf=args.pdf,
            resolver_config={"email": args.email} if args.email else {},
        )

        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolution.json").write_text(
            json.dumps(resolution, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(resolution, ensure_ascii=False, indent=2))

        if args.resolve_only:
            raise SystemExit(0 if resolution["status"] == "resolved" else 2)

        download = fetch_pdf(
            resolution,
            output,
            refresh=args.refresh,
        )
        if download["status"] == "unavailable":
            parser.exit(2, download["reason"] + "\n")

        result = extract_pdf(
            output / "paper.pdf",
            force=args.force_extract,
        )

    except SystemExit:
        raise
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")

    print(
        f"PDF cached: {download['cache_hit']}; "
        f"extraction cached: {result['cache_hit']}"
    )
    print(
        f"Pages: {result['page_count']}; "
        f"empty pages: {result['empty_pages']}"
    )
    print(f"Saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
