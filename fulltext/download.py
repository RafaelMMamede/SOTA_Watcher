"""Validate and cache PDFs; failed fetches never replace a valid cached file."""
from __future__ import annotations

import ipaddress
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests

from ._common import now, read_json, sha256, validate_pdf, write_json


TRUSTED_REMOTE_RESOLVERS = {
    "openalex",
    "unpaywall",
    "semantic_scholar",
}


def _safe_https_url(value):
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
        ):
            return None
    return parsed.geturl()


def _validated_remote_url(resolution):
    if resolution.get("kind") != "remote":
        return None
    if resolution.get("resolver") not in TRUSTED_REMOTE_RESOLVERS:
        return None
    if resolution.get("is_oa") is not True:
        return None
    return _safe_https_url(resolution.get("url"))


def fetch_pdf(
    resolution,
    output_dir,
    *,
    refresh=False,
    timeout=60,
    max_bytes=100 * 1024 * 1024,
    session=None,
):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "download.json"

    if resolution.get("status") != "resolved":
        record = {**resolution, "checked_at": now()}
        write_json(folder / "resolution.json", record)
        return record

    if resolution.get("kind") not in {"local", "arxiv", "remote"}:
        raise ValueError("Unsupported resolution kind.")
    if max_bytes <= 0 or timeout <= 0:
        raise ValueError("max_bytes and timeout must be positive.")

    pdf = folder / "paper.pdf"
    previous = read_json(manifest)

    if (
        not refresh
        and previous
        and previous.get("resolution") == resolution
        and pdf.is_file()
    ):
        local_unchanged = (
            resolution["kind"] != "local"
            or sha256(resolution["path"]) == previous.get("pdf_sha256")
        )
        if local_unchanged and sha256(pdf) == previous.get("pdf_sha256"):
            validate_pdf(pdf)
            return {**previous, "cache_hit": True}

    fd, temporary = tempfile.mkstemp(dir=folder, suffix=".pdf")
    client = session or requests.Session()

    try:
        with os.fdopen(fd, "wb") as target:
            if resolution["kind"] == "local":
                with open(resolution["path"], "rb") as source:
                    _copy(
                        iter(lambda: source.read(1024 * 1024), b""),
                        target,
                        max_bytes,
                    )
                final_url = None

            else:
                if resolution["kind"] == "arxiv":
                    from .resolve import arxiv_id

                    identifier = arxiv_id(resolution.get("arxiv_id"))
                    expected = (
                        f"https://arxiv.org/pdf/{identifier}"
                        if identifier
                        else None
                    )
                    url = (
                        resolution.get("url")
                        if expected and resolution.get("url") == expected
                        else None
                    )
                    if not url:
                        raise ValueError("Invalid arXiv resolution.")
                else:
                    url = _validated_remote_url(resolution)
                    if not url:
                        raise ValueError(
                            "Remote PDF must come from a trusted OA resolver."
                        )

                with client.get(
                    url,
                    stream=True,
                    timeout=timeout,
                    headers={
                        "User-Agent": "SOTA-Watcher/0.1",
                        "Accept": "application/pdf",
                    },
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"PDF retrieval failed: HTTP {response.status_code}. "
                            "Retry later if transient."
                        )
                    final_url = _safe_https_url(response.url)
                    if not final_url:
                        raise ValueError(
                            "PDF resolver redirected to an unsafe/non-HTTPS URL."
                        )
                    _copy(
                        response.iter_content(1024 * 1024),
                        target,
                        max_bytes,
                    )

        count = validate_pdf(temporary)
        record = {
            "status": "downloaded",
            "resolution": resolution,
            "final_url": final_url,
            "retrieved_at": now(),
            "pdf_sha256": sha256(temporary),
            "bytes": Path(temporary).stat().st_size,
            "page_count": count,
        }
        os.replace(temporary, pdf)
        write_json(manifest, record)
        return {**record, "cache_hit": False}

    except Exception as exc:
        write_json(
            folder / "download_error.json",
            {
                "status": "failed",
                "at": now(),
                "error_type": type(exc).__name__,
                "resolution": resolution,
            },
        )
        raise

    finally:
        Path(temporary).unlink(missing_ok=True)
        if session is None:
            client.close()


def _copy(chunks, target, limit):
    size = 0
    for chunk in chunks:
        size += len(chunk)
        if size > limit:
            raise ValueError("PDF exceeds configured size limit.")
        target.write(chunk)
