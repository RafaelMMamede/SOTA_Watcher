"""Append-only discovery events and immutable per-run JSON snapshots."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class DiscoveryLog:
    def __init__(self, root):
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:12]
        self.path = Path(root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.event('run_started')

    def event(self, kind, **data):
        with (self.path / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(json_safe({'event': kind, 'at': utc_now(), 'run_id': self.run_id, **data}), ensure_ascii=False, default=str, allow_nan=False) + '\n')
            stream.flush()

    def snapshot(self, name, data):
        with (self.path / f'{name}.json').open('x', encoding='utf-8') as stream:
            json.dump(json_safe(data), stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Exception messages may contain credential-bearing request URLs.
        self.event('run_failed' if exc_type else 'run_finished',
                   **({'error_type': exc_type.__name__} if exc_type else {}))
        return False
