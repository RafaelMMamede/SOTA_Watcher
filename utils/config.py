from __future__ import annotations

from typing import Dict

import yaml


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str = "config.yaml") -> Dict:
    return load_yaml(path)


def load_search_terms(path: str = "search_terms.yaml") -> Dict:
    return load_yaml(path)