from __future__ import annotations

from typing import Dict

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise ValueError(
                f"Duplicate YAML key {key!r} at line {line}. "
                "Duplicate keys are ambiguous and are not allowed."
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=UniqueKeyLoader)
    return data


def load_config(path: str = "config.yaml") -> Dict:
    return load_yaml(path)


def load_search_terms(path: str = "search_terms.yaml") -> Dict:
    return load_yaml(path)
