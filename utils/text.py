# utils/text.py

from __future__ import annotations

import math
from typing import Any


def is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return False


def clean_text(text: Any) -> str:
    if is_missing(text):
        return ""

    text = str(text)

    return " ".join(text.replace("\n", " ").split())


def normalize_key(text: Any) -> str:
    return clean_text(text).lower().strip()