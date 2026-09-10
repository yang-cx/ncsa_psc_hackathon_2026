"""Read the pinned native TRExFitter setting schemas without ROOT."""

import math
import re
from functools import lru_cache
from pathlib import Path

from .config_format import _without_comment, _unquote, split_top_level


@lru_cache(maxsize=2)
def load_schema(multifit: bool = False) -> dict[str, dict[str, str]]:
    filename = "multiFitSchema.config" if multifit else "jobSchema.config"
    path = Path(__file__).parent / "schemas" / "v1.10.0" / filename
    schema = {}
    current = None
    for raw in path.read_text().splitlines():
        line = _without_comment(raw)
        if not line.strip():
            continue
        key, value = line.split(":", 1)
        if not raw[:1].isspace():
            current = schema.setdefault(key.strip(), {})
        else:
            # Native ConfigSet::SetConfig uses the last definition. The
            # upstream schema itself contains repeated FriendPath entries.
            current[key.strip()] = value.strip()
    return schema


def matches_schema(raw: str, specification: str) -> bool:
    if not raw.strip():
        return False
    if specification == "string":
        return True
    values = split_top_level(raw)
    for alternative in specification.split("/"):
        types = alternative.split(",")
        if len(types) != len(values):
            continue
        if all(_matches(value, kind.strip()) for value, kind in zip(values, types)):
            return True
    return False


def _matches(value: str, kind: str) -> bool:
    value = _unquote(value)
    if kind == "string":
        return True
    if kind == "int":
        return re.fullmatch(r"[+-]?\d+", value) is not None
    if kind == "float":
        try:
            return math.isfinite(float(value))
        except ValueError:
            return False
    return value.upper() == kind.upper()
