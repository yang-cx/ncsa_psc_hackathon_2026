"""Shared native TREx config syntax, independent of execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """A native config is malformed or cannot satisfy a requested contract."""


@dataclass
class _Block:
    kind: str
    name: str
    values: dict[str, str]
    line: int


def _without_comment(line: str) -> str:
    quoted = False
    escaped = False
    output: list[str] = []
    for character in line:
        if character == '"' and not escaped:
            quoted = not quoted
        if character == "%" and not quoted:
            break
        output.append(character)
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    return "".join(output).rstrip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def split_top_level(value: str) -> list[str]:
    """Split a TREx comma list, preserving quoted strings and function calls."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quoted = False
    escaped = False
    for character in value:
        if character == '"' and not escaped:
            quoted = not quoted
        elif not quoted:
            if character in "([":
                depth += 1
            elif character in ")]":
                depth -= 1
            elif character == "," and depth == 0:
                parts.append(_unquote("".join(current)))
                current = []
                continue
        current.append(character)
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    parts.append(_unquote("".join(current)))
    return [part.strip() for part in parts if part.strip()]


def _parse_blocks(path: Path) -> list[_Block]:
    blocks: list[_Block] = []
    current: _Block | None = None
    for line_number, source_line in enumerate(path.read_text().splitlines(), 1):
        line = _without_comment(source_line)
        if not line.strip():
            continue
        if ":" not in line:
            raise ConfigError(f"{path}:{line_number}: expected 'key: value'")
        key, value = line.split(":", 1)
        if not source_line[:1].isspace():
            current = _Block(key.strip(), _unquote(value), {}, line_number)
            blocks.append(current)
        elif current is None:
            raise ConfigError(f"{path}:{line_number}: setting appears before a block")
        else:
            if key.strip() in current.values:
                raise ConfigError(
                    f"{path}:{line_number}: duplicate setting {key.strip()!r} "
                    f"in {current.kind} {current.name!r}"
                )
            current.values[key.strip()] = value.strip()
    return blocks
