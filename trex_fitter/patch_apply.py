"""Apply the bounded patch format exposed by the coding-agent contract."""

from __future__ import annotations


def apply_patch(source: str, patch: str) -> str:
    """Apply exact-context update hunks to ``analysis.config`` only."""
    lines = patch.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError("invalid patch boundary")
    if len(lines) == 2:
        return source
    if lines[1] != "*** Update File: analysis.config":
        raise ValueError("patch may update only analysis.config")
    source_lines = source.splitlines()
    hunk: list[str] = []

    def apply_one(current: list[str], raw: list[str]) -> list[str]:
        if not raw or any(not line.startswith((" ", "+", "-")) for line in raw):
            raise ValueError("invalid patch hunk")
        old = [line[1:] for line in raw if line.startswith((" ", "-"))]
        new = [line[1:] for line in raw if line.startswith((" ", "+"))]
        matches = [
            index for index in range(len(current) - len(old) + 1)
            if current[index:index + len(old)] == old
        ]
        if len(matches) != 1:
            raise ValueError(f"patch hunk matched {len(matches)} locations")
        index = matches[0]
        return current[:index] + new + current[index + len(old):]

    for line in lines[2:-1]:
        if line == "@@":
            if hunk:
                source_lines = apply_one(source_lines, hunk)
                hunk = []
        else:
            hunk.append(line)
    if hunk:
        source_lines = apply_one(source_lines, hunk)
    return "\n".join(source_lines) + "\n"
