"""Model-neutral schema checks for coding-agent supervised trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
TRAINING_ONLY_FIELDS = {
    "assistant_loss_mask",
    "assistant_masks",
    "chat_template_kwargs",
    "input_ids",
    "labels",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def row_id(row: dict[str, Any], index: int | None = None) -> str:
    value = row.get("uuid") or row.get("id")
    if isinstance(value, str) and value:
        return value
    return f"row[{index}]" if index is not None else "row"


def require_string(value: Any, location: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "string" if allow_empty else "non-empty string"
        raise ValueError(f"{location} must be a {qualifier}")


def validate_record(row: dict[str, Any], index: int | None = None) -> None:
    identifier = row_id(row, index)
    leaked = TRAINING_ONLY_FIELDS & row.keys()
    if leaked:
        raise ValueError(f"{identifier}: training-only fields leaked: {sorted(leaked)}")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{identifier}: messages must contain at least two turns")
    tools = row.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError(f"{identifier}: tools must be a list")
    declared: set[str] = set()
    for tool_index, tool in enumerate(tools):
        location = f"{identifier}.tools[{tool_index}]"
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"{location}: expected a function tool")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"{location}.function must be an object")
        require_string(function.get("name"), f"{location}.function.name")
        if not isinstance(function.get("parameters"), dict):
            raise ValueError(f"{location}.function.parameters must be JSON Schema")
        if function["name"] in declared:
            raise ValueError(f"{identifier}: duplicate tool {function['name']!r}")
        declared.add(function["name"])

    pending: set[str] = set()
    called: set[str] = set()
    result_ids: set[str] = set()
    assistant_turns = 0
    for message_index, message in enumerate(messages):
        location = f"{identifier}.messages[{message_index}]"
        if not isinstance(message, dict) or message.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"{location}: invalid message or role")
        role = message["role"]
        require_string(message.get("content", ""), f"{location}.content", allow_empty=True)
        if role == "system" and message_index != 0:
            raise ValueError(f"{location}: system role is allowed only first")
        if pending and role != "tool":
            raise ValueError(f"{location}: unresolved calls before the next turn: {sorted(pending)}")
        if role == "assistant":
            assistant_turns += 1
            content = message.get("content", "")
            if "reasoning_content" in message or "<think>" in content or "</think>" in content:
                raise ValueError(f"{location}: hidden or explicit thinking is forbidden")
        calls = message.get("tool_calls") or []
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                raise ValueError(f"{location}: tool_calls require an assistant list")
            for call_index, call in enumerate(calls):
                call_location = f"{location}.tool_calls[{call_index}]"
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError(f"{call_location}: expected a function call")
                call_id = call.get("id")
                require_string(call_id, f"{call_location}.id")
                if call_id in pending or call_id in result_ids:
                    raise ValueError(f"{identifier}: duplicate call ID {call_id!r}")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError(f"{call_location}.function must be an object")
                require_string(function.get("name"), f"{call_location}.function.name")
                if not isinstance(function.get("arguments"), dict):
                    raise ValueError(f"{call_location}.function.arguments must be an object")
                pending.add(call_id)
                called.add(function["name"])
        if role == "tool":
            call_id = message.get("tool_call_id")
            require_string(call_id, f"{location}.tool_call_id")
            if call_id not in pending:
                raise ValueError(f"{location}: result has no pending call")
            pending.remove(call_id)
            result_ids.add(call_id)

    if pending:
        raise ValueError(f"{identifier}: calls lack results: {sorted(pending)}")
    if called - declared:
        raise ValueError(f"{identifier}: undeclared tools called: {sorted(called - declared)}")
    if assistant_turns == 0:
        raise ValueError(f"{identifier}: no assistant turns")
    final = messages[-1]
    if final.get("role") != "assistant" or final.get("tool_calls") or not final.get("content"):
        raise ValueError(f"{identifier}: trajectory must end with a non-empty assistant response")


def validate_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    checked = list(rows)
    if not checked:
        raise ValueError("dataset is empty")
    identifiers: set[str] = set()
    for index, row in enumerate(checked):
        validate_record(row, index)
        identifier = row_id(row, index)
        if identifier in identifiers:
            raise ValueError(f"duplicate record ID: {identifier}")
        identifiers.add(identifier)
    return checked
