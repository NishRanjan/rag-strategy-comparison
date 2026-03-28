"""Minimal YAML loader for the config shapes used in this repository."""

from __future__ import annotations

from typing import Any


def _parse_scalar(raw_value: str) -> Any:
    """Parse a scalar YAML value into a Python value."""
    value = raw_value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_block(lines: list[str], start_index: int, indent_level: int) -> tuple[dict[str, Any], int]:
    """Parse an indented mapping block into a dictionary."""
    result: dict[str, Any] = {}
    index = start_index
    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent_level:
            break
        if current_indent > indent_level:
            raise ValueError(f"Unexpected indentation at line: {line}")
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {line}")
        key, value = stripped.split(":", 1)
        if value.strip():
            result[key] = _parse_scalar(value.strip())
            index += 1
            continue
        nested_value, new_index = _parse_block(lines, index + 1, indent_level + 2)
        result[key] = nested_value
        index = new_index
    return result, index


def safe_load(text: str | Any) -> dict[str, Any]:
    """Parse a small YAML mapping document into a Python dictionary."""
    if hasattr(text, "read"):
        text = text.read()
    lines = [line.rstrip("\n") for line in str(text).splitlines() if line.strip() and not line.lstrip().startswith("#")]
    parsed, _ = _parse_block(lines, 0, 0)
    return parsed
