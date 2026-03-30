"""Filesystem helpers for configs, JSONL artifacts, and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(records: Iterable[Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "model_dump"):
                payload = record.model_dump()
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def append_jsonl(record: Any, path: str | Path) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        if hasattr(record, "model_dump"):
            payload = record.model_dump()
        else:
            payload = record
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)

