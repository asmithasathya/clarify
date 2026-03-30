"""Optional adapter for user-provided local legal QA datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.schema import LegalQAExample, normalize_answer_label
from src.utils.io import read_csv, read_jsonl


REQUIRED_COLUMNS = {"question", "answer"}


def _coerce_optional_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def _coerce_local_record(record: dict[str, Any], idx: int) -> LegalQAExample:
    return LegalQAExample(
        example_id=str(record.get("example_id") or record.get("id") or f"legalqa-local-{idx}"),
        question=str(record["question"]).strip(),
        answer=normalize_answer_label(record.get("answer")),
        state=str(record.get("state") or "unknown").strip(),
        statutes=_coerce_optional_list(record.get("statutes")),
        citation=_coerce_optional_list(record.get("citation")),
        excerpt=_coerce_optional_list(record.get("excerpt")),
        dataset_name="legalqa_local",
        metadata={
            key: value
            for key, value in record.items()
            if key not in {"example_id", "id", "question", "answer", "state", "statutes", "citation", "excerpt"}
        },
    )


def load_local_legalqa(path: str | Path) -> list[LegalQAExample]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Local legal QA file not found: {source}")

    if source.suffix.lower() == ".jsonl":
        rows = read_jsonl(source)
    elif source.suffix.lower() == ".csv":
        rows = read_csv(source)
    else:
        raise ValueError("Only JSONL and CSV inputs are supported for legalqa_local.")

    missing = REQUIRED_COLUMNS - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Missing required columns for local legal QA data: {sorted(missing)}")

    return [_coerce_local_record(record, idx) for idx, record in enumerate(rows)]
