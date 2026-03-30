"""HousingQA dataset adapters for questions and statute corpora."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from src.data.schema import LegalQAExample, StatutePassage, normalize_answer_label


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    cleaned = str(value).strip()
    return [cleaned] if cleaned else []


def _extract_nested_statute_fields(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    statutes = record.get("statutes")
    if not isinstance(statutes, list):
        return _coerce_list(record.get("statutes")), _coerce_list(record.get("citation")), _coerce_list(record.get("excerpt"))

    statute_ids: list[str] = []
    citations: list[str] = []
    excerpts: list[str] = []
    for item in statutes:
        if isinstance(item, dict):
            statute_idx = item.get("statute_idx") or item.get("idx") or item.get("citation")
            if statute_idx is not None:
                statute_ids.append(str(statute_idx).strip())
            if item.get("citation") is not None:
                citations.append(str(item["citation"]).strip())
            if item.get("excerpt") is not None:
                excerpts.append(str(item["excerpt"]).strip())
        else:
            text = str(item).strip()
            if text:
                statute_ids.append(text)
    if record.get("citation") is not None:
        citations = _coerce_list(record.get("citation"))
    if record.get("excerpt") is not None:
        excerpts = _coerce_list(record.get("excerpt"))
    return statute_ids, citations, excerpts


def _dataset_loader(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def normalize_housingqa_question(record: dict[str, Any], idx: int) -> LegalQAExample:
    statute_ids, citations, excerpts = _extract_nested_statute_fields(record)
    return LegalQAExample(
        example_id=str(record.get("id") or record.get("idx") or record.get("question_id") or f"housingqa-{idx}"),
        question=str(record["question"]).strip(),
        answer=normalize_answer_label(record.get("answer")),
        state=str(record.get("state") or "").strip(),
        statutes=statute_ids,
        citation=citations,
        excerpt=excerpts,
        dataset_name="housingqa",
        metadata={
            key: value
            for key, value in record.items()
            if key not in {"id", "idx", "question_id", "question", "answer", "state", "statutes", "citation", "excerpt"}
        },
    )


def normalize_housingqa_statute(record: dict[str, Any], idx: int) -> StatutePassage:
    doc_id = str(
        record.get("id")
        or record.get("idx")
        or record.get("doc_id")
        or record.get("statute_id")
        or record.get("citation")
        or f"statute-{idx}"
    )
    text = (
        record.get("text")
        or record.get("body")
        or record.get("content")
        or record.get("excerpt")
        or record.get("statute")
        or ""
    )
    return StatutePassage(
        doc_id=doc_id,
        text=str(text).strip(),
        state=(str(record.get("state")).strip() if record.get("state") is not None else None),
        citation=(str(record.get("citation")).strip() if record.get("citation") is not None else None),
        title=(
            str(record.get("title") or record.get("path")).strip()
            if (record.get("title") is not None or record.get("path") is not None)
            else None
        ),
        metadata={
            key: value
            for key, value in record.items()
            if key not in {"id", "idx", "doc_id", "statute_id", "text", "body", "content", "excerpt", "statute", "state", "citation", "title", "path"}
        },
    )


def load_housingqa_questions(
    split: str = "test",
    dataset_name: str = "reglab/housing_qa",
    config_name: str = "questions",
    limit: int | None = None,
    loader: Callable[..., Any] | None = None,
) -> list[LegalQAExample]:
    dataset_loader = loader or _dataset_loader
    dataset = dataset_loader(dataset_name, config_name, split=split)
    records = []
    for idx, record in enumerate(dataset):
        records.append(normalize_housingqa_question(dict(record), idx))
        if limit is not None and len(records) >= limit:
            break
    return records


def iter_housingqa_statutes(
    split: str = "corpus",
    dataset_name: str = "reglab/housing_qa",
    config_name: str = "statutes",
    loader: Callable[..., Any] | None = None,
) -> Iterable[StatutePassage]:
    dataset_loader = loader or _dataset_loader
    dataset = dataset_loader(dataset_name, config_name, split=split)
    for idx, record in enumerate(dataset):
        yield normalize_housingqa_statute(dict(record), idx)


def load_housingqa_statutes(
    split: str = "corpus",
    dataset_name: str = "reglab/housing_qa",
    config_name: str = "statutes",
    limit: int | None = None,
    loader: Callable[..., Any] | None = None,
) -> list[StatutePassage]:
    records = []
    for idx, record in enumerate(
        iter_housingqa_statutes(
            split=split,
            dataset_name=dataset_name,
            config_name=config_name,
            loader=loader,
        )
    ):
        records.append(record)
        if limit is not None and idx + 1 >= limit:
            break
    return records
