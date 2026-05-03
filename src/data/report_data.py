"""Dataset utilities for frozen report runs."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.data.schema import DialogueExample
from src.utils.io import read_jsonl, write_json, write_jsonl


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_split_path(dataset_cfg: dict[str, Any], split_name: str | None) -> Path | None:
    if split_name:
        split_paths = dataset_cfg.get("split_paths") or dataset_cfg.get("local_splits") or {}
        if split_name in split_paths:
            return Path(split_paths[split_name])
    local_path = dataset_cfg.get("local_path")
    if local_path:
        return Path(local_path)
    return None


def dataset_stats(examples: Iterable[DialogueExample]) -> dict[str, Any]:
    rows = list(examples)
    split_counter = Counter(example.split_name or "unspecified" for example in rows)
    ambiguity_counter = Counter(example.ambiguity_type for example in rows)
    domain_counter = Counter(example.domain for example in rows)
    return {
        "n_examples": len(rows),
        "n_requires_clarification": sum(1 for example in rows if example.gold_clarification_needed),
        "by_split": dict(sorted(split_counter.items())),
        "by_ambiguity_type": dict(sorted(ambiguity_counter.items())),
        "by_domain": dict(sorted(domain_counter.items())),
    }


def validate_examples(
    examples: Iterable[DialogueExample],
    *,
    expected_dataset: str | None = None,
    require_split: bool = False,
) -> dict[str, Any]:
    rows = list(examples)
    ids = [example.example_id for example in rows]
    duplicates = [example_id for example_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate example ids found: {duplicates[:10]}")

    if expected_dataset:
        mismatched = [
            example.example_id
            for example in rows
            if example.dataset_name != expected_dataset
        ]
        if mismatched:
            raise ValueError(
                f"Found {len(mismatched)} examples with dataset_name != {expected_dataset}"
            )

    if require_split:
        missing = [example.example_id for example in rows if not example.split_name]
        if missing:
            raise ValueError(f"Examples missing split_name: {missing[:10]}")

    return dataset_stats(rows)


def split_infoquest_examples(
    examples: Iterable[DialogueExample],
    *,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[DialogueExample]]:
    grouped: dict[int, list[DialogueExample]] = defaultdict(list)
    for example in examples:
        seed_id = int(example.metadata.get("seed_id", -1))
        grouped[seed_id].append(example)

    seed_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(seed_ids)
    n_train = max(1, round(len(seed_ids) * train_fraction))
    n_dev = max(1, round(len(seed_ids) * dev_fraction))
    n_train = min(n_train, len(seed_ids) - 2)
    n_dev = min(n_dev, len(seed_ids) - n_train - 1)
    train_ids = set(seed_ids[:n_train])
    dev_ids = set(seed_ids[n_train : n_train + n_dev])

    splits = {"train": [], "dev": [], "test": []}
    for seed_id in sorted(grouped):
        split_name = "test"
        if seed_id in train_ids:
            split_name = "train"
        elif seed_id in dev_ids:
            split_name = "dev"
        for example in grouped[seed_id]:
            payload = example.model_copy(deep=True)
            payload.split_name = split_name
            splits[split_name].append(payload)
    return splits


def load_examples_jsonl(path: str | Path) -> list[DialogueExample]:
    records = read_jsonl(path)
    return [DialogueExample.model_validate(record) for record in records]


def write_dataset_split(
    examples: Iterable[DialogueExample],
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    require_split: bool = False,
) -> dict[str, Any]:
    rows = list(examples)
    stats = validate_examples(
        rows,
        expected_dataset=expected_dataset,
        require_split=require_split,
    )
    write_jsonl(rows, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "stats": stats,
    }


def write_manifest(path: str | Path, *, dataset_name: str, splits: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "dataset_name": dataset_name,
        "splits": splits,
    }
    if extra:
        payload.update(extra)
    write_json(payload, path)
