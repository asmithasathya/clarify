"""Config loading, merging, and ablation helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils.io import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def load_config(path: str | Path, method_name: str | None = None) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    raw = read_yaml(config_path)
    merged: dict[str, Any] = {}

    for include in raw.get("includes", []):
        include_path = resolve_repo_path(include)
        merged = deep_merge(merged, read_yaml(include_path))

    merged = deep_merge(merged, {k: v for k, v in raw.items() if k != "includes"})

    if method_name:
        method_path = REPO_ROOT / "configs" / "method" / f"{method_name}.yaml"
        if not method_path.exists():
            raise FileNotFoundError(f"Unknown method config: {method_path}")
        merged = deep_merge(merged, read_yaml(method_path))
    return merged


def apply_ablation(config: dict[str, Any], ablation_name: str) -> dict[str, Any]:
    updated = deepcopy(config)
    ablations = updated.setdefault("ablations", {})

    mapping = {
        "no_ambiguity_detection": ("ambiguity_detection", False),
        "no_intent_modeling": ("intent_modeling", False),
        "no_strategy_selection": ("strategy_selection", False),
        "no_targeted_question": ("targeted_question", False),
    }
    if ablation_name not in mapping:
        raise ValueError(f"Unsupported ablation: {ablation_name}")

    key, value = mapping[ablation_name]
    ablations[key] = value
    return updated

