"""InfoQuest dataset adapter.

InfoQuest (bryanlincoln/infoquest on HuggingFace) provides ambiguous seed
messages with hidden context in the form of settings (goal, obstacle,
constraints, checklist).  Each seed message has two plausible settings,
meaning the request is genuinely ambiguous until the assistant discovers
which setting applies.

We flatten each (seed_message, setting) pair into a DialogueExample so
that methods can be evaluated on their ability to detect the ambiguity and
ask targeted clarifying questions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from src.data.schema import DialogueExample


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _setting_to_hidden_context(setting: dict[str, Any]) -> str:
    """Collapse a setting dict into a readable hidden-context string."""
    parts: list[str] = []
    if setting.get("goal"):
        parts.append(f"Goal: {setting['goal']}")
    if setting.get("obstacle"):
        parts.append(f"Obstacle: {setting['obstacle']}")
    constraints = setting.get("constraints") or []
    if constraints:
        parts.append("Constraints: " + "; ".join(str(c) for c in constraints))
    if setting.get("solution"):
        parts.append(f"Solution: {setting['solution']}")
    return "\n".join(parts)


def _setting_to_gold_answer(setting: dict[str, Any]) -> str:
    return setting.get("solution") or setting.get("description") or ""


def normalize_infoquest_example(
    seed: dict[str, Any],
    setting: dict[str, Any],
    setting_index: int,
) -> DialogueExample:
    """Build a DialogueExample from one (seed_message, setting) pair."""
    checklist = setting.get("checklist") or []
    personas: list[str] = []
    for key in ("persona1", "persona2", "persona3"):
        p = seed.get(key)
        if isinstance(p, dict) and p.get("persona"):
            personas.append(str(p["persona"]))
        elif isinstance(p, str) and p:
            personas.append(p)

    return DialogueExample(
        example_id=f"infoquest-{seed.get('id', 0)}-s{setting_index}",
        dataset_name="infoquest",
        user_request=str(seed.get("seed_message", "")).strip(),
        hidden_context=_setting_to_hidden_context(setting),
        gold_clarification_needed=True,  # InfoQuest messages are intentionally ambiguous
        gold_clarifying_question=None,   # dataset does not prescribe a specific question
        gold_answer=_setting_to_gold_answer(setting),
        ambiguity_type="missing_context",
        domain=setting.get("description", "general")[:80] if setting.get("description") else "general",
        checklist=[str(c) for c in checklist],
        personas=personas,
        metadata={
            "seed_id": seed.get("id"),
            "setting_index": setting_index,
            "setting_persona": setting.get("persona"),
        },
    )


# ---------------------------------------------------------------------------
# HuggingFace loader
# ---------------------------------------------------------------------------

def _dataset_loader(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


def _hub_json_loader(filename: str, loader: Callable[..., Any]) -> Any:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="bryanlincoln/infoquest",
        repo_type="dataset",
        filename=filename,
    )
    return loader("json", data_files=path, split="train")


def load_infoquest(
    limit: int | None = None,
    *,
    both_settings: bool = True,
    loader: Callable[..., Any] | None = None,
) -> list[DialogueExample]:
    """Load InfoQuest from HuggingFace and normalise into DialogueExamples.

    Parameters
    ----------
    limit : int | None
        Cap on the number of *examples* returned (not seed messages).
    both_settings : bool
        If True (default), each seed message yields two examples (one per
        setting).  If False, only the first setting is used.
    loader : Callable
        Override for ``datasets.load_dataset`` (useful for testing).
    """
    _load = loader or _dataset_loader
    try:
        seeds_ds = _load("bryanlincoln/infoquest", data_files="seed_messages.jsonl", split="train")
        settings_ds = _load("bryanlincoln/infoquest", data_files="settings.jsonl", split="train")
    except Exception:
        seeds_ds = _hub_json_loader("seed_messages.jsonl", _load)
        settings_ds = _hub_json_loader("settings.jsonl", _load)

    settings_by_id: dict[int, dict[str, Any]] = {}
    for record in settings_ds:
        settings_by_id[record["id"]] = dict(record)

    examples: list[DialogueExample] = []
    for seed_record in seeds_ds:
        seed = dict(seed_record)
        sid = seed.get("id")
        setting_row = settings_by_id.get(sid, {})

        indices = [1, 2] if both_settings else [1]
        for idx in indices:
            setting = setting_row.get(f"setting{idx}")
            if setting is None:
                continue
            if isinstance(setting, str):
                try:
                    setting = json.loads(setting)
                except (json.JSONDecodeError, TypeError):
                    setting = {"description": setting}
            examples.append(normalize_infoquest_example(seed, setting, idx))
            if limit is not None and len(examples) >= limit:
                return examples

    return examples


# ---------------------------------------------------------------------------
# Local-file loader (for cached / offline use)
# ---------------------------------------------------------------------------

def load_infoquest_local(path: str | Path, limit: int | None = None) -> list[DialogueExample]:
    """Load pre-processed InfoQuest examples from a local JSONL file."""
    path = Path(path)
    examples: list[DialogueExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            example = DialogueExample.model_validate_json(line)
            if example.dataset_name == "unknown":
                example.dataset_name = "infoquest"
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    return examples
