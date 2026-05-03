"""Local lightweight multi-turn clarification benchmark adapter."""

from __future__ import annotations

from pathlib import Path

from src.data.schema import DialogueExample


def load_clarifybench_local(path: str | Path, limit: int | None = None) -> list[DialogueExample]:
    path = Path(path)
    examples: list[DialogueExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            example = DialogueExample.model_validate_json(line)
            if example.dataset_name == "unknown":
                example.dataset_name = "clarifybench"
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    return examples
