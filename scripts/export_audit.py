"""Export a stratified manual-audit pack from baseline predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_prediction_rows(matrix_root: Path) -> list[dict]:
    rows: list[dict] = []
    for predictions_path in sorted(matrix_root.rglob("predictions.jsonl")):
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                payload["_source_path"] = str(predictions_path)
                rows.append(payload)
    return rows


def _behavior_bucket(row: dict) -> str:
    return "clarification" if row.get("asked_clarification") else "answer"


def _stratified_sample(rows: list[dict], target_size: int) -> list[dict]:
    groups: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset_name", "unknown"),
            row.get("task_model_id", "unknown"),
            row.get("method", "unknown"),
            "correct" if row.get("correct") else "incorrect",
            _behavior_bucket(row),
        )
        groups[key].append(row)

    for group in groups.values():
        group.sort(key=lambda row: row["example_id"])

    selected: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    ordered_keys = sorted(groups)
    while len(selected) < min(target_size, len(rows)):
        progress = False
        for key in ordered_keys:
            group = groups[key]
            while group:
                candidate = group.pop(0)
                dedupe_key = (
                    candidate.get("dataset_name", ""),
                    candidate.get("example_id", ""),
                    candidate.get("method", ""),
                )
                if dedupe_key in seen:
                    continue
                selected.append(candidate)
                seen.add(dedupe_key)
                progress = True
                break
            if len(selected) >= target_size:
                break
        if not progress:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a manual audit pack from matrix outputs.")
    parser.add_argument("--matrix-root", required=True, help="Root directory produced by run_baseline_matrix.py.")
    parser.add_argument("--target-size", type=int, default=100, help="Number of examples to export.")
    parser.add_argument("--output-dir", default=None, help="Directory to write audit files.")
    args = parser.parse_args()

    matrix_root = Path(args.matrix_root)
    output_dir = Path(args.output_dir) if args.output_dir else matrix_root / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_prediction_rows(matrix_root)
    sample = _stratified_sample(rows, args.target_size)

    blinded_fields = [
        "audit_id",
        "example_id",
        "dataset_name",
        "split_name",
        "user_request",
        "response_text",
        "simulated_user_reply",
        "final_answer",
        "human_correct",
        "human_appropriate_action",
        "adjudication_notes",
    ]
    key_fields = [
        "audit_id",
        "dataset_name",
        "split_name",
        "task_model_id",
        "judge_model_id",
        "method",
        "correct",
        "asked_clarification",
        "answered_directly",
        "response_strategy",
        "source_path",
    ]

    blinded_rows: list[dict] = []
    key_rows: list[dict] = []
    for idx, row in enumerate(sample, start=1):
        audit_id = f"audit-{idx:03d}"
        blinded_rows.append(
            {
                "audit_id": audit_id,
                "example_id": row.get("example_id"),
                "dataset_name": row.get("dataset_name"),
                "split_name": row.get("split_name"),
                "user_request": row.get("user_request"),
                "response_text": row.get("response_text"),
                "simulated_user_reply": row.get("simulated_user_reply"),
                "final_answer": row.get("final_answer"),
                "human_correct": "",
                "human_appropriate_action": "",
                "adjudication_notes": "",
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "dataset_name": row.get("dataset_name"),
                "split_name": row.get("split_name"),
                "task_model_id": row.get("task_model_id"),
                "judge_model_id": row.get("judge_model_id"),
                "method": row.get("method"),
                "correct": row.get("correct"),
                "asked_clarification": row.get("asked_clarification"),
                "answered_directly": row.get("answered_directly"),
                "response_strategy": row.get("response_strategy"),
                "source_path": row.get("_source_path"),
            }
        )

    with (output_dir / "audit_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=blinded_fields)
        writer.writeheader()
        writer.writerows(blinded_rows)

    with (output_dir / "audit_key.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_fields)
        writer.writeheader()
        writer.writerows(key_rows)

    with (output_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "matrix_root": str(matrix_root),
                "target_size": args.target_size,
                "exported_size": len(sample),
            },
            handle,
            indent=2,
        )
    print(f"Exported {len(sample)} audit examples to {output_dir}")


if __name__ == "__main__":
    main()
