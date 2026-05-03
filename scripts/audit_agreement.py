"""Compute judge-human agreement from a filled audit sheet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _normalize_label(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "correct", "appropriate"}:
        return 1
    if normalized in {"0", "false", "no", "n", "incorrect", "inappropriate"}:
        return 0
    return None


def _cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    observed = sum(1 for left, right in pairs if left == right) / len(pairs)
    left_pos = sum(left for left, _ in pairs) / len(pairs)
    right_pos = sum(right for _, right in pairs) / len(pairs)
    expected = left_pos * right_pos + (1 - left_pos) * (1 - right_pos)
    if expected == 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute audit agreement metrics.")
    parser.add_argument("--audit-sheet", required=True, help="Path to the filled audit_sheet.csv file.")
    parser.add_argument("--audit-key", required=True, help="Path to the audit_key.csv file.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    with Path(args.audit_sheet).open("r", encoding="utf-8", newline="") as handle:
        audit_rows = {row["audit_id"]: row for row in csv.DictReader(handle)}
    with Path(args.audit_key).open("r", encoding="utf-8", newline="") as handle:
        key_rows = {row["audit_id"]: row for row in csv.DictReader(handle)}

    correct_pairs: list[tuple[int, int]] = []
    action_pairs: list[tuple[int, int]] = []
    for audit_id, audit_row in audit_rows.items():
        if audit_id not in key_rows:
            continue
        key_row = key_rows[audit_id]
        human_correct = _normalize_label(audit_row.get("human_correct", ""))
        human_action = _normalize_label(audit_row.get("human_appropriate_action", ""))
        judge_correct = _normalize_label(str(key_row.get("correct", "")))
        judge_action = 1 if key_row.get("response_strategy") in {"ask_clarification", "present_alternatives"} else 0
        if human_correct is not None and judge_correct is not None:
            correct_pairs.append((human_correct, judge_correct))
        if human_action is not None:
            action_pairs.append((human_action, judge_action))

    payload = {
        "n_correct_pairs": len(correct_pairs),
        "n_action_pairs": len(action_pairs),
        "correct_agreement": sum(1 for left, right in correct_pairs if left == right) / len(correct_pairs) if correct_pairs else 0.0,
        "correct_kappa": _cohen_kappa(correct_pairs),
        "action_agreement": sum(1 for left, right in action_pairs if left == right) / len(action_pairs) if action_pairs else 0.0,
        "action_kappa": _cohen_kappa(action_pairs),
    }
    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
