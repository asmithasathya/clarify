"""Update manuscript sensitivity tables from K-sensitivity artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir


def _load_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "k_sensitivity.json"
    if path.exists():
        return json.loads(path.read_text())

    rows: list[dict[str, Any]] = []
    for child in sorted(root.glob("*/k_sensitivity.json")):
        rows.extend(json.loads(child.read_text()))
    if not rows:
        raise FileNotFoundError(f"Missing sensitivity summary under: {root}")
    return rows


def _prediction_path(row: dict[str, Any]) -> Path:
    return Path(row["output_dir"]) / "resample_clarify" / "predictions.jsonl"


def _reliability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        path = _prediction_path(row)
        if not path.exists():
            continue
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            pred = json.loads(line)
            confidence = float(pred.get("intent_confidence", 0.0))
            lower = int(confidence * 10) / 10
            label = f"{lower:.1f}--{lower + 0.1:.1f}"
            buckets[label].append(pred)
        for label, preds in sorted(buckets.items()):
            output.append(
                {
                    "k": row["initial_samples"],
                    "bin": label,
                    "n": len(preds),
                    "mean_confidence": sum(float(p.get("intent_confidence", 0.0)) for p in preds) / len(preds),
                    "task_success_rate": sum(1 for p in preds if p.get("correct")) / len(preds),
                }
            )
    return output


def _write_k_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched K-sensitivity for all-stage stabilization.}",
        r"\label{tab:dev-sensitivity}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$K$ & Repair & Rounds & TSR & AAR & Conf. & Calls \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=lambda r: int(r["initial_samples"])):
        lines.append(
            f"{row['initial_samples']} & {row['repair_samples']} & {row['max_rounds']} & "
            f"{row['task_success_rate']:.3f} & {row['appropriate_action_rate']:.3f} & "
            f"{row['mean_intent_confidence']:.3f} & {row['avg_task_model_calls']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output_dir / "dev_sensitivity.tex").write_text("\n".join(lines))


def _write_reliability_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    reliability = _reliability_rows(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-hoc reliability of intent-confidence scores in the K-sensitivity runs.}",
        r"\label{tab:confidence-reliability}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$K$ & Confidence bin & N & Mean conf. & Task success \\",
        r"\midrule",
    ]
    for row in reliability:
        lines.append(
            f"{row['k']} & {row['bin']} & {row['n']} & "
            f"{row['mean_confidence']:.3f} & {row['task_success_rate']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output_dir / "confidence_reliability.tex").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Update paper sensitivity tables from run artifacts.")
    parser.add_argument("--sensitivity-root", required=True)
    parser.add_argument("--output-dir", default="paper/generated")
    args = parser.parse_args()

    rows = _load_rows(Path(args.sensitivity_root))
    output_dir = ensure_dir(args.output_dir)
    _write_k_table(rows, output_dir)
    _write_reliability_table(rows, output_dir)
    generated_from = output_dir / "generated_from.txt"
    lines = []
    if generated_from.exists():
        lines = [
            line
            for line in generated_from.read_text().splitlines()
            if not line.startswith("k_sensitivity_root=")
        ]
    lines.append(f"k_sensitivity_root={Path(args.sensitivity_root)}")
    generated_from.write_text("\n".join(lines) + "\n")
    print(f"Updated sensitivity tables in {output_dir}")


if __name__ == "__main__":
    main()
