"""Sweep resample_clarify settings on InfoQuest dev and pick the cheapest winning setup."""

from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
from pathlib import Path
from typing import Any

from src.eval.runner import run_experiment
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_csv, write_json


def _teacher_model_id(config: dict[str, Any]) -> str:
    report_cfg = config.get("report", {})
    task_models = report_cfg.get("task_models", [])
    label = report_cfg.get("teacher_model_label")
    for spec in task_models:
        if spec.get("label") == label:
            return spec["base_model"]
    return config["model"]["generator"]["base_model"]


def _flatten_row(payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    targeted = payload["metrics"]["targeted_clarify"]
    resample = payload["metrics"]["resample_clarify"]
    row = {**params}
    row.update(
        {
            "targeted_task_success_rate": targeted["task_success_rate"],
            "targeted_appropriate_action_rate": targeted["appropriate_action_rate"],
            "resample_task_success_rate": resample["task_success_rate"],
            "resample_appropriate_action_rate": resample["appropriate_action_rate"],
            "resample_avg_task_model_calls": resample.get("avg_task_model_calls", 0.0),
            "resample_avg_estimated_cost": resample.get("avg_estimated_cost", 0.0),
            "resample_mean_intent_confidence": resample.get("mean_intent_confidence", 0.0),
            "delta_task_success_rate": resample["task_success_rate"] - targeted["task_success_rate"],
            "delta_appropriate_action_rate": resample["appropriate_action_rate"] - targeted["appropriate_action_rate"],
            "output_dir": payload["output_dir"],
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resample_clarify dev sweep.")
    parser.add_argument("--config", default="configs/report_baselines.yaml")
    parser.add_argument("--output-dir", default="outputs/dev_sweeps")
    parser.add_argument("--dataset", default="infoquest")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["data"]["primary_dataset"] = args.dataset
    config["data"]["split"] = args.split
    config["model"]["generator"]["base_model"] = _teacher_model_id(config)
    config["model"]["generator"]["model_path"] = None

    output_root = ensure_dir(args.output_dir)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    fallback_row: dict[str, Any] | None = None

    for initial_samples, max_rounds, threshold in itertools.product((3, 5, 7), (1, 2), (0.75, 0.80, 0.85)):
        resolved = deepcopy(config)
        resolved["intent_resampling"]["initial_samples"] = initial_samples
        resolved["intent_resampling"]["repair_samples"] = max(1, initial_samples // 2)
        resolved["intent_resampling"]["max_rounds"] = max_rounds
        resolved["intent_resampling"]["clarification_fallback_threshold"] = threshold
        tag = f"k{initial_samples}_r{max_rounds}_c{str(threshold).replace('.', '')}"
        payload = run_experiment(
            config=resolved,
            methods=["targeted_clarify", "resample_clarify"],
            limit=args.limit,
            output_root=output_root / tag,
        )
        row = _flatten_row(
            payload,
            {
                "initial_samples": initial_samples,
                "repair_samples": resolved["intent_resampling"]["repair_samples"],
                "max_rounds": max_rounds,
                "clarification_fallback_threshold": threshold,
            },
        )
        rows.append(row)

        if fallback_row is None:
            fallback_row = row
        else:
            candidate = (
                -row["delta_task_success_rate"],
                -row["delta_appropriate_action_rate"],
                row["resample_avg_estimated_cost"],
            )
            incumbent = (
                -fallback_row["delta_task_success_rate"],
                -fallback_row["delta_appropriate_action_rate"],
                fallback_row["resample_avg_estimated_cost"],
            )
            if candidate < incumbent:
                fallback_row = row

        wins = (
            row["delta_task_success_rate"] >= 0.03
            and row["delta_appropriate_action_rate"] >= 0.0
        )
        if wins:
            if best_row is None:
                best_row = row
            else:
                candidate = (
                    row["resample_avg_estimated_cost"],
                    -row["resample_task_success_rate"],
                    -row["resample_appropriate_action_rate"],
                )
                incumbent = (
                    best_row["resample_avg_estimated_cost"],
                    -best_row["resample_task_success_rate"],
                    -best_row["resample_appropriate_action_rate"],
                )
                if candidate < incumbent:
                    best_row = row

    write_csv(rows, output_root / "sweep_results.csv")
    write_json(rows, output_root / "sweep_results.json")
    if best_row is not None:
        write_json(best_row, output_root / "best_config.json")
        print(f"Saved best sweep row to {output_root / 'best_config.json'}")
    else:
        print("No sweep setting cleared the configured success gate.")
    if fallback_row is not None:
        write_json(fallback_row, output_root / "fallback_config.json")
        print(f"Saved fallback sweep row to {output_root / 'fallback_config.json'}")


if __name__ == "__main__":
    main()
