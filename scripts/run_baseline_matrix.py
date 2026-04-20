"""Run the report-ready model x dataset x method baseline matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval.runner import (
    METRIC_COLUMNS,
    _flatten_metrics,
    build_resources,
    build_run_manifest,
    load_examples,
    run_experiment,
    run_method,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir, markdown_table, write_csv, write_json


def _resolved_output_root(config: dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        return ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(config["paths"]["baseline_dir"]) / f"baseline_{timestamp}")


def _select_repeat_slice(examples: list[Any], fraction: float) -> list[Any]:
    if not examples:
        return []
    target = max(1, round(len(examples) * fraction))
    ordered = sorted(examples, key=lambda example: example.example_id)
    return ordered[:target]


def _compare_metrics(first: dict[str, Any], second: dict[str, Any], tolerance: float) -> dict[str, float]:
    drifts: dict[str, float] = {}
    for key in (
        "task_success_rate",
        "appropriate_action_rate",
        "clarification_precision",
        "clarification_recall",
        "clarification_rate",
        "answer_rate",
        "final_answer_rate",
        "multi_turn_completion_rate",
        "average_answer_score",
        "average_clarification_quality",
        "average_alternatives_quality",
    ):
        if key not in first or key not in second:
            continue
        delta = abs(float(first[key]) - float(second[key]))
        if delta > tolerance:
            drifts[key] = delta
    return drifts


def _run_repeatability_check(
    config: dict[str, Any],
    *,
    output_root: Path,
    dataset_name: str,
    model_label: str,
    methods: list[str],
) -> dict[str, Any]:
    examples = load_examples(config)
    repeat_cfg = config.get("evaluation", {}).get("repeatability", {})
    slice_examples = _select_repeat_slice(
        examples,
        repeat_cfg.get("slice_fraction", 0.1),
    )
    tolerance = float(repeat_cfg.get("tolerance", 0.02))
    enabled_methods = [method for method in methods if method in repeat_cfg.get("methods", methods)]

    runs: list[dict[str, Any]] = []
    for attempt in (1, 2):
        resources = build_resources(config)
        attempt_root = ensure_dir(output_root / f"run_{attempt}")
        attempt_manifest = build_run_manifest(config, slice_examples, enabled_methods)
        attempt_manifest["repeatability_slice_size"] = len(slice_examples)
        attempt_manifest["repeatability_attempt"] = attempt
        write_json(attempt_manifest, attempt_root / "run_manifest.json")

        metrics_by_method: dict[str, Any] = {}
        for method_name in enabled_methods:
            payload = run_method(
                method_name,
                config=config,
                examples=slice_examples,
                resources=resources,
                output_dir=attempt_root,
                nest_method_dir=True,
                run_manifest=attempt_manifest,
            )
            metrics_by_method[method_name] = payload["metrics"]
        runs.append(metrics_by_method)

    drift_report: dict[str, Any] = {}
    for method_name in enabled_methods:
        drift_report[method_name] = _compare_metrics(
            runs[0][method_name],
            runs[1][method_name],
            tolerance=tolerance,
        )
    payload = {
        "dataset_name": dataset_name,
        "model_label": model_label,
        "slice_size": len(slice_examples),
        "tolerance": tolerance,
        "methods": enabled_methods,
        "drift": drift_report,
    }
    write_json(payload, output_root / "repeatability_summary.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline model matrix.")
    parser.add_argument("--config", default="configs/report_baselines.yaml", help="Report config path.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-run example limit for smoke runs.")
    parser.add_argument("--skip-repeatability", action="store_true", help="Skip the repeatability check.")
    args = parser.parse_args()

    config = load_config(args.config)
    report_cfg = config.get("report", {})
    methods = list(report_cfg.get("methods", []))
    datasets = list(report_cfg.get("datasets", []))
    task_models = list(report_cfg.get("task_models", []))
    root = _resolved_output_root(config, args.output_dir)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: list[dict[str, Any]] = []
    repeatability_rows: list[dict[str, Any]] = []

    for dataset_spec in datasets:
        dataset_name = dataset_spec["name"]
        split_name = dataset_spec.get("split")
        for model_spec in task_models:
            resolved = deepcopy(config)
            resolved["data"]["primary_dataset"] = dataset_name
            resolved["data"]["split"] = split_name
            resolved["model"]["generator"]["base_model"] = model_spec["base_model"]
            resolved["model"]["generator"]["model_path"] = None
            resolved["evaluation"]["llm_judge"]["base_model"] = report_cfg["judge_model"]["base_model"]

            leaf_root = ensure_dir(root / dataset_name / model_spec["label"])
            payload = run_experiment(
                config=resolved,
                methods=methods,
                limit=args.limit,
                output_root=leaf_root,
            )

            for method_name, metrics in payload["metrics"].items():
                row = {
                    "dataset": dataset_name,
                    "split": split_name,
                    "model_label": model_spec["label"],
                    "task_model_id": model_spec["base_model"],
                    "judge_model_id": report_cfg["judge_model"]["base_model"],
                    "method": method_name,
                    **_flatten_metrics("", metrics),
                }
                aggregate_rows.append(row)
                aggregate_json.append(row)

            if not args.skip_repeatability:
                repeat_payload = _run_repeatability_check(
                    resolved,
                    output_root=ensure_dir(root / "repeatability" / dataset_name / model_spec["label"]),
                    dataset_name=dataset_name,
                    model_label=model_spec["label"],
                    methods=methods,
                )
                for method_name, drift in repeat_payload["drift"].items():
                    repeatability_rows.append(
                        {
                            "dataset": dataset_name,
                            "model_label": model_spec["label"],
                            "method": method_name,
                            "slice_size": repeat_payload["slice_size"],
                            "drift_count": len(drift),
                            "max_delta": max(drift.values()) if drift else 0.0,
                        }
                    )

    write_json(aggregate_json, root / "matrix_metrics.json")
    write_csv(aggregate_rows, root / "matrix_metrics.csv")
    if repeatability_rows:
        write_csv(repeatability_rows, root / "repeatability.csv")
        write_json(repeatability_rows, root / "repeatability.json")

    headers = ["dataset", "model_label", *METRIC_COLUMNS]
    rows: list[list[str]] = []
    for row in aggregate_rows:
        rows.append(
            [
                str(row["dataset"]),
                str(row["model_label"]),
                str(row["method"]),
                *[
                    f"{row.get(metric, 0.0):.3f}" if isinstance(row.get(metric), float) else str(row.get(metric, ""))
                    for metric in METRIC_COLUMNS[1:]
                ],
            ]
        )
    (root / "comparison_report.md").write_text(markdown_table(headers, rows) + "\n", encoding="utf-8")
    write_json(
        {
            "config_path": args.config,
            "methods": methods,
            "datasets": datasets,
            "task_models": task_models,
            "judge_model": report_cfg.get("judge_model"),
            "output_root": str(root),
        },
        root / "matrix_manifest.json",
    )
    print(f"Completed baseline matrix. Outputs saved to {root}")


if __name__ == "__main__":
    main()
