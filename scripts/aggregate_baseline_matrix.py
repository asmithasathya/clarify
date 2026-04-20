"""Aggregate completed leaf outputs into top-level baseline matrix artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.eval.runner import METRIC_COLUMNS, _flatten_metrics
from src.utils.config import load_config
from src.utils.io import markdown_table, read_json, write_csv, write_json


def _resolved_output_root(config: dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(config["paths"]["baseline_dir"])


def aggregate_matrix(config: dict[str, Any], *, output_root: Path) -> dict[str, Any]:
    report_cfg = config.get("report", {})
    datasets = list(report_cfg.get("datasets", []))
    task_models = list(report_cfg.get("task_models", []))

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for dataset_spec in datasets:
        dataset_name = dataset_spec["name"]
        split_name = dataset_spec.get("split")
        for model_spec in task_models:
            leaf_root = output_root / dataset_name / model_spec["label"]
            all_metrics_path = leaf_root / "all_metrics.json"
            if not all_metrics_path.exists():
                reason = "missing_all_metrics"
                sharding_path = leaf_root / "sharding.json"
                if sharding_path.exists():
                    sharding = read_json(sharding_path)
                    reason = f"sharded_{sharding.get('status', 'planned')}"
                incomplete.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "model_label": model_spec["label"],
                        "task_model_id": model_spec["base_model"],
                        "reason": reason,
                    }
                )
                continue

            metrics_by_method = read_json(all_metrics_path)
            for method_name, metrics in metrics_by_method.items():
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

    write_json(aggregate_json, output_root / "matrix_metrics.json")
    write_csv(aggregate_rows, output_root / "matrix_metrics.csv")
    write_json(incomplete, output_root / "matrix_incomplete.json")

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
    (output_root / "comparison_report.md").write_text(markdown_table(headers, rows) + "\n", encoding="utf-8")
    payload = {
        "completed_leaf_count": len({(row["dataset"], row["model_label"]) for row in aggregate_rows}),
        "row_count": len(aggregate_rows),
        "incomplete": incomplete,
    }
    write_json(payload, output_root / "matrix_aggregate_status.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed baseline leaves into top-level matrix artifacts.")
    parser.add_argument("--config", default="configs/report_baselines.yaml", help="Report config path.")
    parser.add_argument("--output-dir", required=True, help="Baseline matrix root directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = aggregate_matrix(config, output_root=_resolved_output_root(config, args.output_dir))
    print(
        "Aggregated baseline matrix: "
        f"{payload['completed_leaf_count']} completed leaves, {len(payload['incomplete'])} incomplete leaves"
    )


if __name__ == "__main__":
    main()
