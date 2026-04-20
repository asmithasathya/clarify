"""Evaluate base and distilled students on the frozen test splits."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.eval.runner import _flatten_metrics, run_experiment
from src.utils.config import load_config
from src.utils.io import ensure_dir, markdown_table, write_csv, write_json


def _student_base_model(config: dict[str, Any]) -> str:
    report_cfg = config.get("report", {})
    task_models = report_cfg.get("task_models", [])
    label = report_cfg.get("student_model_label")
    for spec in task_models:
        if spec.get("label") == label:
            return spec["base_model"]
    return config["distillation"].get("student_model", "Qwen/Qwen3-4B-Instruct-2507")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base and distilled student systems.")
    parser.add_argument("--config", default="configs/report_baselines.yaml")
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--dpo-checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/student_eval")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)

    datasets = list(config.get("report", {}).get("datasets", []))
    model_specs = [
        {
            "label": "base_qwen3_4b",
            "base_model": _student_base_model(config),
            "model_path": None,
            "methods": ["targeted_clarify", "resample_clarify"],
        },
        {
            "label": "student_sft",
            "base_model": None,
            "model_path": args.sft_checkpoint,
            "methods": ["targeted_clarify", "resample_clarify"],
        },
    ]
    if args.dpo_checkpoint:
        model_specs.append(
            {
                "label": "student_dpo",
                "base_model": None,
                "model_path": args.dpo_checkpoint,
                "methods": ["targeted_clarify", "resample_clarify"],
            }
        )

    rows: list[dict[str, Any]] = []
    for dataset_spec in datasets:
        for model_spec in model_specs:
            resolved = deepcopy(config)
            resolved["data"]["primary_dataset"] = dataset_spec["name"]
            resolved["data"]["split"] = dataset_spec.get("split")
            resolved["model"]["generator"]["base_model"] = model_spec["base_model"]
            resolved["model"]["generator"]["model_path"] = model_spec["model_path"]
            payload = run_experiment(
                config=resolved,
                methods=model_spec["methods"],
                limit=args.limit,
                output_root=output_dir / dataset_spec["name"] / model_spec["label"],
            )
            for method_name, metrics in payload["metrics"].items():
                rows.append(
                    {
                        "dataset": dataset_spec["name"],
                        "split": dataset_spec.get("split"),
                        "model_label": model_spec["label"],
                        "model_path": model_spec["model_path"],
                        "task_model_id": model_spec["base_model"] or model_spec["model_path"],
                        "method": method_name,
                        **_flatten_metrics("", metrics),
                    }
                )

    write_json(rows, output_dir / "student_eval_metrics.json")
    write_csv(rows, output_dir / "student_eval_metrics.csv")
    table_rows = [
        [
            row["dataset"],
            row["model_label"],
            row["method"],
            f"{float(row.get('task_success_rate', 0.0)):.3f}",
            f"{float(row.get('appropriate_action_rate', 0.0)):.3f}",
            f"{float(row.get('mean_intent_confidence', 0.0)):.3f}",
            f"{float(row.get('avg_task_model_calls', 0.0)):.3f}",
        ]
        for row in rows
    ]
    (Path(output_dir) / "student_eval_report.md").write_text(
        markdown_table(
            ["dataset", "model_label", "method", "task_success_rate", "appropriate_action_rate", "mean_intent_confidence", "avg_task_model_calls"],
            table_rows,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved student evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
