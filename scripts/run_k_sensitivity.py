"""Run matched K-sensitivity experiments for resample_clarify."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.eval.runner import run_experiment
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_csv, write_json


def _resolve_task_model(config: dict[str, Any], model_label: str) -> str:
    for spec in config.get("report", {}).get("task_models", []):
        if spec.get("label") == model_label:
            return spec["base_model"]
    raise ValueError(f"Unknown report task model label: {model_label}")


def _row(output_dir: Path, params: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **params,
        "task_success_rate": metrics.get("task_success_rate", 0.0),
        "appropriate_action_rate": metrics.get("appropriate_action_rate", 0.0),
        "average_answer_score": metrics.get("average_answer_score", 0.0),
        "average_clarification_quality": metrics.get("average_clarification_quality", 0.0),
        "mean_intent_confidence": metrics.get("mean_intent_confidence", 0.0),
        "resample_rate": metrics.get("resample_rate", 0.0),
        "avg_task_model_calls": metrics.get("avg_task_model_calls", 0.0),
        "avg_latency": metrics.get("avg_latency", 0.0),
        "n_examples": metrics.get("n_examples", 0),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-sensitivity for the intent-stabilization method.")
    parser.add_argument("--config", default="configs/report_baselines.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="infoquest")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model-label", default="qwen3_30b")
    parser.add_argument("--k", action="append", type=int, required=True, help="Initial sample K. Repeatable.")
    parser.add_argument("--repair-samples", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--fallback-threshold", type=float, default=0.80)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base = load_config(args.config)
    base["data"]["primary_dataset"] = args.dataset
    base["data"]["split"] = args.split
    base["model"]["generator"]["base_model"] = _resolve_task_model(base, args.model_label)
    base["model"]["generator"]["model_path"] = None
    judge = base.get("report", {}).get("judge_model", {}).get("base_model")
    if judge:
        base["evaluation"]["llm_judge"]["base_model"] = judge

    root = ensure_dir(args.output_dir)
    rows: list[dict[str, Any]] = []
    for k in sorted(set(args.k)):
        resolved = deepcopy(base)
        resolved["intent_resampling"]["initial_samples"] = k
        resolved["intent_resampling"]["repair_samples"] = args.repair_samples
        resolved["intent_resampling"]["max_rounds"] = args.max_rounds
        resolved["intent_resampling"]["clarification_fallback_threshold"] = args.fallback_threshold
        tag = f"k{k}_r{args.max_rounds}_repair{args.repair_samples}_c{str(args.fallback_threshold).replace('.', '')}"
        output_dir = root / tag
        payload = run_experiment(
            config=resolved,
            methods=["resample_clarify"],
            limit=args.limit,
            output_root=output_dir,
        )
        metrics = payload["metrics"]["resample_clarify"]
        rows.append(
            _row(
                output_dir,
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "model_label": args.model_label,
                    "initial_samples": k,
                    "repair_samples": args.repair_samples,
                    "max_rounds": args.max_rounds,
                    "clarification_fallback_threshold": args.fallback_threshold,
                },
                metrics,
            )
        )

    write_csv(rows, root / "k_sensitivity.csv")
    write_json(rows, root / "k_sensitivity.json")
    print(f"Wrote K-sensitivity summary to {root / 'k_sensitivity.json'}")


if __name__ == "__main__":
    main()
