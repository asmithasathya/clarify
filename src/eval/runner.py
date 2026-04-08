"""Experiment runner: load data, run methods, compute metrics, write artifacts."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm.auto import tqdm

from src.data.infoquest import load_infoquest, load_infoquest_local
from src.data.schema import DialogueExample, MethodResult
from src.eval.metrics import compute_all_metrics
from src.llm.generator import build_generator
from src.methods.direct_answer import run_direct_answer
from src.methods.generic_clarify import run_generic_clarify
from src.methods.generic_hedge import run_generic_hedge
from src.methods.targeted_clarify import run_targeted_clarify
from src.understand.ambiguity_detector import AmbiguityDetector
from src.understand.clarification_generator import ClarificationGenerator
from src.understand.intent_model import IntentModeler
from src.understand.strategy_selector import StrategySelector
from src.utils.io import ensure_dir, markdown_table, write_csv, write_json, write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


METHOD_REGISTRY: dict[str, Callable[..., MethodResult]] = {
    "direct_answer": run_direct_answer,
    "generic_hedge": run_generic_hedge,
    "generic_clarify": run_generic_clarify,
    "targeted_clarify": run_targeted_clarify,
}

METRIC_COLUMNS = [
    "method",
    "task_success_rate",
    "appropriate_action_rate",
    "clarification_precision",
    "clarification_recall",
    "clarification_rate",
    "answer_rate",
    "missed_ambiguity_rate",
    "unnecessary_clarification_rate",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_examples(config: dict[str, Any], limit: int | None = None) -> list[DialogueExample]:
    dataset_name = config.get("data", {}).get("primary_dataset", "infoquest")
    if dataset_name == "infoquest":
        iq_cfg = config.get("data", {}).get("infoquest", {})
        local_path = iq_cfg.get("local_path")
        if local_path:
            return load_infoquest_local(local_path, limit=limit)
        return load_infoquest(
            limit=limit,
            both_settings=iq_cfg.get("both_settings", True),
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Resource construction
# ---------------------------------------------------------------------------

def build_resources(config: dict[str, Any]) -> dict[str, Any]:
    generator = build_generator(config)
    return {
        "generator": generator,
        "ambiguity_detector": AmbiguityDetector(config, generator=generator),
        "intent_modeler": IntentModeler(config, generator=generator),
        "strategy_selector": StrategySelector(config, generator=generator),
        "clarification_generator": ClarificationGenerator(config, generator=generator),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(new_key, value))
        elif isinstance(value, list):
            continue
        else:
            flat[new_key] = value
    return flat


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(re.findall(r"[A-Za-z0-9]+", text.lower()))


def _text_matches(
    predicted: str | None,
    gold: str | None,
    *,
    overlap_threshold: float = 0.6,
) -> bool:
    predicted_norm = _normalize_text(predicted)
    gold_norm = _normalize_text(gold)
    if not predicted_norm or not gold_norm:
        return False
    if predicted_norm == gold_norm or predicted_norm in gold_norm or gold_norm in predicted_norm:
        return True

    predicted_tokens = set(predicted_norm.split())
    gold_tokens = set(gold_norm.split())
    if not predicted_tokens or not gold_tokens:
        return False

    recall = len(predicted_tokens & gold_tokens) / len(gold_tokens)
    return recall >= overlap_threshold


def _evaluate_result(
    result: MethodResult,
    example: DialogueExample,
    config: dict[str, Any],
) -> MethodResult:
    """Populate single-turn correctness signals before metric aggregation."""
    overlap_threshold = config.get("evaluation", {}).get("answer_match_threshold", 0.6)
    evaluation: dict[str, Any] = {"overlap_threshold": overlap_threshold}

    if result.final_answer:
        result.correct = _text_matches(
            result.final_answer,
            example.gold_answer,
            overlap_threshold=overlap_threshold,
        )
        evaluation["mode"] = "answer_match"
        evaluation["matched_text"] = result.correct
    elif result.clarification_question:
        if example.gold_clarifying_question:
            result.correct = _text_matches(
                result.clarification_question,
                example.gold_clarifying_question,
                overlap_threshold=overlap_threshold,
            )
            evaluation["mode"] = "clarification_match"
            evaluation["matched_text"] = result.correct
        else:
            result.correct = example.gold_clarification_needed
            evaluation["mode"] = "clarification_proxy"
    elif result.response_strategy == "present_alternatives":
        result.correct = example.gold_clarification_needed and bool(result.response_text.strip())
        evaluation["mode"] = "alternatives_proxy"
    else:
        result.correct = False
        evaluation["mode"] = "no_final_answer"

    result.trace["evaluation"] = evaluation
    return result


def _write_method_outputs(
    method_name: str,
    results: Sequence[MethodResult],
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> None:
    method_dir = ensure_dir(Path(output_dir) / method_name)
    write_jsonl(results, method_dir / "predictions.jsonl")
    write_json(metrics, method_dir / "metrics.json")
    write_csv([_flatten_metrics("", metrics)], method_dir / "summary.csv")

    row = [method_name]
    for col in METRIC_COLUMNS[1:]:
        val = metrics.get(col, "")
        row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
    report = markdown_table(METRIC_COLUMNS, [row])
    (method_dir / "report_snippet.md").write_text(report + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Run one method
# ---------------------------------------------------------------------------

def run_method(
    method_name: str,
    *,
    config: dict[str, Any],
    examples: Sequence[DialogueExample],
    resources: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    results: list[MethodResult] = []
    iterator = tqdm(examples, desc=method_name, leave=False)

    for example in iterator:
        if method_name == "direct_answer":
            result = run_direct_answer(
                example, generator=resources["generator"], config=config,
            )
        elif method_name == "generic_hedge":
            result = run_generic_hedge(
                example, generator=resources["generator"], config=config,
            )
        elif method_name == "generic_clarify":
            result = run_generic_clarify(
                example, generator=resources["generator"], config=config,
            )
        elif method_name == "targeted_clarify":
            result = run_targeted_clarify(
                example,
                config=config,
                ambiguity_detector=resources["ambiguity_detector"],
                intent_modeler=resources["intent_modeler"],
                strategy_selector=resources["strategy_selector"],
                clarification_generator=resources["clarification_generator"],
            )
        else:
            raise ValueError(f"Unsupported method: {method_name}")

        results.append(_evaluate_result(result, example, config))

    metrics = compute_all_metrics(results)
    _write_method_outputs(method_name, results, metrics, output_dir)
    return {"results": results, "metrics": metrics}


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    config: dict[str, Any],
    methods: Sequence[str],
    limit: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(config.get("project", {}).get("seed", 42))
    examples = load_examples(config, limit=limit)
    resources = build_resources(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = output_root or Path(config["paths"]["outputs_dir"]) / f"run_{timestamp}"
    output_dir = ensure_dir(output_root)
    write_json(config, Path(output_dir) / "resolved_config.json")

    run_summary: dict[str, Any] = {}
    summary_rows: list[list[str]] = []

    for method_name in methods:
        LOGGER.info("Running method: %s", method_name)
        payload = run_method(
            method_name,
            config=config,
            examples=examples,
            resources=resources,
            output_dir=output_dir,
        )
        run_summary[method_name] = payload["metrics"]
        m = payload["metrics"]
        row = [method_name]
        for col in METRIC_COLUMNS[1:]:
            val = m.get(col, "")
            row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        summary_rows.append(row)

    comparison_table = markdown_table(METRIC_COLUMNS, summary_rows)
    (Path(output_dir) / "comparison_report.md").write_text(comparison_table + "\n", encoding="utf-8")
    write_json(run_summary, Path(output_dir) / "all_metrics.json")
    write_csv(
        [
            {"method": method_name, **_flatten_metrics("", metrics)}
            for method_name, metrics in run_summary.items()
        ],
        Path(output_dir) / "all_metrics.csv",
    )
    return {"output_dir": str(output_dir), "metrics": run_summary}
