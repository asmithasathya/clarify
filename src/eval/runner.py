"""Experiment runner: load data, run methods, compute metrics, write artifacts."""

from __future__ import annotations

from copy import deepcopy
import re
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from tqdm.auto import tqdm

from src.data.clarifybench import load_clarifybench_local
from src.data.infoquest import load_infoquest, load_infoquest_local
from src.data.report_data import resolve_split_path, sha256_file
from src.data.schema import DialogueExample, MethodResult
from src.eval.judge import populate_quality_scores, LLMJudge
from src.eval.metrics import compute_all_metrics
from src.llm.generator import build_generator
from src.methods.direct_answer import run_direct_answer
from src.methods.generic_clarify import run_generic_clarify
from src.methods.generic_hedge import run_generic_hedge
from src.methods.resample_clarify import run_resample_clarify
from src.methods.targeted_clarify import run_targeted_clarify
from src.understand.ambiguity_detector import AmbiguityDetector
from src.understand.clarification_generator import ClarificationGenerator
from src.understand.intent_model import IntentModeler
from src.understand.strategy_selector import StrategySelector
from src.utils.io import append_jsonl, ensure_dir, markdown_table, read_json, write_csv, write_json, write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


METHOD_REGISTRY: dict[str, Callable[..., MethodResult]] = {
    "direct_answer": run_direct_answer,
    "generic_hedge": run_generic_hedge,
    "generic_clarify": run_generic_clarify,
    "targeted_clarify": run_targeted_clarify,
    "resample_clarify": run_resample_clarify,
}

METRIC_COLUMNS = [
    "method",
    "task_success_rate",
    "appropriate_action_rate",
    "clarification_precision",
    "clarification_recall",
    "clarification_rate",
    "answer_rate",
    "final_answer_rate",
    "multi_turn_completion_rate",
    "missed_ambiguity_rate",
    "unnecessary_clarification_rate",
    "average_answer_score",
    "average_clarification_quality",
    "average_alternatives_quality",
    "mean_intent_confidence",
    "intent_stability_rate",
    "resample_rate",
    "avg_resample_rounds",
    "clarification_after_resampling_rate",
    "resolved_without_clarification_rate",
    "avg_task_model_calls",
    "avg_latency",
    "avg_estimated_cost",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_examples(config: dict[str, Any], limit: int | None = None) -> list[DialogueExample]:
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("primary_dataset", "infoquest")
    split_name = data_cfg.get("split")
    if dataset_name == "infoquest":
        iq_cfg = data_cfg.get("infoquest", {})
        local_path = resolve_split_path(iq_cfg, split_name)
        if local_path:
            examples = load_infoquest_local(local_path, limit=limit)
        else:
            examples = load_infoquest(
                limit=limit,
                both_settings=iq_cfg.get("both_settings", True),
            )
        for example in examples:
            if not example.split_name and split_name:
                example.split_name = split_name
            if example.dataset_name == "unknown":
                example.dataset_name = "infoquest"
        return examples
    if dataset_name == "clarifybench":
        bench_cfg = data_cfg.get("clarifybench", {})
        local_path = resolve_split_path(bench_cfg, split_name)
        if not local_path:
            raise ValueError("clarifybench requires data.clarifybench.local_path")
        examples = load_clarifybench_local(local_path, limit=limit)
        for example in examples:
            if not example.split_name and split_name:
                example.split_name = split_name
            if example.dataset_name == "unknown":
                example.dataset_name = "clarifybench"
        return examples
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Resource construction
# ---------------------------------------------------------------------------

def build_resources(config: dict[str, Any]) -> dict[str, Any]:
    generator = build_generator(config)
    resources = {
        "generator": generator,
        "ambiguity_detector": AmbiguityDetector(config, generator=generator),
        "intent_modeler": IntentModeler(config, generator=generator),
        "strategy_selector": StrategySelector(config, generator=generator),
        "clarification_generator": ClarificationGenerator(config, generator=generator),
    }
    judge_cfg = config.get("evaluation", {}).get("llm_judge", {})
    if judge_cfg.get("enabled", False):
        judge_generator = generator
        if judge_cfg.get("base_model") or judge_cfg.get("model_path"):
            judge_config = deepcopy(config)
            judge_model_cfg = judge_config.setdefault("model", {}).setdefault("generator", {})
            if judge_cfg.get("base_model"):
                judge_model_cfg["base_model"] = judge_cfg["base_model"]
                judge_model_cfg["model_path"] = None
            if judge_cfg.get("model_path"):
                judge_model_cfg["model_path"] = judge_cfg["model_path"]
            judge_generator = build_generator(judge_config)
        resources["llm_judge"] = LLMJudge(judge_generator)
    else:
        resources["llm_judge"] = None
    return resources


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


def get_task_model_id(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {}).get("generator", {})
    return model_cfg.get("model_path") or model_cfg.get("base_model") or model_cfg.get("backend", "unknown")


def get_judge_model_id(config: dict[str, Any]) -> str | None:
    judge_cfg = config.get("evaluation", {}).get("llm_judge", {})
    if not judge_cfg.get("enabled", False):
        return None
    return judge_cfg.get("model_path") or judge_cfg.get("base_model") or get_task_model_id(config)


def build_run_manifest(config: dict[str, Any], examples: Sequence[DialogueExample], methods: Sequence[str]) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("primary_dataset", "infoquest")
    split_name = data_cfg.get("split")
    dataset_cfg = data_cfg.get(dataset_name, {})
    source_path = resolve_split_path(dataset_cfg, split_name)
    return {
        "created_at": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "split_name": split_name,
        "dataset_source_path": str(source_path) if source_path else None,
        "dataset_sha256": sha256_file(source_path) if source_path and Path(source_path).exists() else None,
        "n_examples": len(examples),
        "methods": list(methods),
        "task_model_id": get_task_model_id(config),
        "judge_model_id": get_judge_model_id(config),
        "prompt_version": config.get("project", {}).get("prompt_version", "unversioned"),
        "judge_version": config.get("evaluation", {}).get("llm_judge", {}).get("version"),
        "seed": config.get("project", {}).get("seed", 42),
    }


def _annotate_result_metadata(
    result: MethodResult,
    example: DialogueExample,
    config: dict[str, Any],
) -> MethodResult:
    result.dataset_name = example.dataset_name
    result.split_name = example.split_name or config.get("data", {}).get("split")
    result.task_model_id = get_task_model_id(config)
    result.judge_model_id = get_judge_model_id(config)
    result.prompt_version = config.get("project", {}).get("prompt_version", "unversioned")
    result.judge_version = config.get("evaluation", {}).get("llm_judge", {}).get("version")
    return result


def _confidence_band_for_value(config: dict[str, Any], confidence: float) -> str:
    calibration_cfg = config.get("intent_calibration", {})
    medium_threshold = float(calibration_cfg.get("medium_threshold", 0.55))
    high_threshold = float(calibration_cfg.get("high_threshold", 0.80))
    if confidence >= high_threshold:
        return "high"
    if confidence >= medium_threshold:
        return "medium"
    return "low"


def _evaluate_result(
    result: MethodResult,
    example: DialogueExample,
    config: dict[str, Any],
    resources: dict[str, Any],
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
    return populate_quality_scores(result, example, judge=resources.get("llm_judge"))


def _fallback_user_reply(example: DialogueExample) -> str:
    if example.simulated_user_reply:
        return example.simulated_user_reply
    hidden = example.hidden_context.strip()
    if not hidden:
        return "Here are the relevant details."
    return f"Here are the relevant details: {hidden}"


def _maybe_continue_dialogue(
    result: MethodResult,
    example: DialogueExample,
    config: dict[str, Any],
    resources: dict[str, Any],
) -> MethodResult:
    max_turns = config.get("understand", {}).get("max_clarification_turns", 1)
    if max_turns <= 1:
        return result
    if result.response_strategy not in {"ask_clarification", "present_alternatives"}:
        return result

    clarification_generator: ClarificationGenerator = resources["clarification_generator"]
    simulation_cfg = config.get("evaluation", {}).get("user_reply_simulation", {})

    if example.simulated_user_reply:
        simulated_user_reply = example.simulated_user_reply
        source = "dataset"
    elif simulation_cfg.get("use_generator", False):
        simulated = clarification_generator.simulate_user_reply(
            request=example.user_request,
            assistant_message=result.response_text,
            hidden_context=example.hidden_context,
        )
        simulated_user_reply = simulated.user_reply
        source = "generator"
        result.trace["simulated_user_reply"] = simulated.model_dump()
    else:
        simulated_user_reply = _fallback_user_reply(example)
        source = "fallback"

    conversation = [
        {"role": "user", "content": example.user_request},
        {"role": "assistant", "content": result.response_text},
        {"role": "user", "content": simulated_user_reply},
    ]
    answer = clarification_generator.generate_conversation_answer(conversation)
    result.task_model_calls += 1

    result.simulated_user_reply = simulated_user_reply
    result.final_answer = answer.answer
    result.answered_after_clarification = True
    result.num_turns = len(conversation)
    result.trace["followup_turn"] = {
        "simulated_user_reply": simulated_user_reply,
        "source": source,
        "conversation_answer": answer.model_dump(),
    }
    if source == "generator":
        result.task_model_calls += 1
    return result


def _write_case_studies(results: Sequence[MethodResult], path: Path, limit: int = 5) -> None:
    sections: list[str] = ["# Case Studies", ""]
    selected = list(results[:limit])
    for result in selected:
        sections.extend(
            [
                f"## {result.example_id}",
                f"- strategy: `{result.response_strategy}`",
                f"- correct: `{result.correct}`",
                f"- turns: `{result.num_turns}`",
                f"- answer_score: `{result.answer_score}`",
                f"- clarification_quality: `{result.clarification_quality_score}`",
                f"- alternatives_quality: `{result.alternatives_quality_score}`",
                "",
                "### Response",
                result.response_text,
                "",
            ]
        )
        if result.simulated_user_reply:
            sections.extend(
                [
                    "### Simulated User Reply",
                    result.simulated_user_reply,
                    "",
                ]
            )
        if result.final_answer:
            sections.extend(
                [
                    "### Final Answer",
                    result.final_answer,
                    "",
                ]
            )
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _write_method_outputs(
    method_name: str,
    results: Sequence[MethodResult],
    metrics: dict[str, Any],
    output_dir: str | Path,
    *,
    nest_method_dir: bool = True,
    case_study_limit: int = 5,
    run_manifest: dict[str, Any] | None = None,
) -> None:
    method_dir = ensure_dir(Path(output_dir) / method_name) if nest_method_dir else ensure_dir(output_dir)
    write_jsonl(results, method_dir / "predictions.jsonl")
    write_json(metrics, method_dir / "metrics.json")
    write_csv([_flatten_metrics("", metrics)], method_dir / "summary.csv")
    if run_manifest is not None:
        write_json({**run_manifest, "method": method_name}, method_dir / "manifest.json")

    row = [method_name]
    for col in METRIC_COLUMNS[1:]:
        val = metrics.get(col, "")
        row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
    report = markdown_table(METRIC_COLUMNS, [row])
    (method_dir / "report_snippet.md").write_text(report + "\n", encoding="utf-8")
    _write_case_studies(results, method_dir / "case_studies.md", limit=case_study_limit)


def _completed_method_metrics(
    method_name: str,
    output_dir: str | Path,
    *,
    run_manifest: dict[str, Any],
    nest_method_dir: bool = True,
) -> dict[str, Any] | None:
    method_dir = Path(output_dir) / method_name if nest_method_dir else Path(output_dir)
    metrics_path = method_dir / "metrics.json"
    manifest_path = method_dir / "manifest.json"
    if not metrics_path.exists() or not manifest_path.exists():
        return None
    existing_manifest = read_json(manifest_path)
    comparable_keys = (
        "dataset_name",
        "split_name",
        "dataset_sha256",
        "task_model_id",
        "judge_model_id",
        "prompt_version",
        "judge_version",
    )
    for key in comparable_keys:
        if existing_manifest.get(key) != run_manifest.get(key):
            return None
    return read_json(metrics_path)


def _load_partial_method_results(
    method_name: str,
    output_dir: str | Path,
    *,
    run_manifest: dict[str, Any] | None,
    nest_method_dir: bool = True,
) -> tuple[list[MethodResult], bool]:
    method_dir = Path(output_dir) / method_name if nest_method_dir else Path(output_dir)
    predictions_path = method_dir / "predictions.jsonl"
    manifest_path = method_dir / "manifest.json"
    if run_manifest is None or not predictions_path.exists() or not manifest_path.exists():
        return [], False

    existing_manifest = read_json(manifest_path)
    comparable_keys = (
        "dataset_name",
        "split_name",
        "dataset_sha256",
        "task_model_id",
        "judge_model_id",
        "prompt_version",
        "judge_version",
    )
    for key in comparable_keys:
        if existing_manifest.get(key) != run_manifest.get(key):
            return [], False

    results: list[MethodResult] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                results.append(MethodResult.model_validate_json(stripped))
            except Exception as exc:
                LOGGER.warning(
                    "Ignoring malformed partial prediction at %s line %s: %s",
                    method_name,
                    lineno,
                    exc,
                )
                break
    return results, True


def _reset_partial_method_outputs(method_dir: Path) -> None:
    for name in (
        "predictions.jsonl",
        "metrics.json",
        "summary.csv",
        "report_snippet.md",
        "case_studies.md",
    ):
        path = method_dir / name
        if path.exists():
            path.unlink()


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
    nest_method_dir: bool = True,
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method_dir = ensure_dir(Path(output_dir) / method_name) if nest_method_dir else ensure_dir(output_dir)
    partial_results, resumable = _load_partial_method_results(
        method_name,
        output_dir,
        run_manifest=run_manifest,
        nest_method_dir=nest_method_dir,
    )
    manifest_path = method_dir / "manifest.json"
    predictions_path = method_dir / "predictions.jsonl"

    if not resumable:
        _reset_partial_method_outputs(method_dir)
    if run_manifest is not None:
        write_json({**run_manifest, "method": method_name}, manifest_path)

    loaded_by_id = {result.example_id: result for result in partial_results}
    results: list[MethodResult] = []
    remaining_examples: list[DialogueExample] = []
    for example in examples:
        existing = loaded_by_id.get(example.example_id)
        if existing is not None:
            results.append(existing)
        else:
            remaining_examples.append(example)

    if partial_results:
        LOGGER.info(
            "Resuming method: %s from %s saved predictions",
            method_name,
            len(partial_results),
        )

    iterator = tqdm(
        remaining_examples,
        desc=method_name,
        leave=False,
        initial=len(results),
        total=len(examples),
    )

    for example in iterator:
        started = time.perf_counter()
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
        elif method_name == "resample_clarify":
            result = run_resample_clarify(
                example,
                config=config,
                ambiguity_detector=resources["ambiguity_detector"],
                intent_modeler=resources["intent_modeler"],
                strategy_selector=resources["strategy_selector"],
                clarification_generator=resources["clarification_generator"],
            )
        else:
            raise ValueError(f"Unsupported method: {method_name}")

        result = _maybe_continue_dialogue(result, example, config, resources)
        result = _annotate_result_metadata(result, example, config)
        if result.task_model_calls <= 0:
            result.task_model_calls = 1
        if result.sample_count <= 0:
            result.sample_count = result.task_model_calls
        if result.estimated_cost <= 0.0:
            result.estimated_cost = float(result.task_model_calls)
        if result.latency_seconds <= 0.0:
            result.latency_seconds = time.perf_counter() - started
        if result.intent_confidence is None:
            result.intent_confidence = result.confidence
        if result.confidence_band is None:
            result.confidence_band = _confidence_band_for_value(config, result.intent_confidence)
        result = _evaluate_result(result, example, config, resources)
        results.append(result)
        append_jsonl(result, predictions_path)

    metrics = compute_all_metrics(results)
    _write_method_outputs(
        method_name,
        results,
        metrics,
        output_dir,
        nest_method_dir=nest_method_dir,
        case_study_limit=config.get("evaluation", {}).get("case_study_limit", 5),
        run_manifest=run_manifest,
    )
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
    examples: Sequence[DialogueExample] | None = None,
) -> dict[str, Any]:
    set_seed(config.get("project", {}).get("seed", 42))
    examples = list(examples) if examples is not None else load_examples(config, limit=limit)
    resources = build_resources(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = output_root or Path(config["paths"]["outputs_dir"]) / f"run_{timestamp}"
    output_dir = ensure_dir(output_root)
    write_json(config, Path(output_dir) / "resolved_config.json")
    run_manifest = build_run_manifest(config, examples, methods)
    write_json(run_manifest, Path(output_dir) / "run_manifest.json")

    run_summary: dict[str, Any] = {}
    summary_rows: list[list[str]] = []

    for method_name in methods:
        LOGGER.info("Running method: %s", method_name)
        existing_metrics = _completed_method_metrics(
            method_name,
            output_dir,
            run_manifest=run_manifest,
            nest_method_dir=True,
        )
        if existing_metrics is not None:
            LOGGER.info("Skipping completed method: %s", method_name)
            run_summary[method_name] = existing_metrics
            m = existing_metrics
            row = [method_name]
            for col in METRIC_COLUMNS[1:]:
                val = m.get(col, "")
                row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            summary_rows.append(row)
            continue
        payload = run_method(
            method_name,
            config=config,
            examples=examples,
            resources=resources,
            output_dir=output_dir,
            nest_method_dir=True,
            run_manifest=run_manifest,
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
