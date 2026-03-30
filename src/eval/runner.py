"""Experiment runner for methods, metrics, and artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm.auto import tqdm

from src.data.housingqa import load_housingqa_questions
from src.data.legalqa_local import load_local_legalqa
from src.data.schema import MethodResult
from src.eval.metrics import compute_all_metrics
from src.llm.generator import build_generator
from src.methods.closed_book import run_closed_book
from src.methods.hedge import run_hedge
from src.methods.rag_direct import run_rag_direct
from src.methods.revise_verify import run_revise_verify
from src.retrieval.query_rewrite import AlternativeTrajectoryBuilder
from src.retrieval.retrieve import build_retriever
from src.utils.io import ensure_dir, markdown_table, write_csv, write_json, write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.verify.abstain import AbstentionPolicy
from src.verify.claim_extraction import ClaimExtractor
from src.verify.claim_scoring import build_support_scorer
from src.verify.revise import RevisionEngine
from src.verify.support_checker import LLMSupportJudge


LOGGER = get_logger(__name__)


METHOD_REGISTRY: dict[str, Callable[..., MethodResult]] = {
    "closed_book": run_closed_book,
    "rag_direct": run_rag_direct,
    "hedge": run_hedge,
    "revise_verify": run_revise_verify,
}


def load_examples(config: dict[str, Any], limit: int | None = None) -> list[Any]:
    dataset_name = config.get("data", {}).get("primary_dataset", "housingqa")
    if dataset_name == "housingqa":
        housing_cfg = config["data"]["housingqa"]
        return load_housingqa_questions(
            dataset_name=housing_cfg["questions_dataset"],
            config_name=housing_cfg["questions_config"],
            split=housing_cfg["questions_split"],
            limit=limit,
        )
    if dataset_name == "legalqa_local":
        local_cfg = config["data"]["legalqa_local"]
        return load_local_legalqa(local_cfg["path"])[:limit]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def gold_lookup_from_examples(examples: Sequence[Any]) -> dict[str, Sequence[str]]:
    return {example.example_id: list(example.statutes) + list(example.citation) for example in examples}


def build_resources(config: dict[str, Any], methods: Sequence[str]) -> dict[str, Any]:
    generator = build_generator(config)
    need_retrieval = any(method != "closed_book" for method in methods)
    retriever = build_retriever(config) if need_retrieval else None
    judge = LLMSupportJudge(config, generator=generator)
    support_scorer = build_support_scorer(config, judge=judge)
    return {
        "generator": generator,
        "retriever": retriever,
        "claim_extractor": ClaimExtractor(config, generator=generator),
        "support_scorer": support_scorer,
        "trajectory_builder": AlternativeTrajectoryBuilder(config, generator=generator),
        "revision_engine": RevisionEngine(config, generator=generator),
        "abstention_policy": AbstentionPolicy(config, generator=generator),
    }


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


def _write_case_studies(
    results: Sequence[MethodResult],
    output_dir: str | Path,
    limit: int = 25,
) -> None:
    case_root = ensure_dir(Path(output_dir) / "case_studies")
    buckets = {
        "improved_after_revision": [],
        "abstained_correctly": [],
        "failed_revision": [],
        "over-abstained": [],
    }
    for result in results:
        initial_answer = result.trace.get("initial_answer", {}).get("answer")
        if result.method == "revise_verify" and result.correct and initial_answer and initial_answer != result.predicted_answer:
            buckets["improved_after_revision"].append(result)
        if result.abstained and initial_answer and initial_answer != result.gold_answer:
            buckets["abstained_correctly"].append(result)
        if result.method == "revise_verify" and not result.abstained and not result.correct:
            buckets["failed_revision"].append(result)
        if result.abstained and initial_answer and initial_answer == result.gold_answer:
            buckets["over-abstained"].append(result)

    for bucket_name, bucket_results in buckets.items():
        bucket_dir = ensure_dir(case_root / bucket_name)
        for index, result in enumerate(bucket_results[:limit], start=1):
            path = bucket_dir / f"{index:03d}_{result.example_id}.md"
            lines = [
                f"# {bucket_name.replace('_', ' ').title()}",
                "",
                f"- Example ID: {result.example_id}",
                f"- Method: {result.method}",
                f"- State: {result.state}",
                f"- Gold answer: {result.gold_answer}",
                f"- Predicted answer: {result.predicted_answer}",
                f"- Confidence: {result.confidence:.3f}",
                "",
                "## Question",
                result.question,
                "",
                "## Explanation",
                result.explanation,
                "",
                "## Trace",
                "```json",
                result.model_dump_json(indent=2),
                "```",
            ]
            path.write_text("\n".join(lines), encoding="utf-8")


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
    report = markdown_table(
        ["method", "exact_match_accuracy", "bad_acceptance_proxy", "useful_answer_rate", "selective_accuracy", "coverage"],
        [[
            method_name,
            f"{metrics['exact_match_accuracy']:.3f}",
            f"{metrics['bad_acceptance_proxy']:.3f}",
            f"{metrics['useful_answer_rate']:.3f}",
            f"{metrics['selective_accuracy']:.3f}",
            f"{metrics['coverage']:.3f}",
        ]],
    )
    (method_dir / "report_snippet.md").write_text(report + "\n", encoding="utf-8")
    _write_case_studies(results, method_dir)


def run_method(
    method_name: str,
    *,
    config: dict[str, Any],
    examples: Sequence[Any],
    resources: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    method_fn = METHOD_REGISTRY[method_name]
    results: list[MethodResult] = []
    iterator = tqdm(examples, desc=f"{method_name}", leave=False)

    for example in iterator:
        if method_name == "closed_book":
            result = method_fn(example, generator=resources["generator"], config=config)
        elif method_name in {"rag_direct", "hedge"}:
            result = method_fn(
                example,
                generator=resources["generator"],
                retriever=resources["retriever"],
                config=config,
                support_scorer=resources["support_scorer"],
            )
        elif method_name == "revise_verify":
            result = method_fn(
                example,
                generator=resources["generator"],
                retriever=resources["retriever"],
                config=config,
                claim_extractor=resources["claim_extractor"],
                support_scorer=resources["support_scorer"],
                trajectory_builder=resources["trajectory_builder"],
                revision_engine=resources["revision_engine"],
                abstention_policy=resources["abstention_policy"],
            )
        else:
            raise ValueError(f"Unsupported method: {method_name}")

        result.trace["gold_statutes"] = list(example.statutes) + list(example.citation)
        results.append(result)

    metrics = compute_all_metrics(
        results,
        gold_lookup=gold_lookup_from_examples(examples),
        retrieval_ks=config.get("evaluation", {}).get("retrieval_ks", [1, 3, 5, 10]),
        unsupported_threshold=config.get("evaluation", {}).get("unsupported_threshold", 0.55),
    )
    _write_method_outputs(method_name, results, metrics, output_dir)
    return {"results": results, "metrics": metrics}


def run_experiment(
    *,
    config: dict[str, Any],
    methods: Sequence[str],
    limit: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(config.get("project", {}).get("seed", 42))
    examples = load_examples(config, limit=limit)
    resources = build_resources(config, methods)
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
        metrics = payload["metrics"]
        summary_rows.append(
            [
                method_name,
                f"{metrics['exact_match_accuracy']:.3f}",
                f"{metrics['bad_acceptance_proxy']:.3f}",
                f"{metrics['useful_answer_rate']:.3f}",
                f"{metrics['selective_accuracy']:.3f}",
                f"{metrics['coverage']:.3f}",
            ]
        )

    comparison_table = markdown_table(
        ["method", "exact_match_accuracy", "bad_acceptance_proxy", "useful_answer_rate", "selective_accuracy", "coverage"],
        summary_rows,
    )
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
