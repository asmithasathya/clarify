"""Shard planning and aggregation helpers for long-running evaluation leaves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.data.schema import DialogueExample, MethodResult
from src.eval.metrics import compute_all_metrics
from src.eval.runner import (
    METRIC_COLUMNS,
    _completed_method_metrics,
    _flatten_metrics,
    _write_method_outputs,
    build_run_manifest,
)
from src.utils.io import ensure_dir, markdown_table, write_csv, write_json


def partition_examples(examples: Sequence[DialogueExample], shard_count: int) -> list[list[DialogueExample]]:
    """Split examples deterministically across shards while preserving source order."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not examples:
        return [[] for _ in range(shard_count)]

    shards: list[list[DialogueExample]] = [[] for _ in range(shard_count)]
    for index, example in enumerate(examples):
        shards[index % shard_count].append(example)
    return shards


def shard_dir(leaf_root: str | Path, shard_index: int, shard_count: int) -> Path:
    return ensure_dir(Path(leaf_root) / "shards" / f"shard_{shard_index:03d}_of_{shard_count:03d}")


def shard_plan_payload(
    *,
    config: dict[str, Any],
    methods: Sequence[str],
    examples: Sequence[DialogueExample],
    shard_count: int,
) -> dict[str, Any]:
    base_manifest = build_run_manifest(config, examples, methods)
    shard_sizes = [len(shard) for shard in partition_examples(examples, shard_count)]
    return {
        **base_manifest,
        "shard_count": shard_count,
        "shard_sizes": shard_sizes,
        "status": "planned",
    }


def write_shard_plan(
    leaf_root: str | Path,
    *,
    config: dict[str, Any],
    methods: Sequence[str],
    examples: Sequence[DialogueExample],
    shard_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = shard_plan_payload(
        config=config,
        methods=methods,
        examples=examples,
        shard_count=shard_count,
    )
    if extra:
        payload.update(extra)
    write_json(payload, Path(leaf_root) / "sharding.json")
    return payload


def _read_method_results(predictions_path: Path) -> list[MethodResult]:
    results: list[MethodResult] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                results.append(MethodResult.model_validate_json(stripped))
    return results


def _shard_run_manifest(
    config: dict[str, Any],
    methods: Sequence[str],
    examples: Sequence[DialogueExample],
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    manifest = build_run_manifest(config, examples, methods)
    manifest["shard_index"] = shard_index
    manifest["shard_count"] = shard_count
    return manifest


def shard_completion_status(
    *,
    config: dict[str, Any],
    methods: Sequence[str],
    examples: Sequence[DialogueExample],
    leaf_root: str | Path,
    shard_count: int,
) -> dict[str, Any]:
    shard_sets = partition_examples(examples, shard_count)
    status: dict[str, Any] = {
        "complete": True,
        "completed_shards": 0,
        "total_shards": shard_count,
        "missing": [],
        "shard_sizes": [len(shard) for shard in shard_sets],
    }
    for shard_index, shard_examples in enumerate(shard_sets):
        shard_root = shard_dir(leaf_root, shard_index, shard_count)
        shard_manifest = _shard_run_manifest(config, methods, shard_examples, shard_index, shard_count)
        shard_complete = True
        for method_name in methods:
            metrics = _completed_method_metrics(
                method_name,
                shard_root,
                run_manifest=shard_manifest,
                nest_method_dir=True,
            )
            if metrics is None:
                status["complete"] = False
                shard_complete = False
                status["missing"].append(
                    {
                        "shard_index": shard_index,
                        "method": method_name,
                        "shard_root": str(shard_root),
                    }
                )
        if shard_complete:
            status["completed_shards"] += 1
    return status


def aggregate_sharded_experiment(
    *,
    config: dict[str, Any],
    methods: Sequence[str],
    examples: Sequence[DialogueExample],
    leaf_root: str | Path,
    shard_count: int,
) -> dict[str, Any]:
    leaf_root = ensure_dir(leaf_root)
    status = shard_completion_status(
        config=config,
        methods=methods,
        examples=examples,
        leaf_root=leaf_root,
        shard_count=shard_count,
    )
    if not status["complete"]:
        write_shard_plan(
            leaf_root,
            config=config,
            methods=methods,
            examples=examples,
            shard_count=shard_count,
            extra=status,
        )
        return {
            "output_dir": str(leaf_root),
            "complete": False,
            "status": status,
            "metrics": {},
        }

    run_manifest = build_run_manifest(config, examples, methods)
    run_manifest["shard_count"] = shard_count
    run_manifest["aggregated_from_shards"] = True
    write_json(config, Path(leaf_root) / "resolved_config.json")
    write_json(run_manifest, Path(leaf_root) / "run_manifest.json")

    example_order = {example.example_id: index for index, example in enumerate(examples)}
    expected_ids = set(example_order)
    run_summary: dict[str, Any] = {}
    summary_rows: list[list[str]] = []

    shard_sets = partition_examples(examples, shard_count)
    for method_name in methods:
        results: list[MethodResult] = []
        seen_ids: set[str] = set()
        for shard_index, _ in enumerate(shard_sets):
            shard_root = shard_dir(leaf_root, shard_index, shard_count)
            predictions_path = shard_root / method_name / "predictions.jsonl"
            shard_results = _read_method_results(predictions_path)
            for result in shard_results:
                if result.example_id in seen_ids:
                    raise ValueError(f"Duplicate example {result.example_id} in aggregated {method_name} shard outputs.")
                seen_ids.add(result.example_id)
                results.append(result)

        missing_ids = expected_ids - seen_ids
        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} examples while aggregating {method_name}: {sorted(missing_ids)[:5]}"
            )

        results.sort(key=lambda result: example_order[result.example_id])
        metrics = compute_all_metrics(results)
        _write_method_outputs(
            method_name,
            results,
            metrics,
            leaf_root,
            nest_method_dir=True,
            case_study_limit=config.get("evaluation", {}).get("case_study_limit", 5),
            run_manifest=run_manifest,
        )
        run_summary[method_name] = metrics
        row = [method_name]
        for col in METRIC_COLUMNS[1:]:
            value = metrics.get(col, "")
            row.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        summary_rows.append(row)

    comparison_table = markdown_table(METRIC_COLUMNS, summary_rows)
    (Path(leaf_root) / "comparison_report.md").write_text(comparison_table + "\n", encoding="utf-8")
    write_json(run_summary, Path(leaf_root) / "all_metrics.json")
    write_csv(
        [
            {"method": method_name, **_flatten_metrics("", metrics)}
            for method_name, metrics in run_summary.items()
        ],
        Path(leaf_root) / "all_metrics.csv",
    )
    write_shard_plan(
        leaf_root,
        config=config,
        methods=methods,
        examples=examples,
        shard_count=shard_count,
        extra={
            "complete": True,
            "completed_shards": shard_count,
            "total_shards": shard_count,
            "missing": [],
            "status": "complete",
        },
    )
    return {
        "output_dir": str(leaf_root),
        "complete": True,
        "status": {"complete": True, "completed_shards": shard_count, "total_shards": shard_count, "missing": []},
        "metrics": run_summary,
    }
