"""Run sharded teacher rollouts and materialize SFT / preference corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.schema import MethodResult
from src.eval.runner import _annotate_result_metadata, _evaluate_result, build_resources, load_examples
from src.eval.sharding import partition_examples, shard_dir
from src.methods.resample_clarify import run_resample_clarify
from src.train.distillation import build_preference_pair, build_sft_records, index_predictions_by_example
from src.utils.config import load_config
from src.utils.io import append_jsonl, ensure_dir, read_jsonl, write_json


def _teacher_model_id(config: dict[str, Any]) -> str:
    report_cfg = config.get("report", {})
    task_models = report_cfg.get("task_models", [])
    label = report_cfg.get("teacher_model_label")
    for spec in task_models:
        if spec.get("label") == label:
            return spec["base_model"]
    return config["model"]["generator"]["base_model"]


def _load_baseline_predictions(baseline_root: str | None) -> dict[str, dict[str, Any]]:
    if not baseline_root:
        return {}
    rows: list[dict[str, Any]] = []
    for predictions_path in sorted(Path(baseline_root).rglob("predictions.jsonl")):
        rows.extend(read_jsonl(predictions_path))
    return index_predictions_by_example(rows)


def _aggregate_if_complete(root: Path, *, shard_count: int, expected_per_shard: list[int]) -> dict[str, Any]:
    shard_status: list[dict[str, Any]] = []
    complete = True
    for shard_index in range(shard_count):
        shard_root = shard_dir(root, shard_index, shard_count)
        rollouts_path = shard_root / "teacher_rollouts.jsonl"
        n_rows = len(read_jsonl(rollouts_path)) if rollouts_path.exists() else 0
        shard_status.append(
            {
                "shard_index": shard_index,
                "root": str(shard_root),
                "expected_rollouts": expected_per_shard[shard_index],
                "actual_rollouts": n_rows,
                "complete": n_rows >= expected_per_shard[shard_index],
            }
        )
        complete = complete and (n_rows >= expected_per_shard[shard_index])

    payload = {
        "complete": complete,
        "shards": shard_status,
        "shard_count": shard_count,
    }
    write_json(payload, root / "teacher_rollouts_status.json")
    if not complete:
        return payload

    for filename in ("teacher_rollouts.jsonl", "sft_corpus.jsonl", "preference_corpus.jsonl"):
        output_path = root / filename
        if output_path.exists():
            output_path.unlink()
        for shard_index in range(shard_count):
            shard_root = shard_dir(root, shard_index, shard_count)
            shard_file = shard_root / filename
            if not shard_file.exists():
                continue
            with shard_file.open("r", encoding="utf-8") as src, output_path.open("a", encoding="utf-8") as dst:
                for line in src:
                    dst.write(line)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export teacher rollouts and distillation corpora.")
    parser.add_argument("--config", default="configs/report_baselines.yaml")
    parser.add_argument("--output-dir", default="outputs/teacher_rollouts")
    parser.add_argument("--dataset", default="infoquest")
    parser.add_argument("--split", default="train")
    parser.add_argument("--rollouts-per-example", type=int, default=4)
    parser.add_argument("--baseline-root", default=None, help="Optional baseline matrix root for preference-pair rejections.")
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["data"]["primary_dataset"] = args.dataset
    config["data"]["split"] = args.split
    config["model"]["generator"]["base_model"] = _teacher_model_id(config)
    config["model"]["generator"]["model_path"] = None
    config["evaluation"]["llm_judge"]["enabled"] = False

    output_root = ensure_dir(args.output_dir)
    examples = load_examples(config)
    shards = partition_examples(examples, args.shard_count)
    expected_per_shard = [len(shard) * args.rollouts_per_example for shard in shards]

    if args.aggregate_only:
        status = _aggregate_if_complete(output_root, shard_count=args.shard_count, expected_per_shard=expected_per_shard)
        print(json.dumps(status, indent=2))
        return

    if args.shard_index is None:
        raise SystemExit("--shard-index is required unless --aggregate-only is set.")

    shard_examples = shards[args.shard_index]
    shard_root = shard_dir(output_root, args.shard_index, args.shard_count)
    baseline_by_example = _load_baseline_predictions(args.baseline_root)
    resources = build_resources(config)

    existing_rollouts = {
        row["rollout_id"]
        for row in read_jsonl(shard_root / "teacher_rollouts.jsonl")
        if "rollout_id" in row
    } if (shard_root / "teacher_rollouts.jsonl").exists() else set()

    for example in shard_examples:
        for rollout_index in range(args.rollouts_per_example):
            rollout_id = f"{example.example_id}-r{rollout_index + 1}"
            if rollout_id in existing_rollouts:
                continue
            result = run_resample_clarify(
                example,
                config=config,
                ambiguity_detector=resources["ambiguity_detector"],
                intent_modeler=resources["intent_modeler"],
                strategy_selector=resources["strategy_selector"],
                clarification_generator=resources["clarification_generator"],
            )
            result = _annotate_result_metadata(result, example, config)
            result = _evaluate_result(result, example, config, resources)

            append_jsonl({"rollout_id": rollout_id, **result.model_dump()}, shard_root / "teacher_rollouts.jsonl")
            for record in build_sft_records(result, rollout_id=rollout_id):
                append_jsonl(record, shard_root / "sft_corpus.jsonl")

            baseline = baseline_by_example.get(example.example_id)
            if baseline is not None:
                rejected = MethodResult.model_validate(baseline)
                pair = build_preference_pair(result, rejected, rollout_id=rollout_id)
                if pair is not None:
                    append_jsonl(pair, shard_root / "preference_corpus.jsonl")

    write_json(
        {
            "dataset": args.dataset,
            "split": args.split,
            "teacher_model_id": config["model"]["generator"]["base_model"],
            "rollouts_per_example": args.rollouts_per_example,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "n_examples": len(shard_examples),
        },
        shard_root / "teacher_rollout_manifest.json",
    )
    status = _aggregate_if_complete(output_root, shard_count=args.shard_count, expected_per_shard=expected_per_shard)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
