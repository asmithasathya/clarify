"""Run or aggregate a sharded dataset/model baseline leaf."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from src.eval.runner import load_examples, run_experiment
from src.eval.sharding import (
    aggregate_sharded_experiment,
    partition_examples,
    shard_completion_status,
    shard_dir,
    write_shard_plan,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir, read_json, write_json


def _resolve_report_specs(config: dict, dataset_name: str, split_name: str | None, model_label: str) -> dict:
    report_cfg = config.get("report", {})
    dataset_match = None
    for dataset_spec in report_cfg.get("datasets", []):
        if dataset_spec["name"] == dataset_name and (
            split_name is None or dataset_spec.get("split") == split_name
        ):
            dataset_match = dataset_spec
            break
    if dataset_match is None:
        raise ValueError(f"Could not resolve dataset {dataset_name!r} with split {split_name!r} from report config.")

    model_match = None
    for model_spec in report_cfg.get("task_models", []):
        if model_spec["label"] == model_label:
            model_match = model_spec
            break
    if model_match is None:
        raise ValueError(f"Could not resolve model label {model_label!r} from report config.")

    resolved = deepcopy(config)
    resolved["data"]["primary_dataset"] = dataset_match["name"]
    resolved["data"]["split"] = split_name or dataset_match.get("split")
    resolved["model"]["generator"]["base_model"] = model_match["base_model"]
    resolved["model"]["generator"]["model_path"] = None
    resolved["evaluation"]["llm_judge"]["base_model"] = report_cfg["judge_model"]["base_model"]
    return {
        "config": resolved,
        "dataset_spec": dataset_match,
        "model_spec": model_match,
        "methods": list(report_cfg.get("methods", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one shard of a sharded baseline leaf and aggregate when ready.")
    parser.add_argument("--config", default="configs/report_baselines.yaml", help="Report config path.")
    parser.add_argument("--output-dir", required=True, help="Baseline matrix root directory.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. infoquest.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to the report config split.")
    parser.add_argument("--model-label", required=True, help="Report model label, e.g. qwen3_30b.")
    parser.add_argument("--shard-count", type=int, required=True, help="Number of shards for the leaf.")
    parser.add_argument("--shard-index", type=int, default=None, help="Zero-based shard index to run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit for smoke runs.")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip shard execution and only attempt aggregation.")
    args = parser.parse_args()

    config = load_config(args.config)
    resolved_specs = _resolve_report_specs(config, args.dataset, args.split, args.model_label)
    resolved = resolved_specs["config"]
    methods = resolved_specs["methods"]

    examples = load_examples(resolved, limit=args.limit)
    shard_count = min(args.shard_count, len(examples)) if examples else args.shard_count
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if args.shard_index is not None and not 0 <= args.shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count - 1}]")

    leaf_root = ensure_dir(Path(args.output_dir) / args.dataset / args.model_label)
    shard_sets = partition_examples(examples, shard_count)
    write_shard_plan(
        leaf_root,
        config=resolved,
        methods=methods,
        examples=examples,
        shard_count=shard_count,
    )

    if not args.aggregate_only:
        if args.shard_index is None:
            raise ValueError("--shard-index is required unless --aggregate-only is set.")
        shard_examples = shard_sets[args.shard_index]
        shard_root = shard_dir(leaf_root, args.shard_index, shard_count)
        payload = run_experiment(
            config=resolved,
            methods=methods,
            output_root=shard_root,
            examples=shard_examples,
        )
        shard_manifest_path = Path(payload["output_dir"]) / "run_manifest.json"
        shard_manifest = read_json(shard_manifest_path)
        shard_manifest["shard_index"] = args.shard_index
        shard_manifest["shard_count"] = shard_count
        shard_manifest["leaf_output_dir"] = str(leaf_root)
        write_json(shard_manifest, shard_manifest_path)

    aggregated = aggregate_sharded_experiment(
        config=resolved,
        methods=methods,
        examples=examples,
        leaf_root=leaf_root,
        shard_count=shard_count,
    )
    if aggregated["complete"]:
        print(f"Completed sharded leaf. Outputs saved to {leaf_root}")
        return

    status = shard_completion_status(
        config=resolved,
        methods=methods,
        examples=examples,
        leaf_root=leaf_root,
        shard_count=shard_count,
    )
    print(
        "Shard run finished but leaf is still incomplete: "
        f"{status['completed_shards']}/{status['total_shards']} shards complete"
    )


if __name__ == "__main__":
    main()
