"""Emit terminal-safe commands for the expensive research phases."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config


RESAMPLE_ABLATIONS = [
    "single_sample_only",
    "no_selective_resample",
    "resample_all_stages",
    "one_round_only",
    "clarify_immediately",
    "no_clarification_fallback",
    "no_calibration",
    "generic_question_fallback",
]


def _teacher_model_label(config: dict) -> str:
    return config.get("report", {}).get("teacher_model_label", "qwen3_30b")


def _student_model_label(config: dict) -> str:
    return config.get("report", {}).get("student_model_label", "qwen3_4b")


def _shard_count(config: dict, dataset_name: str, model_label: str) -> int:
    runtime = config.get("runtime", {}).get("parallelism", {})
    if dataset_name == "clarifybench" and model_label == _student_model_label(config):
        return 1
    if dataset_name == "infoquest" and model_label == _student_model_label(config):
        return 1
    if dataset_name == "clarifybench":
        return int(runtime.get("teacher_shard_count_clarifybench", 3))
    return int(runtime.get("teacher_shard_count_infoquest", 4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print shard-safe terminal commands for the research pipeline.")
    parser.add_argument("--config", default="configs/report_baselines.yaml")
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3", "all"], default="all")
    parser.add_argument("--output-root", default="outputs/research_pipeline")
    args = parser.parse_args()

    config = load_config(args.config)
    report = config.get("report", {})
    output_root = Path(args.output_root)
    py = "./.venv/bin/python"
    manifest_reader = f"{py} scripts/read_manifest_value.py"

    commands: list[str] = []
    if args.phase in {"phase1", "all"}:
        commands.append(f"# Phase 1A: dev sweep\n{py} scripts/run_dev_sweep.py --config {args.config} --output-dir {output_root / 'dev_sweep'}")
        commands.append(f"# Phase 1B: qwen3_4b full leaf\n{py} scripts/run_sharded_leaf.py --config {args.config} --output-dir {output_root / 'baseline_matrix'} --dataset infoquest --split test --model-label {_student_model_label(config)} --shard-count 1 --shard-index 0")
        commands.append(f"{py} scripts/run_sharded_leaf.py --config {args.config} --output-dir {output_root / 'baseline_matrix'} --dataset clarifybench --split test --model-label {_student_model_label(config)} --shard-count 1 --shard-index 0")
        for dataset in report.get("datasets", []):
            dataset_name = dataset["name"]
            split = dataset.get("split", "test")
            for model in report.get("task_models", []):
                label = model["label"]
                if label == _student_model_label(config):
                    continue
                shard_count = _shard_count(config, dataset_name, label)
                for shard_index in range(shard_count):
                    commands.append(
                        f"{py} scripts/run_sharded_leaf.py --config {args.config} --output-dir {output_root / 'baseline_matrix'} "
                        f"--dataset {dataset_name} --split {split} --model-label {label} --shard-count {shard_count} --shard-index {shard_index}"
                    )
        commands.append(f"# Aggregate baseline leaves after shard completion\n{py} scripts/aggregate_baseline_matrix.py --config {args.config} --output-dir {output_root / 'baseline_matrix'}")
        commands.append(f"# Repeatability (run after matrix completion)\n{py} scripts/run_baseline_matrix.py --config {args.config} --output-dir {output_root / 'baseline_matrix'}")
        for ablation in RESAMPLE_ABLATIONS:
            commands.append(
                f"{py} scripts/run_ablation.py --config {args.config} --dataset infoquest --split test "
                f"--output-dir {output_root / 'ablations'} --method resample_clarify --ablation {ablation}"
            )
        commands.append(
            f"# Audit export and paper tables\n"
            f"{py} scripts/export_audit.py --matrix-root {output_root / 'baseline_matrix'} --output-dir {output_root / 'audit'}\n"
            f"{py} scripts/paper_tables.py --matrix-root {output_root / 'baseline_matrix'} --ablation-root {output_root / 'ablations'} --student-root {output_root / 'student_eval'} --output-dir paper/generated\n"
            f"{py} scripts/build_paper.py"
        )

    if args.phase in {"phase2", "all"}:
        teacher_root = output_root / "teacher_rollouts"
        shard_count = int(config.get("runtime", {}).get("parallelism", {}).get("rollout_shard_count", 4))
        for shard_index in range(shard_count):
            commands.append(
                f"{py} scripts/export_teacher_rollouts.py --config {args.config} --output-dir {teacher_root} "
                f"--dataset infoquest --split train --rollouts-per-example {config.get('distillation', {}).get('rollouts_per_example', 4)} "
                f"--baseline-root {output_root / 'baseline_matrix'} --shard-count {shard_count} --shard-index {shard_index}"
            )
        commands.append(f"{py} scripts/export_teacher_rollouts.py --config {args.config} --output-dir {teacher_root} --dataset infoquest --split train --rollouts-per-example {config.get('distillation', {}).get('rollouts_per_example', 4)} --shard-count {shard_count} --aggregate-only")
        commands.append(
            f"{py} scripts/run_student_sft.py --config {args.config} --train-corpus {teacher_root / 'sft_corpus.jsonl'} "
            f"--output-dir {output_root / 'student_sft'}"
        )
        commands.append(
            f"{py} scripts/run_student_eval.py --config {args.config} "
            f"--sft-checkpoint $({manifest_reader} --path {output_root / 'student_sft' / 'student_sft_manifest.json'} --key checkpoint_path) "
            f"--output-dir {output_root / 'student_eval'}"
        )

    if args.phase in {"phase3", "all"}:
        commands.append(
            f"{py} scripts/run_student_dpo.py --config {args.config} "
            f"--preference-corpus {output_root / 'teacher_rollouts' / 'preference_corpus.jsonl'} "
            f"--sft-checkpoint $({manifest_reader} --path {output_root / 'student_sft' / 'student_sft_manifest.json'} --key checkpoint_path) "
            f"--output-dir {output_root / 'student_dpo'} --prepare-only"
        )

    print("\n\n".join(commands))


if __name__ == "__main__":
    main()
