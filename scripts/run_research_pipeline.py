"""Run the end-to-end intent-stabilization research pipeline."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import ensure_dir, read_json, write_json, write_yaml


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


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("[run]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full intent-stabilization research pipeline.")
    parser.add_argument("--config", default="configs/report_baselines.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--skip-dev-sweep", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-teacher-rollouts", action="store_true")
    parser.add_argument("--skip-student-sft", action="store_true")
    parser.add_argument("--skip-student-dpo", action="store_true")
    parser.add_argument("--skip-student-eval", action="store_true")
    parser.add_argument("--skip-paper", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable
    config = load_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ensure_dir(args.output_root or (Path(config["paths"]["outputs_dir"]) / f"research_pipeline_{timestamp}"))

    write_json(
        {
            "legacy_baseline_root": "outputs/report_pipeline_20260414_123119",
            "note": "Legacy clarification-baseline artifacts retained for reference only; not part of the pivoted final paper.",
        },
        output_root / "legacy_outputs_note.json",
    )

    selected_config = deepcopy(config)
    selected_config_path = output_root / "selected_config.yaml"
    baseline_root = output_root / "baseline_matrix"
    ablation_root = output_root / "ablations"
    teacher_root = output_root / "teacher_rollouts"
    student_sft_root = output_root / "student_sft"
    student_dpo_root = output_root / "student_dpo"
    student_eval_root = output_root / "student_eval"
    audit_root = output_root / "audit"
    calibrator_path = output_root / "intent_calibrator.json"

    if not args.skip_prepare_data:
        cmd = [py, "scripts/prepare_data.py"]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        _run(cmd, cwd=repo_root, dry_run=args.dry_run)
        _run(
            [
                py,
                "scripts/validate_data.py",
                "--manifest",
                "data/infoquest_manifest.json",
                "--manifest",
                "data/clarifybench_v1_manifest.json",
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_dev_sweep:
        cmd = [py, "scripts/run_dev_sweep.py", "--config", args.config, "--output-dir", str(output_root / "dev_sweep")]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        _run(cmd, cwd=repo_root, dry_run=args.dry_run)
        if not args.dry_run:
            best_config_path = output_root / "dev_sweep" / "best_config.json"
            if not best_config_path.exists():
                best_config_path = output_root / "dev_sweep" / "fallback_config.json"
            best = read_json(best_config_path)
            selected_config["intent_resampling"]["initial_samples"] = int(best["initial_samples"])
            selected_config["intent_resampling"]["repair_samples"] = int(best["repair_samples"])
            selected_config["intent_resampling"]["max_rounds"] = int(best["max_rounds"])
            selected_config["intent_resampling"]["clarification_fallback_threshold"] = float(best["clarification_fallback_threshold"])
            predictions_path = Path(best["output_dir"]) / "resample_clarify" / "predictions.jsonl"
            _run(
                [
                    py,
                    "scripts/fit_intent_calibrator.py",
                    "--predictions",
                    str(predictions_path),
                    "--output",
                    str(calibrator_path),
                ],
                cwd=repo_root,
                dry_run=args.dry_run,
            )
            selected_config["intent_calibration"]["calibrator_path"] = str(calibrator_path)

    write_yaml(selected_config, selected_config_path)

    if not args.skip_baseline:
        cmd = [py, "scripts/run_baseline_matrix.py", "--config", str(selected_config_path), "--output-dir", str(baseline_root)]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        _run(cmd, cwd=repo_root, dry_run=args.dry_run)

    if not args.skip_ablation:
        for ablation in RESAMPLE_ABLATIONS:
            cmd = [
                py,
                "scripts/run_ablation.py",
                "--config",
                str(selected_config_path),
                "--dataset",
                "infoquest",
                "--split",
                "test",
                "--output-dir",
                str(ablation_root),
                "--method",
                "resample_clarify",
                "--ablation",
                ablation,
            ]
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            _run(cmd, cwd=repo_root, dry_run=args.dry_run)

        _run(
            [py, "scripts/export_audit.py", "--matrix-root", str(baseline_root), "--output-dir", str(audit_root)],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_teacher_rollouts:
        shard_count = int(selected_config.get("runtime", {}).get("parallelism", {}).get("rollout_shard_count", 4))
        for shard_index in range(shard_count):
            cmd = [
                py,
                "scripts/export_teacher_rollouts.py",
                "--config",
                str(selected_config_path),
                "--output-dir",
                str(teacher_root),
                "--dataset",
                "infoquest",
                "--split",
                "train",
                "--baseline-root",
                str(baseline_root),
                "--rollouts-per-example",
                str(selected_config.get("distillation", {}).get("rollouts_per_example", 4)),
                "--shard-count",
                str(shard_count),
                "--shard-index",
                str(shard_index),
            ]
            _run(cmd, cwd=repo_root, dry_run=args.dry_run)
        _run(
            [
                py,
                "scripts/export_teacher_rollouts.py",
                "--config",
                str(selected_config_path),
                "--output-dir",
                str(teacher_root),
                "--dataset",
                "infoquest",
                "--split",
                "train",
                "--rollouts-per-example",
                str(selected_config.get("distillation", {}).get("rollouts_per_example", 4)),
                "--shard-count",
                str(shard_count),
                "--aggregate-only",
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_student_sft:
        cmd = [
            py,
            "scripts/run_student_sft.py",
            "--config",
            str(selected_config_path),
            "--train-corpus",
            str(teacher_root / "sft_corpus.jsonl"),
            "--output-dir",
            str(student_sft_root),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        _run(cmd, cwd=repo_root, dry_run=args.dry_run)

    checkpoint_path = None
    if not args.dry_run and (student_sft_root / "student_sft_manifest.json").exists():
        checkpoint_path = read_json(student_sft_root / "student_sft_manifest.json")["checkpoint_path"]

    if checkpoint_path and not args.skip_student_dpo and (teacher_root / "preference_corpus.jsonl").exists():
        cmd = [
            py,
            "scripts/run_student_dpo.py",
            "--config",
            str(selected_config_path),
            "--preference-corpus",
            str(teacher_root / "preference_corpus.jsonl"),
            "--sft-checkpoint",
            checkpoint_path,
            "--output-dir",
            str(student_dpo_root),
            "--prepare-only",
        ]
        _run(cmd, cwd=repo_root, dry_run=args.dry_run)

    if checkpoint_path and not args.skip_student_eval:
        cmd = [
            py,
            "scripts/run_student_eval.py",
            "--config",
            str(selected_config_path),
            "--sft-checkpoint",
            checkpoint_path,
            "--output-dir",
            str(student_eval_root),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        _run(cmd, cwd=repo_root, dry_run=args.dry_run)

    if not args.skip_paper:
        _run(
            [
                py,
                "scripts/paper_tables.py",
                "--matrix-root",
                str(baseline_root),
                "--ablation-root",
                str(ablation_root),
                "--student-root",
                str(student_eval_root),
                "--output-dir",
                "paper/generated",
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )
        if shutil.which("pdflatex") is not None:
            _run([py, "scripts/build_paper.py"], cwd=repo_root, dry_run=args.dry_run)

    write_json(
        {
            "selected_config": str(selected_config_path),
            "dev_sweep_selection": str(best_config_path) if not args.dry_run and not args.skip_dev_sweep else None,
            "baseline_root": str(baseline_root),
            "ablation_root": str(ablation_root),
            "teacher_rollout_root": str(teacher_root),
            "student_sft_root": str(student_sft_root),
            "student_dpo_root": str(student_dpo_root),
            "student_eval_root": str(student_eval_root),
            "audit_root": str(audit_root),
            "calibrator_path": str(calibrator_path),
        },
        output_root / "research_pipeline_manifest.json",
    )
    print(f"Research pipeline artifacts rooted at {output_root}")


if __name__ == "__main__":
    main()
