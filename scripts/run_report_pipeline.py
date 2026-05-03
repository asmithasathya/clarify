"""Run the full baseline-ready report pipeline from one command."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import load_config
from src.utils.io import ensure_dir, write_yaml


ABLATIONS = [
    "no_ambiguity_detection",
    "no_intent_modeling",
    "no_strategy_selection",
    "no_targeted_question",
    "heuristic_ambiguity_detection",
    "heuristic_intent_modeling",
    "heuristic_strategy_selection",
]


def _timestamped_output_root(base_dir: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(base_dir) / f"report_pipeline_{timestamp}")


def _command_str(command: list[str]) -> str:
    return " ".join(command)


def _run_step(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
) -> dict[str, Any]:
    print(f"[step] {name}")
    print(f"  {_command_str(command)}")
    if dry_run:
        return {"name": name, "status": "dry_run", "command": command}
    subprocess.run(command, cwd=cwd, check=True)
    return {"name": name, "status": "completed", "command": command}


def _resolve_primary_ablation_model(config: dict[str, Any]) -> str:
    report_cfg = config.get("report", {})
    primary_label = report_cfg.get("primary_ablation_model")
    task_models = report_cfg.get("task_models", [])
    if not primary_label:
        return config["model"]["generator"]["base_model"]
    for model_spec in task_models:
        if model_spec.get("label") == primary_label:
            return model_spec["base_model"]
    raise ValueError(f"primary_ablation_model={primary_label!r} not found in report.task_models")


def _write_ablation_config(config: dict[str, Any], output_root: Path) -> Path:
    ablation_config = deepcopy(config)
    ablation_config["model"]["generator"]["base_model"] = _resolve_primary_ablation_model(config)
    ablation_config["model"]["generator"]["model_path"] = None
    path = output_root / "ablation_config.yaml"
    write_yaml(ablation_config, path)
    return path


def _run_optional_paper_build(
    *,
    python_executable: str,
    cwd: Path,
    dry_run: bool,
    strict: bool,
) -> dict[str, Any]:
    if shutil.which("pdflatex") is None:
        status = "skipped_missing_pdflatex"
        detail = "pdflatex is not installed or not on PATH."
        print(f"[warn] {detail}")
        if strict and not dry_run:
            raise RuntimeError(detail)
        return {"name": "paper", "status": status, "detail": detail}

    return _run_step(
        "paper",
        [python_executable, "scripts/build_paper.py"],
        cwd=cwd,
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full report pipeline from one command.")
    parser.add_argument("--config", default="configs/report_baselines.yaml", help="Report config path.")
    parser.add_argument("--output-root", default=None, help="Directory for pipeline-specific outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-run example limit for smoke runs.")
    parser.add_argument("--audit-sheet", default=None, help="Optional filled audit sheet to score agreement after export.")
    parser.add_argument("--skip-prepare-data", action="store_true", help="Skip dataset preparation.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the Tinker/environment preflight.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the model-matrix baseline run.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the primary ablation run.")
    parser.add_argument("--skip-audit-export", action="store_true", help="Skip audit pack export.")
    parser.add_argument("--skip-paper-tables", action="store_true", help="Skip manuscript table generation.")
    parser.add_argument("--skip-paper-build", action="store_true", help="Skip LaTeX manuscript build.")
    parser.add_argument("--strict-paper-build", action="store_true", help="Fail if pdflatex is unavailable instead of warning.")
    parser.add_argument("--dry-run", action="store_true", help="Print the pipeline steps without executing them.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_executable = sys.executable
    config = load_config(args.config)
    report_cfg = config.get("report", {})

    output_root = (
        ensure_dir(args.output_root)
        if args.output_root
        else _timestamped_output_root(Path(config["paths"].get("outputs_dir", "outputs")))
    )
    baseline_root = ensure_dir(output_root / "baseline_matrix")
    ablation_root = ensure_dir(output_root / "ablations")
    audit_root = ensure_dir(output_root / "audit")
    paper_generated_dir = repo_root / "paper" / "generated"
    agreement_output = audit_root / "agreement.json"

    ablation_config_path = _write_ablation_config(config, output_root)
    summary: dict[str, Any] = {
        "config": str(Path(args.config)),
        "output_root": str(output_root),
        "baseline_root": str(baseline_root),
        "ablation_root": str(ablation_root),
        "audit_root": str(audit_root),
        "paper_generated_dir": str(paper_generated_dir),
        "steps": [],
    }

    if not args.skip_prepare_data:
        command = [python_executable, "scripts/prepare_data.py"]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        summary["steps"].append(
            _run_step("prepare-data", command, cwd=repo_root, dry_run=args.dry_run)
        )

    summary["steps"].append(
        _run_step(
            "validate-data",
            [
                python_executable,
                "scripts/validate_data.py",
                "--manifest",
                "data/infoquest_manifest.json",
                "--manifest",
                "data/clarifybench_v1_manifest.json",
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )
    )

    if not args.skip_preflight:
        summary["steps"].append(
            _run_step(
                "preflight",
                [
                    python_executable,
                    "scripts/preflight.py",
                    "--config",
                    args.config,
                    "--output-dir",
                    str(output_root),
                ],
                cwd=repo_root,
                dry_run=args.dry_run,
            )
        )

    if not args.skip_baseline:
        command = [
            python_executable,
            "scripts/run_baseline_matrix.py",
            "--config",
            args.config,
            "--output-dir",
            str(baseline_root),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        summary["steps"].append(
            _run_step("baseline", command, cwd=repo_root, dry_run=args.dry_run)
        )

    if not args.skip_ablation:
        command = [
            python_executable,
            "scripts/run_ablation.py",
            "--config",
            str(ablation_config_path),
            "--dataset",
            "infoquest",
            "--split",
            "test",
            "--output-dir",
            str(ablation_root),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        for ablation in ABLATIONS:
            command.extend(["--ablation", ablation])
        summary["steps"].append(
            _run_step("ablation", command, cwd=repo_root, dry_run=args.dry_run)
        )

    if not args.skip_audit_export:
        command = [
            python_executable,
            "scripts/export_audit.py",
            "--matrix-root",
            str(baseline_root),
            "--target-size",
            str(report_cfg.get("audit_target_size", 100)),
            "--output-dir",
            str(audit_root),
        ]
        summary["steps"].append(
            _run_step("audit-export", command, cwd=repo_root, dry_run=args.dry_run)
        )
        if args.audit_sheet:
            summary["steps"].append(
                _run_step(
                    "audit-agreement",
                    [
                        python_executable,
                        "scripts/audit_agreement.py",
                        "--audit-sheet",
                        args.audit_sheet,
                        "--audit-key",
                        str(audit_root / "audit_key.csv"),
                        "--output",
                        str(agreement_output),
                    ],
                    cwd=repo_root,
                    dry_run=args.dry_run,
                )
            )

    if not args.skip_paper_tables:
        command = [
            python_executable,
            "scripts/paper_tables.py",
            "--matrix-root",
            str(baseline_root),
            "--data-dir",
            "data",
            "--output-dir",
            str(paper_generated_dir),
            "--ablation-root",
            str(ablation_root),
        ]
        if args.audit_sheet:
            command.extend(["--agreement-json", str(agreement_output)])
        summary["steps"].append(
            _run_step("paper-tables", command, cwd=repo_root, dry_run=args.dry_run)
        )

    if not args.skip_paper_build:
        summary["steps"].append(
            _run_optional_paper_build(
                python_executable=python_executable,
                cwd=repo_root,
                dry_run=args.dry_run,
                strict=args.strict_paper_build,
            )
        )

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    (output_root / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Pipeline complete. Summary written to {output_root / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
