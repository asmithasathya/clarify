import csv
import json
import sys

import pytest

from scripts.aggregate_baseline_matrix import aggregate_matrix
from scripts.audit_agreement import _cohen_kappa
from scripts.export_audit import _stratified_sample
from scripts.plan_jobs import main as plan_jobs_main
from scripts.paper_tables import main as paper_tables_main
from scripts.run_research_pipeline import main as run_research_pipeline_main
from scripts.run_report_pipeline import main as run_report_pipeline_main
from src.utils.config import load_config
from src.data.report_data import validate_examples
from src.data.schema import DialogueExample
from src.eval.runner import build_run_manifest, get_judge_model_id, get_task_model_id


def test_validate_examples_rejects_duplicates():
    duplicate = DialogueExample(
        example_id="dup-1",
        dataset_name="clarifybench",
        split_name="test",
        user_request="Help me decide",
        hidden_context="Some hidden context",
        gold_clarification_needed=True,
    )
    with pytest.raises(ValueError):
        validate_examples([duplicate, duplicate], expected_dataset="clarifybench", require_split=True)


def test_report_manifest_uses_separate_judge():
    config = {
        "data": {"primary_dataset": "clarifybench", "split": "test", "clarifybench": {"local_path": "data/clarifybench_v1_full.jsonl"}},
        "model": {"generator": {"backend": "tinker", "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507"}},
        "project": {"seed": 42, "prompt_version": "baseline-v1"},
        "evaluation": {"llm_judge": {"enabled": True, "base_model": "meta-llama/Llama-3.3-70B-Instruct", "version": "judge-v1"}},
    }
    example = DialogueExample(
        example_id="clarifybench-v1-001",
        dataset_name="clarifybench",
        split_name="test",
        user_request="Help me decide",
        hidden_context="Some hidden context",
        gold_clarification_needed=True,
    )

    manifest = build_run_manifest(config, [example], ["targeted_clarify"])
    assert get_task_model_id(config) == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert get_judge_model_id(config) == "meta-llama/Llama-3.3-70B-Instruct"
    assert manifest["judge_model_id"] != manifest["task_model_id"]


def test_nested_config_includes_are_resolved():
    config = load_config("configs/report_baselines.yaml")
    assert config["model"]["generator"]["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert config["evaluation"]["llm_judge"]["base_model"] == "meta-llama/Llama-3.3-70B-Instruct"


def test_stratified_audit_sample_hits_target():
    rows = []
    for idx in range(120):
        rows.append(
            {
                "example_id": f"example-{idx:03d}",
                "dataset_name": "clarifybench" if idx % 2 else "infoquest",
                "task_model_id": "model-a" if idx % 3 else "model-b",
                "method": "targeted_clarify" if idx % 4 else "generic_clarify",
                "correct": bool(idx % 5),
                "asked_clarification": bool(idx % 2),
            }
        )
    sample = _stratified_sample(rows, 100)
    assert len(sample) == 100


def test_cohen_kappa_matches_identity():
    assert _cohen_kappa([(1, 1), (0, 0), (1, 1), (0, 0)]) == 1.0


def test_paper_tables_generation(tmp_path, monkeypatch):
    matrix_root = tmp_path / "matrix"
    matrix_root.mkdir()
    with (matrix_root / "matrix_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "model_label",
                "method",
                "task_success_rate",
                "appropriate_action_rate",
                "final_answer_rate",
                "average_answer_score",
                "average_clarification_quality",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "clarifybench",
                "model_label": "qwen3_30b",
                "method": "targeted_clarify",
                "task_success_rate": "0.9",
                "appropriate_action_rate": "1.0",
                "final_answer_rate": "0.9",
                "average_answer_score": "0.8",
                "average_clarification_quality": "0.7",
            }
        )
    with (matrix_root / "repeatability.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "model_label", "method", "slice_size", "drift_count", "max_delta"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "clarifybench",
                "model_label": "qwen3_30b",
                "method": "targeted_clarify",
                "slice_size": "12",
                "drift_count": "0",
                "max_delta": "0.0",
            }
        )
    with (matrix_root / "matrix_metrics.csv").open("a", encoding="utf-8", newline="") as handle:
        pass

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (data_dir / "clarifybench_v1_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "test": {
                    "n_examples": 90,
                    "by_ambiguity_type": {"lexical": 20},
                    "by_domain": {"travel": 18},
                }
            },
            handle,
        )
    student_root = tmp_path / "student_eval"
    student_root.mkdir()
    with (student_root / "student_eval_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "model_label",
                "method",
                "task_success_rate",
                "appropriate_action_rate",
                "mean_intent_confidence",
                "avg_task_model_calls",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "infoquest",
                "model_label": "student_sft",
                "method": "resample_clarify",
                "task_success_rate": "0.7",
                "appropriate_action_rate": "0.8",
                "mean_intent_confidence": "0.75",
                "avg_task_model_calls": "6.0",
            }
        )

    output_dir = tmp_path / "paper_generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper_tables.py",
            "--matrix-root",
            str(matrix_root),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--student-root",
            str(student_root),
        ],
    )
    paper_tables_main()

    assert (output_dir / "main_results.tex").exists()
    assert (output_dir / "dataset_stats.tex").exists()
    assert (output_dir / "student_results.tex").exists()
    assert (output_dir / "efficiency.tex").exists()


def test_report_pipeline_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_report_pipeline.py",
            "--config",
            "configs/report_baselines.yaml",
            "--output-root",
            str(tmp_path / "pipeline"),
            "--dry-run",
        ],
    )
    run_report_pipeline_main()
    captured = capsys.readouterr()
    assert "prepare-data" in captured.out
    assert "baseline" in captured.out
    assert "paper-tables" in captured.out


def test_research_pipeline_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_pipeline.py",
            "--config",
            "configs/report_baselines.yaml",
            "--output-root",
            str(tmp_path / "research"),
            "--dry-run",
        ],
    )
    run_research_pipeline_main()
    captured = capsys.readouterr()
    assert "run_dev_sweep.py" in captured.out
    assert "run_baseline_matrix.py" in captured.out
    assert "run_student_sft.py" in captured.out


def test_plan_jobs_uses_manifest_reader_not_jq(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_jobs.py",
            "--config",
            "configs/report_baselines.yaml",
            "--phase",
            "phase2",
            "--output-root",
            "outputs/research_pipeline",
        ],
    )
    plan_jobs_main()
    captured = capsys.readouterr()
    assert "read_manifest_value.py" in captured.out
    assert "jq" not in captured.out


def test_aggregate_baseline_matrix_tracks_incomplete_leaves(tmp_path):
    config = {
        "report": {
            "datasets": [
                {"name": "infoquest", "split": "test"},
                {"name": "clarifybench", "split": "test"},
            ],
            "task_models": [
                {"label": "qwen3_4b", "base_model": "Qwen/Qwen3-4B-Instruct-2507"},
                {"label": "qwen3_30b", "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
            ],
            "judge_model": {"base_model": "meta-llama/Llama-3.3-70B-Instruct"},
        }
    }
    leaf_root = tmp_path / "infoquest" / "qwen3_4b"
    leaf_root.mkdir(parents=True)
    with (leaf_root / "all_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "targeted_clarify": {
                    "task_success_rate": 0.8,
                    "appropriate_action_rate": 0.9,
                    "clarification_precision": 0.9,
                    "clarification_recall": 0.8,
                    "clarification_rate": 0.7,
                    "answer_rate": 0.3,
                    "final_answer_rate": 1.0,
                    "multi_turn_completion_rate": 0.7,
                    "missed_ambiguity_rate": 0.1,
                    "unnecessary_clarification_rate": 0.0,
                    "average_answer_score": 0.75,
                    "average_clarification_quality": 0.6,
                    "average_alternatives_quality": 0.0,
                }
            },
            handle,
        )
    sharded_leaf = tmp_path / "clarifybench" / "qwen3_30b"
    sharded_leaf.mkdir(parents=True)
    with (sharded_leaf / "sharding.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "planned"}, handle)

    payload = aggregate_matrix(config, output_root=tmp_path)

    assert payload["completed_leaf_count"] == 1
    assert len(payload["incomplete"]) == 3
    assert (tmp_path / "matrix_metrics.csv").exists()
    assert (tmp_path / "matrix_incomplete.json").exists()
