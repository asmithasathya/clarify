"""Generate LaTeX tables for the manuscript from experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import ensure_dir, read_csv, read_json


def _latex_escape(text: object) -> str:
    value = str(text)
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _summarize_counter(counter_map: dict[str, int], limit: int = 5) -> str:
    def _compact_label(label: str, width: int = 24) -> str:
        if len(label) <= width:
            return label
        return label[: width - 3] + "..."

    items = sorted(counter_map.items(), key=lambda item: (-item[1], item[0]))
    head = ", ".join(f"{_compact_label(key)}:{value}" for key, value in items[:limit])
    if len(items) <= limit:
        return head
    return f"{head}, +{len(items) - limit} more"


def _write_table(
    path: Path,
    headers: list[str],
    rows: list[list[object]],
    caption: str,
    label: str,
    *,
    column_spec: str | None = None,
) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        "\\begin{tabular}{%s}" % (column_spec or ("l" * len(headers))),
        "\\hline",
        " & ".join(_latex_escape(header) for header in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(cell) for cell in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _placeholder_table(path: Path, caption: str, label: str, message: str) -> None:
    _write_table(path, ["Status"], [[message]], caption, label, column_spec="p{0.82\\linewidth}")


def _has_complete_pivot_matrix(rows: list[dict[str, object]]) -> bool:
    if not rows:
        return False
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        model = str(row.get("model_label", ""))
        method = str(row.get("method", ""))
        grouped.setdefault((dataset, model), set()).add(method)
    if not grouped:
        return False
    return all("resample_clarify" in methods and len(methods) >= 5 for methods in grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript tables from baseline outputs.")
    parser.add_argument("--matrix-root", required=True, help="Root directory produced by run_baseline_matrix.py.")
    parser.add_argument("--data-dir", default="data", help="Directory containing dataset stats JSON files.")
    parser.add_argument("--output-dir", default="paper/generated", help="Directory to write .tex table fragments.")
    parser.add_argument("--ablation-root", default=None, help="Optional ablation output root.")
    parser.add_argument("--student-root", default=None, help="Optional student evaluation output root.")
    parser.add_argument("--agreement-json", default=None, help="Optional audit agreement JSON.")
    args = parser.parse_args()

    matrix_root = Path(args.matrix_root)
    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(args.output_dir)

    matrix_csv = matrix_root / "matrix_metrics.csv"
    if matrix_csv.exists():
        rows = read_csv(matrix_csv)
        if _has_complete_pivot_matrix(rows):
            main_rows = [
                [
                    row["dataset"],
                    row["model_label"],
                    row["method"],
                    f"{float(row['task_success_rate']):.3f}",
                    f"{float(row['appropriate_action_rate']):.3f}",
                    f"{float(row.get('mean_intent_confidence', 0.0)):.3f}",
                    f"{float(row.get('avg_task_model_calls', 0.0)):.2f}",
                    f"{float(row.get('avg_estimated_cost', 0.0)):.4f}",
                ]
                for row in rows
            ]
            _write_table(
                output_dir / "main_results.tex",
                ["Dataset", "Model", "Method", "TSR", "AAR", "IntentConf", "Calls", "Cost"],
                main_rows,
                "Main intent-stabilization baseline results across datasets, models, and methods.",
                "tab:main-results",
            )
        else:
            _placeholder_table(
                output_dir / "main_results.tex",
                "Main intent-stabilization baseline results across datasets, models, and methods.",
                "tab:main-results",
                "Pending a fresh pivot-compatible matrix with all five methods, including resample_clarify. Legacy clarification-only baselines are retained for reference only.",
            )
    else:
        _placeholder_table(
            output_dir / "main_results.tex",
            "Main intent-stabilization baseline results across datasets, models, and methods.",
            "tab:main-results",
            "Pending completion of the frozen baseline matrix.",
        )

    dataset_rows: list[list[object]] = []
    for stats_name in ("infoquest_stats.json", "clarifybench_v1_stats.json"):
        stats_path = data_dir / stats_name
        if not stats_path.exists():
            continue
        stats = read_json(stats_path)
        for split_name, split_stats in stats.items():
            dataset_rows.append(
                [
                    stats_name.replace("_stats.json", ""),
                    split_name,
                    split_stats["n_examples"],
                    _summarize_counter(split_stats["by_ambiguity_type"]),
                    _summarize_counter(split_stats["by_domain"]),
                ]
            )
    if dataset_rows:
        _write_table(
            output_dir / "dataset_stats.tex",
            ["Dataset", "Split", "N", "Ambiguity", "Domains"],
            dataset_rows,
            "Dataset composition for frozen report runs.",
            "tab:dataset-stats",
            column_spec="lllp{0.26\\linewidth}p{0.26\\linewidth}",
        )
    else:
        _placeholder_table(
            output_dir / "dataset_stats.tex",
            "Dataset composition for frozen report runs.",
            "tab:dataset-stats",
            "Pending completion of the frozen data-preparation workflow.",
        )

    repeatability_csv = matrix_root / "repeatability.csv"
    if repeatability_csv.exists():
        rows = read_csv(repeatability_csv)
        repeat_rows = [
            [
                row["dataset"],
                row["model_label"],
                row["method"],
                row["slice_size"],
                row["drift_count"],
                f"{float(row['max_delta']):.3f}",
            ]
            for row in rows
        ]
        _write_table(
            output_dir / "repeatability.tex",
            ["Dataset", "Model", "Method", "Slice", "Drift", "MaxDelta"],
            repeat_rows,
            "Repeatability check over a fixed 10\\% slice.",
            "tab:repeatability",
        )
    else:
        _placeholder_table(
            output_dir / "repeatability.tex",
            "Repeatability check over a fixed 10\\% slice.",
            "tab:repeatability",
            "Pending completion of the key-claim repeatability runs.",
        )

    if args.ablation_root:
        ablation_rows: list[list[object]] = []
        for summary_path in sorted(Path(args.ablation_root).rglob("summary.csv")):
            rows = read_csv(summary_path)
            if not rows:
                continue
            row = rows[0]
            ablation_rows.append(
                [
                    summary_path.parent.parent.name,
                    summary_path.parent.name,
                    f"{float(row.get('task_success_rate', 0.0)):.3f}",
                    f"{float(row.get('appropriate_action_rate', 0.0)):.3f}",
                    f"{float(row.get('mean_intent_confidence', 0.0)):.3f}",
                    f"{float(row.get('avg_task_model_calls', 0.0)):.2f}",
                ]
            )
        if ablation_rows:
            _write_table(
                output_dir / "ablations.tex",
                ["Ablation", "Method", "TSR", "AAR", "IntentConf", "Calls"],
                ablation_rows,
                "Selective-resampling ablations on the primary teacher model.",
                "tab:ablations",
            )
        else:
            _placeholder_table(
                output_dir / "ablations.tex",
                "Selective-resampling ablations on the primary teacher model.",
                "tab:ablations",
                "Pending completion of the selective-resampling ablation sweep.",
            )
    else:
        _placeholder_table(
            output_dir / "ablations.tex",
            "Selective-resampling ablations on the primary teacher model.",
            "tab:ablations",
            "Pending completion of the selective-resampling ablation sweep.",
        )

    if args.student_root:
        student_csv = Path(args.student_root) / "student_eval_metrics.csv"
        if student_csv.exists():
            rows = read_csv(student_csv)
            student_rows = [
                [
                    row["dataset"],
                    row["model_label"],
                    row["method"],
                    f"{float(row.get('task_success_rate', 0.0)):.3f}",
                    f"{float(row.get('appropriate_action_rate', 0.0)):.3f}",
                    f"{float(row.get('mean_intent_confidence', 0.0)):.3f}",
                    f"{float(row.get('avg_task_model_calls', 0.0)):.2f}",
                ]
                for row in rows
            ]
            _write_table(
                output_dir / "student_results.tex",
                ["Dataset", "System", "Method", "TSR", "AAR", "IntentConf", "Calls"],
                student_rows,
                "Student-model comparisons for the modular distillation stage.",
                "tab:student-results",
            )
        else:
            _placeholder_table(
                output_dir / "student_results.tex",
                "Student-model comparisons for the modular distillation stage.",
                "tab:student-results",
                "Pending teacher rollout export, LoRA SFT, and student evaluation.",
            )
    else:
        _placeholder_table(
            output_dir / "student_results.tex",
            "Student-model comparisons for the modular distillation stage.",
            "tab:student-results",
            "Pending teacher rollout export, LoRA SFT, and student evaluation.",
        )

    if matrix_csv.exists():
        rows = read_csv(matrix_csv)
        if _has_complete_pivot_matrix(rows):
            efficiency_rows = [
                [
                    row["dataset"],
                    row["model_label"],
                    row["method"],
                    f"{float(row.get('avg_task_model_calls', 0.0)):.2f}",
                    f"{float(row.get('avg_latency', 0.0)):.2f}",
                    f"{float(row.get('avg_estimated_cost', 0.0)):.4f}",
                    f"{float(row.get('resample_rate', 0.0)):.3f}",
                ]
                for row in rows
            ]
            _write_table(
                output_dir / "efficiency.tex",
                ["Dataset", "Model", "Method", "Calls", "Latency", "Cost", "Resample"],
                efficiency_rows,
                "Inference efficiency and resampling footprint for the main evaluation matrix.",
                "tab:efficiency",
            )
        else:
            _placeholder_table(
                output_dir / "efficiency.tex",
                "Inference efficiency and resampling footprint for the main evaluation matrix.",
                "tab:efficiency",
                "Pending a fresh pivot-compatible matrix. Legacy baseline artifacts do not contain the full resampling metrics needed for this table.",
            )
    else:
        _placeholder_table(
            output_dir / "efficiency.tex",
            "Inference efficiency and resampling footprint for the main evaluation matrix.",
            "tab:efficiency",
            "Pending completion of the frozen baseline matrix.",
        )

    if args.agreement_json and Path(args.agreement_json).exists():
        agreement = read_json(args.agreement_json)
        _write_table(
            output_dir / "audit_agreement.tex",
            ["Metric", "Value"],
            [
                ["Correct agreement", f"{float(agreement['correct_agreement']):.3f}"],
                ["Correct kappa", f"{float(agreement['correct_kappa']):.3f}"],
                ["Action agreement", f"{float(agreement['action_agreement']):.3f}"],
                ["Action kappa", f"{float(agreement['action_kappa']):.3f}"],
            ],
            "Judge-human agreement on the manual audit slice.",
            "tab:audit-agreement",
        )
    else:
        _placeholder_table(
            output_dir / "audit_agreement.tex",
            "Judge-human agreement on the manual audit slice.",
            "tab:audit-agreement",
            "Pending manual audit annotation and judge-human agreement scoring.",
        )

    roots = [f"matrix_root={matrix_root}"]
    if args.ablation_root:
        roots.append(f"ablation_root={Path(args.ablation_root)}")
    if args.student_root:
        roots.append(f"student_root={Path(args.student_root)}")
    if args.agreement_json:
        roots.append(f"agreement_json={Path(args.agreement_json)}")
    (output_dir / "generated_from.txt").write_text("\n".join(roots) + "\n", encoding="utf-8")
    print(f"Wrote manuscript tables to {output_dir}")


if __name__ == "__main__":
    main()
