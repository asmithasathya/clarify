# Manuscript Workflow

The paper is meant to stay synchronized with experiment artifacts.

## Generate Tables

```bash
./.venv/bin/python scripts/paper_tables.py \
  --matrix-root outputs/baselines/<run_id> \
  --ablation-root outputs/ablations \
  --student-root outputs/student_eval \
  --agreement-json outputs/audit/agreement.json \
  --output-dir paper/generated
```

Generated files include:

- `main_results.tex`
- `dataset_stats.tex`
- `repeatability.tex`
- `ablations.tex`
- `student_results.tex`
- `efficiency.tex`
- `audit_agreement.tex`

## Build The PDF

```bash
./.venv/bin/python scripts/build_paper.py
```

This requires `pdflatex` on `PATH`.

## What The Paper Now Says

The manuscript is centered on:

- intent stabilization via selective resampling
- clarification as a fallback rather than the only interesting behavior
- a baseline matrix that retains the older clarification policies
- resampling ablations
- modular student distillation
- efficiency and reproducibility

## Updating The Narrative

When experiment decisions change:

1. update `paper/experiment_log.md`
2. regenerate the tables
3. rebuild the paper

Do not copy numbers into the manuscript by hand.

## Placeholder Policy

The paper should never silently fall back to stale or incompatible artifacts.

- If the fresh pivot-compatible matrix is missing, `main_results.tex` and `efficiency.tex` should stay as explicit placeholder tables.
- If resampling ablations, student evaluation, repeatability, or audit agreement are unfinished, their generated tables should remain placeholders rather than disappear.
- Narrative text may mention the currently verified status, but the main claims should stay tied to completed artifact roots only.
