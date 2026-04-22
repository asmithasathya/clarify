# End-To-End Workflow

This is the shortest reproducible path for the pivoted project.

## 1. Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./.venv/bin/pip install -e ".[dev]"
export TINKER_API_KEY=...
```

## 2. Prepare Frozen Data

```bash
./.venv/bin/python scripts/prepare_data.py
./.venv/bin/python scripts/validate_data.py \
  --manifest data/infoquest_manifest.json \
  --manifest data/clarifybench_v1_manifest.json
```

This produces:

- `InfoQuest train/dev/test`
- `ClarifyBench v1 dev/test`
- manifests and stats used by report runs

## 3. Sweep And Calibrate `resample_clarify`

```bash
./.venv/bin/python scripts/run_dev_sweep.py \
  --config configs/report_baselines.yaml \
  --output-dir outputs/dev_sweep
```

Then fit the calibrator from the selected dev run:

```bash
./.venv/bin/python scripts/fit_intent_calibrator.py \
  --predictions outputs/dev_sweep/<best_run>/resample_clarify/predictions.jsonl \
  --output outputs/dev_sweep/intent_calibrator.json
```

## 4. Frozen Test Matrix

```bash
./.venv/bin/python scripts/run_baseline_matrix.py \
  --config configs/report_baselines.yaml \
  --output-dir outputs/baselines/<run_id>
```

This is the baseline section of the paper.

Current status note:

- the first fresh `InfoQuest test / qwen3_30b` gate for the original selective controller underperformed (`resample_clarify = 0.22`, `targeted_clarify = 0.26`, `generic_clarify = 0.275`)
- the active fix path is a simpler `resample_clarify` revision: full repair pass, generic clarification fallback, calibration used mainly for reporting
- rerun that fix into a new output root before starting Phase 2

## 5. Ablations

```bash
./.venv/bin/python scripts/run_ablation.py \
  --config configs/report_baselines.yaml \
  --dataset infoquest \
  --split test \
  --method resample_clarify \
  --ablation single_sample_only \
  --ablation no_selective_resample \
  --ablation resample_all_stages \
  --ablation one_round_only \
  --ablation clarify_immediately \
  --ablation no_clarification_fallback \
  --ablation no_calibration \
  --ablation generic_question_fallback
```

This is the mechanism section of the paper.

Interpretation guide for the current run:

- if the full-repair style ablations outperform the base controller, selective weak-point repair is likely the problem
- if generic-question fallback outperforms the base controller, targeted clarification generation is likely the problem
- if no-calibration barely changes the result, calibration is not the main bottleneck

## 6. Audit

```bash
./.venv/bin/python scripts/export_audit.py \
  --matrix-root outputs/baselines/<run_id> \
  --output-dir outputs/audit
```

After annotation:

```bash
./.venv/bin/python scripts/audit_agreement.py \
  --audit-sheet outputs/audit/audit_sheet.csv \
  --audit-key outputs/audit/audit_key.csv \
  --output outputs/audit/agreement.json
```

This is the judge-validation section of the paper.

## 7. Teacher Rollouts And Student SFT

```bash
./.venv/bin/python scripts/export_teacher_rollouts.py \
  --config configs/report_baselines.yaml \
  --dataset infoquest \
  --split train \
  --output-dir outputs/teacher_rollouts

./.venv/bin/python scripts/run_student_sft.py \
  --config configs/report_baselines.yaml \
  --train-corpus outputs/teacher_rollouts/sft_corpus.jsonl \
  --output-dir outputs/student_sft
```

Then evaluate the student:

```bash
./.venv/bin/python scripts/run_student_eval.py \
  --config configs/report_baselines.yaml \
  --sft-checkpoint "$(./.venv/bin/python scripts/read_manifest_value.py --path outputs/student_sft/student_sft_manifest.json --key checkpoint_path)" \
  --output-dir outputs/student_eval
```

## 8. Optional DPO

```bash
./.venv/bin/python scripts/run_student_dpo.py \
  --config configs/report_baselines.yaml \
  --preference-corpus outputs/teacher_rollouts/preference_corpus.jsonl \
  --sft-checkpoint "$(./.venv/bin/python scripts/read_manifest_value.py --path outputs/student_sft/student_sft_manifest.json --key checkpoint_path)" \
  --output-dir outputs/student_dpo \
  --prepare-only
```

This stage is optional in the current paper path.

## 9. Paper

```bash
./.venv/bin/python scripts/paper_tables.py \
  --matrix-root outputs/baselines/<run_id> \
  --ablation-root outputs/ablations \
  --student-root outputs/student_eval \
  --agreement-json outputs/audit/agreement.json \
  --output-dir paper/generated

./.venv/bin/python scripts/build_paper.py
```

## One-Command Alternative

```bash
./.venv/bin/python scripts/run_research_pipeline.py \
  --config configs/report_baselines.yaml
```

Use `--skip-*` flags to resume incomplete phases.
