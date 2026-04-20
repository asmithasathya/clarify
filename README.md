# clarify

Research code for **intent stabilization via selective resampling**. The core idea is to make a model work harder on understanding user intent before it answers: sample multiple structured intent hypotheses, detect unstable stages, selectively resample the weak stage, and ask a clarification question only if intent confidence remains low.

## What The Repo Does

The repo now supports the full research workflow:

- frozen `InfoQuest` `train/dev/test` splits for development and post-training
- held-out `ClarifyBench v1` for generalization evaluation
- Tinker-backed inference across multiple task models plus a separate judge model
- five inference methods:
  - `direct_answer`
  - `generic_hedge`
  - `generic_clarify`
  - `targeted_clarify`
  - `resample_clarify`
- dev sweeps and confidence-calibrator fitting for the new method
- teacher rollout export for modular distillation
- LoRA SFT training scaffolding for a smaller student model
- optional DPO preparation
- sharded/resumable evaluation, audit export, and manuscript table generation

The current paper framing treats the first four methods as baselines and `resample_clarify` as the main method.

## Models

Default report models:

- teacher task model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- student base model: `Qwen/Qwen3-4B-Instruct-2507`
- strong inference baseline: `openai/gpt-oss-120b`
- judge model: `meta-llama/Llama-3.3-70B-Instruct`

All inference and training paths are Tinker-first. Local `vllm` / `transformers` backends are not part of the active workflow anymore.

## Installation

Python `3.11` is the supported runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./.venv/bin/pip install -e ".[dev]"
export TINKER_API_KEY=...
```

You can also place the key in `.env`. See `.env.example`.

## Data Artifacts

Frozen dataset files:

- `data/infoquest_snapshot.jsonl`
- `data/infoquest_train.jsonl`
- `data/infoquest_dev.jsonl`
- `data/infoquest_test.jsonl`
- `data/clarifybench_v1_full.jsonl`
- `data/clarifybench_v1_dev.jsonl`
- `data/clarifybench_v1_test.jsonl`

All report runs should start from those local files, not from a live remote fetch.

## Fast Start

Prepare and validate the data:

```bash
./.venv/bin/python scripts/prepare_data.py
./.venv/bin/python scripts/validate_data.py \
  --manifest data/infoquest_manifest.json \
  --manifest data/clarifybench_v1_manifest.json
```

Check the Tinker environment:

```bash
./.venv/bin/python scripts/preflight.py --config configs/report_baselines.yaml
```

Run a single method:

```bash
./.venv/bin/python scripts/run_inference.py \
  --method resample_clarify \
  --dataset clarifybench \
  --split test \
  --limit 8
```

## End-To-End Research Workflow

### 1. Dev Sweep For `resample_clarify`

This searches a small resampling grid on `InfoQuest dev`, compares against `targeted_clarify`, and writes `best_config.json`.

```bash
./.venv/bin/python scripts/run_dev_sweep.py \
  --config configs/report_baselines.yaml \
  --output-dir outputs/dev_sweep
```

Fit the intent-confidence calibrator from the selected dev run:

```bash
./.venv/bin/python scripts/fit_intent_calibrator.py \
  --predictions outputs/dev_sweep/<best_run>/resample_clarify/predictions.jsonl \
  --output outputs/dev_sweep/intent_calibrator.json
```

### 2. Frozen Baseline Matrix

Run the full matrix over the frozen test splits:

```bash
./.venv/bin/python scripts/run_baseline_matrix.py \
  --config configs/report_baselines.yaml
```

This runs all report methods across all report task models and datasets. It is resumable, and the sharded leaf workflow can be used when a single leaf is too slow.

### 3. Resample Ablations

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

### 4. Manual Audit Export

```bash
./.venv/bin/python scripts/export_audit.py \
  --matrix-root outputs/baselines/<run_id>
```

After annotation:

```bash
./.venv/bin/python scripts/audit_agreement.py \
  --audit-sheet outputs/baselines/<run_id>/audit/audit_sheet.csv \
  --audit-key outputs/baselines/<run_id>/audit/audit_key.csv \
  --output outputs/baselines/<run_id>/audit/agreement.json
```

### 5. Teacher Rollouts

```bash
./.venv/bin/python scripts/export_teacher_rollouts.py \
  --config configs/report_baselines.yaml \
  --dataset infoquest \
  --split train \
  --output-dir outputs/teacher_rollouts
```

This writes:

- `teacher_rollouts.jsonl`
- `sft_corpus.jsonl`
- `preference_corpus.jsonl`

### 6. Student LoRA SFT

```bash
./.venv/bin/python scripts/run_student_sft.py \
  --config configs/report_baselines.yaml \
  --train-corpus outputs/teacher_rollouts/sft_corpus.jsonl \
  --output-dir outputs/student_sft
```

### 7. Student Evaluation

```bash
./.venv/bin/python scripts/run_student_eval.py \
  --config configs/report_baselines.yaml \
  --sft-checkpoint "$(./.venv/bin/python scripts/read_manifest_value.py --path outputs/student_sft/student_sft_manifest.json --key checkpoint_path)" \
  --output-dir outputs/student_eval
```

### 8. Optional DPO Preparation

```bash
./.venv/bin/python scripts/run_student_dpo.py \
  --config configs/report_baselines.yaml \
  --preference-corpus outputs/teacher_rollouts/preference_corpus.jsonl \
  --sft-checkpoint "$(./.venv/bin/python scripts/read_manifest_value.py --path outputs/student_sft/student_sft_manifest.json --key checkpoint_path)" \
  --output-dir outputs/student_dpo \
  --prepare-only
```

The repository prepares the manifest and dataset for DPO. Full execution depends on an optional `tinker-cookbook` workflow that is not required for the first paper-critical path.

### 9. Paper Tables And Build

```bash
./.venv/bin/python scripts/paper_tables.py \
  --matrix-root outputs/baselines/<run_id> \
  --ablation-root outputs/ablations \
  --student-root outputs/student_eval \
  --agreement-json outputs/baselines/<run_id>/audit/agreement.json \
  --output-dir paper/generated

./.venv/bin/python scripts/build_paper.py
```

## One-Command Pipeline

For a resumable end-to-end orchestration:

```bash
./.venv/bin/python scripts/run_research_pipeline.py \
  --config configs/report_baselines.yaml
```

This script can prepare data, run the dev sweep, fit the calibrator, run the baseline matrix, launch ablations, export audit artifacts, export teacher rollouts, run student SFT, prepare DPO, run student eval, generate paper tables, and build the manuscript when `pdflatex` is available.

Use `--skip-*` flags to resume only the missing phases.

## Multi-Terminal Execution

To generate shard-safe commands for parallel execution:

```bash
./.venv/bin/python scripts/plan_jobs.py \
  --config configs/report_baselines.yaml \
  --phase all \
  --output-root outputs/research_pipeline
```

That script emits commands that can be pasted into multiple terminals or a `tmux` session. See `docs/job_parallelism.md` for recommended terminal allocation.

## Key Outputs

Single method:

- `predictions.jsonl`
- `metrics.json`
- `summary.csv`
- `case_studies.md`
- `manifest.json`

Baseline matrix:

- `matrix_metrics.csv`
- `matrix_metrics.json`
- `comparison_report.md`
- `repeatability.csv`
- nested per-dataset / per-model / per-method artifacts

Teacher rollouts:

- `teacher_rollouts.jsonl`
- `sft_corpus.jsonl`
- `preference_corpus.jsonl`
- `teacher_rollout_manifest.json`

Student training/eval:

- `student_sft_manifest.json`
- `student_sft_history.jsonl`
- `student_eval_metrics.csv`
- `student_eval_report.md`
- `student_dpo_manifest.json`

Paper:

- `paper/generated/*.tex`
- `paper/main.pdf`

## Main Docs

- [docs/end_to_end.md](docs/end_to_end.md)
- [docs/training.md](docs/training.md)
- [docs/job_parallelism.md](docs/job_parallelism.md)
- [docs/paper.md](docs/paper.md)

## Make Targets

```bash
make install-dev
make prepare-data
make preflight
make dev-sweep
make baseline
make ablation
make teacher-rollouts
make student-sft TRAIN_CORPUS=outputs/teacher_rollouts/sft_corpus.jsonl
make student-eval SFT_CHECKPOINT=...
make student-dpo PREF_CORPUS=outputs/teacher_rollouts/preference_corpus.jsonl SFT_CHECKPOINT=...
make job-plan
make research-pipeline
make paper-tables MATRIX_ROOT=outputs/baselines/<run_id> ABLATION_ROOT=outputs/ablations STUDENT_ROOT=outputs/student_eval
make paper
```

## Current Scope And Limits

- The active paper-critical path is inference plus LoRA SFT. DPO is optional and scaffolded, not required.
- The v1 method uses disagreement across structured samples rather than token-logprob tracing.
- Live Tinker runs still require `TINKER_API_KEY` to be visible to the current process.
- `pdflatex` is required to build the PDF manuscript.
- Legacy artifacts under `outputs/report_pipeline_20260414_123119` are retained for comparison, but they are not the final pivoted paper root.
