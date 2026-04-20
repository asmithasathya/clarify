PYTHON ?= ./.venv/bin/python

.PHONY: install install-dev test demo eval download prepare-data build-clarifybench preflight baseline ablation audit-export paper-tables paper pipeline dev-sweep teacher-rollouts student-sft student-dpo student-eval research-pipeline job-plan legacy-pipeline

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

download:
	$(PYTHON) -m scripts.download_infoquest

prepare-data:
	$(PYTHON) -m scripts.prepare_data

build-clarifybench:
	$(PYTHON) -m scripts.build_clarifybench

preflight:
	$(PYTHON) -m scripts.preflight

demo:
	$(PYTHON) -m scripts.demo_example

eval:
	$(PYTHON) scripts/run_eval.py --methods direct_answer generic_hedge generic_clarify targeted_clarify resample_clarify

dev-sweep:
	$(PYTHON) scripts/run_dev_sweep.py --config configs/report_baselines.yaml --output-dir outputs/dev_sweep

baseline:
	$(PYTHON) scripts/run_baseline_matrix.py --config configs/report_baselines.yaml

ablation:
	$(PYTHON) scripts/run_ablation.py --config configs/report_baselines.yaml --dataset infoquest --split test --method resample_clarify --ablation single_sample_only --ablation no_selective_resample --ablation resample_all_stages --ablation one_round_only --ablation clarify_immediately --ablation no_clarification_fallback --ablation no_calibration --ablation generic_question_fallback

teacher-rollouts:
	$(PYTHON) scripts/export_teacher_rollouts.py --config configs/report_baselines.yaml --dataset infoquest --split train --output-dir outputs/teacher_rollouts

student-sft:
	$(PYTHON) scripts/run_student_sft.py --config configs/report_baselines.yaml --train-corpus $${TRAIN_CORPUS:?set TRAIN_CORPUS} --output-dir outputs/student_sft

student-dpo:
	$(PYTHON) scripts/run_student_dpo.py --config configs/report_baselines.yaml --preference-corpus $${PREF_CORPUS:?set PREF_CORPUS} --sft-checkpoint $${SFT_CHECKPOINT:?set SFT_CHECKPOINT} --output-dir outputs/student_dpo --prepare-only

student-eval:
	$(PYTHON) scripts/run_student_eval.py --config configs/report_baselines.yaml --sft-checkpoint $${SFT_CHECKPOINT:?set SFT_CHECKPOINT} --output-dir outputs/student_eval

job-plan:
	$(PYTHON) scripts/plan_jobs.py --config configs/report_baselines.yaml --phase all --output-root outputs/research_pipeline

audit-export:
	$(PYTHON) scripts/export_audit.py --matrix-root $${MATRIX_ROOT:?set MATRIX_ROOT}

paper-tables:
	$(PYTHON) scripts/paper_tables.py --matrix-root $${MATRIX_ROOT:?set MATRIX_ROOT} $${ABLATION_ROOT:+--ablation-root $$ABLATION_ROOT} $${STUDENT_ROOT:+--student-root $$STUDENT_ROOT} $${AGREEMENT_JSON:+--agreement-json $$AGREEMENT_JSON}

paper:
	$(PYTHON) scripts/build_paper.py

pipeline:
	$(PYTHON) scripts/run_research_pipeline.py

research-pipeline:
	$(PYTHON) scripts/run_research_pipeline.py

legacy-pipeline:
	$(PYTHON) scripts/run_report_pipeline.py
