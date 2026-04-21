# OVERVIEW

This file explains the project from start to finish in plain language.

It is intentionally longer and more explicit than the README. The README is for quick orientation. This document is for understanding exactly:

- what the project is trying to prove
- what code and scripts exist
- what each phase does
- what can be reused versus what must be rerun
- how the paper, experiments, and artifacts fit together
- what is already complete
- what is still a placeholder

The goal is that someone could read this file, open the repo, and understand the full workflow without guessing.

---

# 1. What This Project Is

## 1.1 The core problem

Many assistant failures are not primarily failures of factual knowledge. They are failures of **intent understanding**.

Examples:

- The user asks for “the best school near me,” but the model does not know whether “best” means academic ranking, affordability, commute time, or safety.
- The user says “book a flight for Friday,” but the model does not know which Friday, from where, or with what constraints.
- The user says “should I upgrade,” but the model does not know which product, budget, or use case matters.

If the model answers too quickly, it often answers under the wrong hidden assumption.

If the model clarifies too aggressively, it turns every ambiguous request into a slow, annoying interaction.

So the real question is:

> Can we make a model work harder on understanding user intent before it responds, and ask a clarification question only when that extra effort still fails?

That is what this repository is built around.

## 1.2 The project’s main idea

The main method is called `resample_clarify`.

The idea is:

1. Sample multiple structured interpretations of a request.
2. Measure where those interpretations disagree.
3. Treat the disagreement as a signal of unstable intent understanding.
4. Resample only the weakest stage of understanding.
5. If the system still cannot stabilize intent, ask a clarification question.

So clarification is not the first move. It is the **fallback**.

## 1.3 What changed relative to the older version of the repo

The repo originally centered on a clarification benchmark:

- `direct_answer`
- `generic_hedge`
- `generic_clarify`
- `targeted_clarify`

That older work is still useful, because it established that ambiguity handling matters.

But the repo has now pivoted to a broader and stronger framing:

> intent stabilization via selective resampling

The older methods are now treated as **baselines**.

The new main method is:

- `resample_clarify`

The paper and presentation are written around that pivot.

---

# 2. What The Repo Contains

At a high level, the repo has five major layers.

## 2.1 Data layer

This defines datasets, frozen splits, and validation.

Important files:

- `src/data/infoquest.py`
- `src/data/clarifybench_v1.py`
- `src/data/report_data.py`
- `scripts/prepare_data.py`
- `scripts/validate_data.py`

## 2.2 Inference / LLM layer

This is the Tinker-backed generation layer and prompt/schema machinery.

Important files:

- `src/llm/generator.py`
- `src/llm/prompts.py`
- `src/llm/schemas.py`

## 2.3 Method layer

This contains the inference-time policies.

Important files:

- `src/methods/direct_answer.py`
- `src/methods/generic_hedge.py`
- `src/methods/generic_clarify.py`
- `src/methods/targeted_clarify.py`
- `src/methods/resample_clarify.py`

## 2.4 Evaluation layer

This runs experiments, computes metrics, aggregates shards, exports audits, and generates report artifacts.

Important files:

- `src/eval/runner.py`
- `src/eval/metrics.py`
- `src/eval/judge.py`
- `src/eval/sharding.py`
- `scripts/run_baseline_matrix.py`
- `scripts/run_sharded_leaf.py`
- `scripts/run_ablation.py`
- `scripts/export_audit.py`
- `scripts/audit_agreement.py`
- `scripts/paper_tables.py`

## 2.5 Training / post-training layer

This handles teacher rollout export, student LoRA SFT, and optional DPO preparation.

Important files:

- `src/train/sft.py`
- `src/train/distillation.py`
- `scripts/export_teacher_rollouts.py`
- `scripts/run_student_sft.py`
- `scripts/run_student_eval.py`
- `scripts/run_student_dpo.py`

---

# 3. The High-Level End-to-End Workflow

The complete workflow has three main research phases, plus a data phase before them.

## Phase 0: Freeze and validate data

Goal:

- make the datasets deterministic and local
- avoid live remote fetches for report runs
- define train/dev/test clearly

Outputs:

- frozen `InfoQuest` snapshot and splits
- frozen `ClarifyBench v1`
- manifests and dataset stats

## Phase 1: Validate the new teacher method

Goal:

- determine whether `resample_clarify` is actually better than the older targeted baseline

Substeps:

1. run a development sweep on `InfoQuest dev`
2. pick a promising `resample_clarify` configuration
3. fit a lightweight intent-confidence calibrator
4. run a fresh test-set gate on the main teacher model
5. run the new resampling ablations
6. optionally export a manual audit slice

This phase is the decision point.

If the new method does not hold up here, it is not worth paying for teacher rollouts and training.

## Phase 2: Distill the teacher into a smaller student

Goal:

- see how much of the teacher-wrapper gain can be transferred into a smaller model

Substeps:

1. export teacher rollouts over `InfoQuest train`
2. build an SFT corpus and preference corpus
3. run LoRA SFT on the 4B student
4. evaluate the student on held-out test sets

## Phase 3: Optional DPO refinement

Goal:

- see if preference-based post-training improves on the SFT student

This phase is optional and not required for the first paper-critical path.

---

# 4. The Data Phase In Detail

## 4.1 What datasets are used

There are two datasets in the active workflow.

### InfoQuest

Role:

- primary teacher-development dataset
- primary student-training dataset

Current split:

- snapshot: 1000 examples
- train: 600
- dev: 200
- test: 200

Important property:

- split assignment is grouped by `seed_id`
- related variants stay together

### ClarifyBench v1

Role:

- held-out generalization benchmark only

Current split:

- full: 120 examples
- dev: 30
- test: 90

Important property:

- it is intentionally **not** used for student training

## 4.2 Why the frozen split matters

The project uses a report pipeline. That means:

- numbers should be reproducible
- training and test data should be stable
- paper tables should not silently drift because the data source changed

That is why the workflow starts from local JSONL files instead of live remote downloads.

## 4.3 The scripts involved

### Prepare the data

```bash
./.venv/bin/python scripts/prepare_data.py
```

What it does:

- downloads or loads InfoQuest
- creates the deterministic train/dev/test split
- builds ClarifyBench v1
- writes manifests and stats

### Validate the data

```bash
./.venv/bin/python scripts/validate_data.py \
  --manifest data/infoquest_manifest.json \
  --manifest data/clarifybench_v1_manifest.json
```

What it checks:

- no duplicate example IDs
- expected dataset names
- required split labels
- manifest integrity

## 4.4 The main data artifacts

Important files:

- `data/infoquest_snapshot.jsonl`
- `data/infoquest_train.jsonl`
- `data/infoquest_dev.jsonl`
- `data/infoquest_test.jsonl`
- `data/clarifybench_v1_full.jsonl`
- `data/clarifybench_v1_dev.jsonl`
- `data/clarifybench_v1_test.jsonl`
- `data/infoquest_manifest.json`
- `data/clarifybench_v1_manifest.json`
- `data/infoquest_stats.json`
- `data/clarifybench_v1_stats.json`

---

# 5. The Model Setup

## 5.1 Inference backend

The active workflow is Tinker-first.

That means:

- no local GPU requirement
- no local `vllm` path in the active paper workflow
- no local `transformers` inference path in the active paper workflow

## 5.2 Main models in the report config

From `configs/report_baselines.yaml`:

- teacher task model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- student base model: `Qwen/Qwen3-4B-Instruct-2507`
- strong inference baseline: `openai/gpt-oss-120b`
- judge model: `meta-llama/Llama-3.3-70B-Instruct`

## 5.3 Why there is a separate judge model

The evaluation uses an LLM judge.

The task model generates the response.
The judge model scores the result.

This separation is important because:

- using the exact same model for both generation and judging is weaker evidence
- the paper wants the evaluation story to be more defensible

It is still not the same thing as full human evaluation, which is why the audit slice exists.

---

# 6. The Five Inference Methods

The main evaluation compares five methods.

## 6.1 `direct_answer`

Behavior:

- answer immediately
- no ambiguity handling

Purpose:

- hard lower baseline

## 6.2 `generic_hedge`

Behavior:

- answer directly
- use hedging language

Purpose:

- test whether “being cautious in wording” helps without true ambiguity handling

## 6.3 `generic_clarify`

Behavior:

- ask a generic clarification question

Purpose:

- test whether clarification alone helps, even without targeted reasoning

## 6.4 `targeted_clarify`

Behavior:

- detect ambiguity
- model interpretations
- choose a strategy
- generate a targeted clarification or another action

Purpose:

- strongest older clarification baseline

## 6.5 `resample_clarify`

Behavior:

1. sample multiple structured ambiguity assessments
2. sample multiple intent models
3. compute agreement
4. identify the weakest stage
5. selectively resample that stage
6. calibrate intent confidence
7. clarify only if instability persists

Purpose:

- main new method

---

# 7. How `resample_clarify` Works

This section is the conceptual heart of the repository.

## 7.1 Why sample multiple times?

A single model pass can look confident even when the underlying interpretation is unstable.

If several independently sampled structured outputs disagree, that is a useful signal that the model has not really locked onto user intent.

The method uses disagreement as signal.

## 7.2 What gets sampled?

At minimum:

- ambiguity assessments
- intent models

Depending on configuration and downstream behavior, strategy decisions and clarification-related stages can also be part of the pipeline.

## 7.3 What counts as a weak point?

A weak point is the stage with the lowest agreement.

Example:

- ambiguity samples agree strongly that the request is ambiguous
- intent samples disagree about the missing variable
- strategy samples mostly agree that clarification is needed

Then the weak point is the intent stage, not the ambiguity stage and not the strategy stage.

## 7.4 Why resample selectively?

Because rerunning the whole pipeline is expensive and unfocused.

Selective resampling tries to spend extra model calls only where uncertainty is concentrated.

That is the key efficiency idea.

## 7.5 What the confidence score means

This project’s confidence score is not “how true the answer is.”

It is closer to:

> how stable the model’s interpretation of the user request is

That is why the repo calls it intent confidence.

## 7.6 What the calibrator does

The raw agreement signals are useful but not directly report-ready.

The calibrator maps those raw features into:

- a calibrated score
- a low / medium / high confidence band

This lets the method expose:

- `intent_confidence`
- `confidence_band`
- `uncertainty_explanation`

## 7.7 Why clarification is a fallback

The method is trying to avoid two extremes:

- answer immediately under a bad assumption
- ask a clarification question before trying to stabilize understanding

So the logic is:

- first try to stabilize
- only then clarify if still unresolved

---

# 8. The Development Sweep

## 8.1 Why the dev sweep exists

The new method has hyperparameters:

- initial sample count
- repair sample count
- number of repair rounds
- clarification fallback threshold

These choices affect:

- task success
- action appropriateness
- call budget
- latency

So the repo runs a development sweep on `InfoQuest dev` first.

## 8.2 The current search grid

In `scripts/run_dev_sweep.py`, the search is over:

- `initial_samples in {3, 5, 7}`
- `max_rounds in {1, 2}`
- `clarification_fallback_threshold in {0.75, 0.80, 0.85}`

`repair_samples` is derived from `initial_samples`.

## 8.3 The baseline used in the sweep

The sweep compares `resample_clarify` against `targeted_clarify`.

The targeted baseline is run once and reused.

That matters because it avoids rerunning the older method for every sweep configuration.

## 8.4 The sweep gate

The dev selection logic prefers a setup that:

- improves task success by at least `+0.03`
- does not reduce appropriate action rate

Then, among winners, it prefers the cheaper one.

## 8.5 The key output files

Main outputs:

- `sweep_results.csv`
- `sweep_results.json`
- `best_config.json`
- `fallback_config.json`

Then the calibrator is fit from the chosen run:

```bash
./.venv/bin/python scripts/fit_intent_calibrator.py \
  --predictions outputs/dev_sweep/<best_run>/resample_clarify/predictions.jsonl \
  --output outputs/dev_sweep/intent_calibrator.json
```

## 8.6 Why the dev sweep is not the final result

Because it is still a tuning set.

A dev win is enough to justify the next phase.
It is not enough to claim success in the paper.

That is why the fresh test-set gate exists.

---

# 9. The Fresh Phase 1 Test Gate

This is the most important live decision point in the current workflow.

## 9.1 What the test gate is checking

The project wants to know whether the new method actually holds up on test data.

The main gate is:

- teacher model: `qwen3_30b`
- datasets: `InfoQuest test` and `ClarifyBench test`
- methods: at least the baseline set plus `resample_clarify`

## 9.2 Why this gate happens before training

Teacher rollouts and student training are expensive.

If `resample_clarify` does not beat or tie the older targeted method on the fresh test gate, then:

- teacher rollouts are not justified yet
- student distillation is probably not worth the spend yet

So the project is intentionally gated.

## 9.3 The acceptance logic

The working gate is:

- `resample_clarify` should beat `targeted_clarify` on `InfoQuest test`
- it should tie or win on `ClarifyBench test`
- it should not materially regress on `appropriate_action_rate`

If it fails this gate, the right move is to stop and rethink the method, not continue into training.

## 9.4 How this is run

The full matrix can be run in one command:

```bash
./.venv/bin/python scripts/run_baseline_matrix.py \
  --config configs/report_baselines.yaml
```

But in practice, the repo also supports sharded leaf execution:

```bash
./.venv/bin/python scripts/run_sharded_leaf.py ...
```

That is how large model runs are made practical.

---

# 10. The Ablation Suite

The ablation suite asks:

> which parts of `resample_clarify` are actually responsible for the gain?

## 10.1 Why ablations matter

Without ablations, a new method can look like a black box.

The ablation suite tests whether the gain comes from:

- multi-sampling itself
- selective repair
- calibration
- delayed clarification

## 10.2 The ablations currently defined

The main ablations are:

- `single_sample_only`
- `no_selective_resample`
- `resample_all_stages`
- `one_round_only`
- `clarify_immediately`
- `no_clarification_fallback`
- `no_calibration`
- `generic_question_fallback`

## 10.3 How to interpret them

Examples:

- `single_sample_only`
  - removes the multi-sample structure
  - tests whether sampling and agreement actually matter

- `no_selective_resample`
  - removes weak-point targeting
  - tests whether selective repair matters

- `clarify_immediately`
  - skips the stabilization attempt
  - tests whether “stabilize first, clarify later” is better than just clarifying as soon as ambiguity appears

- `no_calibration`
  - removes the confidence-calibration layer
  - tests whether the confidence band and thresholding are meaningful

## 10.4 Why the ablations are part of Phase 1

Because they are still about the teacher method itself.

Before training a student, the project should know:

- whether the teacher method works
- which components matter

---

# 11. The Audit And Repeatability Layers

These are not the main experiment, but they are important for the paper.

## 11.1 Manual audit

The repo exports a 100-example audit slice.

Purpose:

- sanity-check the judge-based evaluation
- inspect near-boundary and interesting cases
- compute judge-human agreement after annotation

Scripts:

```bash
./.venv/bin/python scripts/export_audit.py \
  --matrix-root outputs/baselines/<run_id> \
  --output-dir outputs/audit
```

Then, after humans annotate:

```bash
./.venv/bin/python scripts/audit_agreement.py \
  --audit-sheet outputs/audit/audit_sheet.csv \
  --audit-key outputs/audit/audit_key.csv \
  --output outputs/audit/agreement.json
```

## 11.2 Repeatability

The repo does not rerun the entire full matrix for repeatability.

Instead, it reruns only the key claims on a fixed 10% slice.

Purpose:

- check that the main claim is not just one lucky run
- avoid doubling cost for the full matrix

This is lower priority than the main gate, but still useful for the final paper.

---

# 12. Teacher Rollouts

Teacher rollouts are the bridge from inference-time method development to student training.

## 12.1 What a teacher rollout is

A teacher rollout is a full run of `resample_clarify` on a training example, storing:

- stage traces
- stabilized outputs
- final action/response
- optional rejected comparison for preference data

## 12.2 Why the rollouts matter

They are used to build:

- the SFT corpus
- the preference corpus

## 12.3 The main command

```bash
./.venv/bin/python scripts/export_teacher_rollouts.py \
  --config configs/report_baselines.yaml \
  --dataset infoquest \
  --split train \
  --output-dir outputs/teacher_rollouts
```

## 12.4 What gets written

Important outputs:

- `teacher_rollouts.jsonl`
- `sft_corpus.jsonl`
- `preference_corpus.jsonl`

## 12.5 Why sharding matters here

This can be expensive.

So the repo supports rollout sharding, just like it supports matrix sharding.

That is especially important when:

- the teacher model is large
- multiple rollouts per training example are requested

---

# 13. LoRA SFT For The Student

## 13.1 What the student is

The main student is the 4B model:

- `Qwen/Qwen3-4B-Instruct-2507`

## 13.2 What kind of training this is

The paper-critical path uses:

- LoRA
- supervised fine-tuning

It does **not** require full dense fine-tuning.

## 13.3 Why the training path is modular

The student is not being trained to replace the whole system end to end.

Instead, it is trained on the teacher’s structured stage outputs.

That means:

- the existing runtime architecture can stay the same
- the student can plug into the same method pipeline

## 13.4 The command

```bash
./.venv/bin/python scripts/run_student_sft.py \
  --config configs/report_baselines.yaml \
  --train-corpus outputs/teacher_rollouts/sft_corpus.jsonl \
  --output-dir outputs/student_sft
```

## 13.5 The important artifacts

Expected outputs include:

- `student_sft_manifest.json`
- `student_sft_history.jsonl`

The manifest contains the trained checkpoint path or alias.

To read it:

```bash
./.venv/bin/python scripts/read_manifest_value.py \
  --path outputs/student_sft/student_sft_manifest.json \
  --key checkpoint_path
```

---

# 14. Student Evaluation

Once the student exists, the repo compares several systems.

## 14.1 Main student comparisons

The intended evaluation includes:

- base `Qwen3-4B + targeted_clarify`
- base `Qwen3-4B + resample_clarify`
- distilled `Qwen3-4B-SFT + targeted_clarify`
- distilled `Qwen3-4B-SFT + resample_clarify`

## 14.2 Why both base and distilled versions matter

Because the paper wants to separate:

- the benefit of the wrapper itself
- the benefit of the training itself

## 14.3 The command

```bash
./.venv/bin/python scripts/run_student_eval.py \
  --config configs/report_baselines.yaml \
  --sft-checkpoint "$(./.venv/bin/python scripts/read_manifest_value.py --path outputs/student_sft/student_sft_manifest.json --key checkpoint_path)" \
  --output-dir outputs/student_eval
```

## 14.4 What success would look like

At a high level:

- the student should recover some meaningful fraction of the teacher-wrapper gain
- it should not completely collapse on ClarifyBench

---

# 15. Optional DPO

## 15.1 Why DPO is optional

Because the paper-critical path is already:

- large enough
- expensive enough
- scientifically meaningful without it

So DPO is not required to have a complete and defensible first paper.

## 15.2 What the repo currently supports

The repo prepares the preference data and DPO configuration.

Main command:

```bash
./.venv/bin/python scripts/run_student_dpo.py \
  --config configs/report_baselines.yaml \
  --preference-corpus outputs/teacher_rollouts/preference_corpus.jsonl \
  --sft-checkpoint "$(./.venv/bin/python scripts/read_manifest_value.py --path outputs/student_sft/student_sft_manifest.json --key checkpoint_path)" \
  --output-dir outputs/student_dpo \
  --prepare-only
```

That is enough to keep DPO in scope without making it a blocker.

---

# 16. The Paper Workflow

The paper is designed to be artifact-driven.

That means:

- the paper should be updated from experiment outputs
- numbers should not be copied in by hand
- incomplete phases should appear as explicit placeholders

## 16.1 Table generation

Main command:

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

## 16.2 PDF build

```bash
./.venv/bin/python scripts/build_paper.py
```

This requires `pdflatex`.

## 16.3 Placeholder policy

This is extremely important.

If a phase is unfinished:

- the corresponding table remains a placeholder
- it is not silently omitted
- it is not silently replaced with stale or incompatible artifacts

This makes the manuscript structurally complete even when some experiment roots are still pending.

---

# 17. The Presentation Workflow

There is also an HTML presentation in:

- `presentation/spotlight.html`

It now mirrors the same policy as the paper:

- verified artifacts are shown as verified
- legacy artifacts are labeled as legacy
- unfinished artifacts are shown as placeholders

So the deck can be shown safely before the entire pipeline is finished.

---

# 18. Efficient Execution Strategy

This repo supports both sequential and parallel execution.

## 18.1 The easy sequential option

Use:

```bash
./.venv/bin/python scripts/run_research_pipeline.py \
  --config configs/report_baselines.yaml
```

This orchestrates the full pipeline.

Pros:

- simple
- one command
- writes a unified manifest

Cons:

- slow
- may waste compute if you do not stop at the important gates

## 18.2 The better practical option

Use:

```bash
./.venv/bin/python scripts/plan_jobs.py \
  --config configs/report_baselines.yaml \
  --phase all \
  --output-root outputs/research_pipeline
```

This prints shard-safe commands that can be pasted into multiple terminals or `tmux`.

## 18.3 Why `tmux` matters

These runs are:

- long
- expensive
- resumable

So `tmux` is the right default, not a normal terminal.

## 18.4 Best practical operator strategy

If you want to minimize wasted compute:

1. finish the dev sweep
2. lock a selected config
3. run the fresh `qwen3_30b` gate first
4. run ablations in parallel
5. only if the gate passes, continue into teacher rollouts and student SFT

This is much better than running every phase blindly in sequence.

## 18.5 Sharding strategy

The repo is designed so expensive phases can be split.

For matrix runs:

- shard at the dataset/model leaf

For teacher rollouts:

- shard at rollout-export shards

After all shards complete:

- aggregate the outputs

This is the main way to make the project practical on remote inference.

---

# 19. What Can Be Reused And What Cannot

This is one of the most important practical sections.

## 19.1 Things that should usually be reused

- frozen data files
- manifests
- completed dev sweep runs
- fitted calibrators
- partially completed shard roots
- legacy baseline root for historical reference

## 19.2 Things that should not be reused as final pivot results

The old clarification-only matrix root should not be reused as the final main result for the new paper.

Why:

- it does not include `resample_clarify`
- it predates the pivoted evaluation framing
- it does not populate the new resampling-aware efficiency metrics

It is still useful as legacy context.

## 19.3 What is safe to reuse for training

Depending on workflow, legacy baseline artifacts can still help as a source of rejected or comparison behavior for preference data.

But that is different from using them as the final main table in the paper.

---

# 20. What Is Complete Right Now Versus Placeholder

This document is meant to stay honest.

## 20.1 Structurally complete now

The following are already implemented:

- frozen data workflow
- five-method inference stack
- `resample_clarify`
- development sweep
- calibrator fitting
- fresh matrix runner
- ablation runner
- audit export
- teacher rollout export
- student SFT scaffolding
- student evaluation
- DPO preparation
- sharding and aggregation
- paper generation
- presentation generation

## 20.2 Usually verified now

At the time of writing, the repo already has:

- the older clarification baseline matrix
- a first dev-set win for `resample_clarify`
- a placeholder-safe manuscript
- a placeholder-safe deck

## 20.3 Often still pending in live runs

Depending on where the current experiment session is:

- fresh pivot-compatible full matrix
- selective-resampling ablation completion
- teacher rollouts
- student SFT
- student evaluation
- audit agreement
- repeatability

Those should stay placeholders until the artifacts exist.

---

# 21. Recommended Reading Order For A New Contributor

If you are new to the repo, the fastest understanding path is:

1. `README.md`
2. this file: `OVERVIEW.MD`
3. `docs/end_to_end.md`
4. `docs/job_parallelism.md`
5. `paper/main.tex`
6. `presentation/spotlight.html`
7. `src/methods/resample_clarify.py`
8. `src/eval/runner.py`
9. `scripts/run_research_pipeline.py`

---

# 22. Recommended Order To Actually Run The Project

If the goal is to make progress quickly and rationally:

1. activate the environment and verify `TINKER_API_KEY`
2. reuse the frozen data
3. finish or resume the dev sweep
4. fit the calibrator
5. create the selected config
6. run the fresh `qwen3_30b` gate
7. run ablations in parallel
8. inspect the gate
9. only if it passes, start teacher rollouts
10. run student SFT
11. run student evaluation
12. refresh paper tables
13. rebuild the paper

That is the most rational order because it delays the expensive training phase until the new method has earned it.

---

# 23. Bottom Line

This project is not just “a clarification benchmark.”

It is now a full research pipeline for:

- modeling intent uncertainty
- stabilizing intent via selective resampling
- comparing that against older ambiguity-handling baselines
- optionally distilling the improved behavior into a smaller model
- and generating a paper and presentation directly from artifacts

The most important conceptual shift is this:

> clarification is not the only response to ambiguity, and it should not be the first response by default

The system should first try to understand the user better through structured, agreement-based reasoning.
Only if that fails should it ask the user to clarify.

That is the central idea of the repository.
