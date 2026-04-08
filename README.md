# clarify

Research repository for studying when and how an assistant should clarify ambiguous user requests instead of answering prematurely.

## Overview

The repository has already been pivoted from legal QA verification to ambiguity-aware clarification. The central question is whether an assistant should:

1. answer immediately,
2. answer with hedging,
3. ask a generic clarification question, or
4. detect the specific ambiguity and respond with a targeted strategy.

The object of uncertainty here is the assistant's understanding of the user, not factual support from retrieved evidence.

## Current Pipeline

```text
User request
  -> ambiguity detection
  -> intent modeling
  -> strategy selection
  -> one of:
       - direct answer
       - targeted clarification question
       - narrowed answer under an explicit assumption
       - alternatives presentation
       - abstention
```

The current implementation is single-turn. The system does not yet consume a follow-up user answer and continue the dialogue.

## Implemented Methods

- `direct_answer`: answer immediately with no ambiguity handling.
- `generic_hedge`: answer directly with hedging language.
- `generic_clarify`: always ask a generic clarification question.
- `targeted_clarify`: detect ambiguity, model interpretations, choose a strategy, and act.

## Data and Schemas

The primary dataset is [InfoQuest](https://huggingface.co/datasets/bryanlincoln/infoquest). Each example is normalized into `DialogueExample`, which stores:

- the visible user request,
- hidden context,
- whether clarification is needed,
- an optional gold answer,
- ambiguity metadata,
- checklist and persona fields when available.

The main runtime output is `MethodResult`, which records:

- the chosen response strategy,
- the emitted response text,
- whether the method answered directly,
- whether it actually asked a clarification question,
- ambiguity-detection signals,
- evaluation flags and trace metadata.

## Evaluation

Current aggregate metrics:

- `task_success_rate`
- `appropriate_action_rate`
- `clarification_rate`
- `answer_rate`
- `abstention_rate`
- `clarification_precision`
- `clarification_recall`
- `ambiguity_detection_accuracy`
- `unnecessary_clarification_rate`
- `missed_ambiguity_rate`
- `wrong_answer_under_ambiguity`
- `strategy_distribution`

Current correctness is an offline proxy computed during evaluation in `src/eval/runner.py`:

- Final answers are scored with normalized text overlap against `gold_answer`.
- Clarification questions use `gold_clarifying_question` when available.
- If no gold clarification question exists, asking for clarification is treated as correct when `gold_clarification_needed` is true.
- `present_alternatives` is currently treated as an appropriate ambiguity-handling action and gets a proxy correctness signal when clarification is needed.

This is intentionally lightweight and should be treated as approximate open-ended evaluation, not a final benchmark design.

## Installation

Python 3.10+ is required.

Base install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional installs:

```bash
pip install -e ".[dev,vllm]"
pip install -e ".[dev,vllm,peft]"
```

## Quickstart

Smoke demo with a mock generator:

```bash
python -m scripts.demo_example
```

Download InfoQuest locally to JSONL:

```bash
python -m scripts.download_infoquest
```

Run one method:

```bash
python scripts/run_inference.py --method targeted_clarify --limit 100
```

Run full evaluation:

```bash
python scripts/run_eval.py --methods direct_answer generic_hedge generic_clarify targeted_clarify
```

Run ablations for `targeted_clarify`:

```bash
python scripts/run_ablation.py \
  --ablation no_ambiguity_detection \
  --ablation no_intent_modeling \
  --ablation no_strategy_selection \
  --ablation no_targeted_question
```

Switch to the Transformers backend:

```bash
python scripts/run_eval.py --methods targeted_clarify --backend transformers
```

Make targets:

```bash
make install-dev
make download
make demo
make eval
```

## Configuration

`configs/default.yaml` currently includes:

- `configs/model/qwen25_7b.yaml`
- `configs/method/targeted_clarify.yaml`

Key defaults:

- dataset: `infoquest`
- generation temperature: `0.0`
- backend defaults come from the model config
- ambiguity threshold: `0.5`
- max clarification turns: `1`

Current ablation flags:

- `ambiguity_detection`
- `intent_modeling`
- `strategy_selection`
- `targeted_question`

## Outputs

`run_eval.py` and `run_ablation.py` write timestamped or user-specified directories under `outputs/`.

Per-method outputs currently include:

- `predictions.jsonl`
- `metrics.json`
- `summary.csv`
- `report_snippet.md`

Cross-method evaluation writes:

- `comparison_report.md`
- `all_metrics.json`
- `all_metrics.csv`
- `resolved_config.json`

Single-method inference writes outputs under `outputs/<method>/` by default, and `run_method()` creates a nested method directory inside the selected output root.

## Repository Layout

```text
clarify/
  README.md
  TODO_PLAN.md
  Makefile
  pyproject.toml
  configs/
    default.yaml
    model/
    method/
  scripts/
    demo_example.py
    download_infoquest.py
    run_ablation.py
    run_eval.py
    run_inference.py
  src/
    data/
    eval/
    llm/
    methods/
    understand/
    utils/
  tests/
```

## Current Limitations

- Single-turn evaluation only.
- No live follow-up clarification loop yet.
- Open-ended correctness is based on a simple overlap proxy.
- No ProMISe adapter is implemented.
- `targeted_clarify` supports `present_alternatives`, but the evaluation still treats it with a coarse proxy rather than a dedicated metric.
- Running the real model stack requires a Python 3.10+ environment with the appropriate optional dependencies installed.
