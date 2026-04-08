# clarify

Research repository for studying when and how an assistant should clarify ambiguous user requests instead of answering prematurely.

## Core thesis

> A socially intelligent assistant should know when it does not yet understand the user well enough to answer. We study whether targeted clarification and interpretation revision outperform premature answering and generic hedging in ambiguous help-seeking dialogue.

The object of uncertainty is **the assistant's understanding of the user**, not factual correctness. When a request is underspecified, the system should detect what is missing, model candidate interpretations, and choose whether to ask a targeted clarifying question, narrow the interpretation explicitly, or answer directly.

## Pipeline

```
User request
  → Ambiguity detection (is this underspecified?)
  → Intent modeling (what could the user mean?)
  → Strategy selection (clarify / narrow / answer / present alternatives / abstain)
  → Response generation (targeted question, narrowed answer, or direct answer)
```

## Implemented methods

The repository implements four assistant conditions:

1. **`direct_answer`** — Answer immediately with no ambiguity checking. Baseline.
2. **`generic_hedge`** — Answer with hedging language when confidence is low. No clarification.
3. **`generic_clarify`** — Always ask a generic clarification question regardless of actual ambiguity.
4. **`targeted_clarify`** — Main method. Detect ambiguity, identify the specific missing variable, choose the best strategy, and act accordingly.

## Primary dataset

[InfoQuest](https://huggingface.co/datasets/bryanlincoln/infoquest) — open-ended requests with hidden context, where models should ask clarifying questions before answering.

Each seed message is intentionally ambiguous, with two plausible settings (goal, obstacle, constraints) that the assistant must discover through clarification.

## Model stack

- Generator: `Qwen/Qwen2.5-7B-Instruct`
- Default backend: vLLM (fallback: Transformers)
- Structured output via Pydantic schemas with JSON retry

## Evaluation metrics

- **appropriate_action_rate** — answered when clear + clarified when ambiguous
- **clarification_precision** — when asked, was clarification actually needed?
- **clarification_recall** — of ambiguous cases, how often did the method clarify?
- **missed_ambiguity_rate** — answered directly when it should have clarified
- **unnecessary_clarification_rate** — asked when it didn't need to
- **task_success_rate** — final answer correctness
- **ambiguity_detection_accuracy** — detector agreement with gold labels

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For vLLM:

```bash
pip install -e ".[dev,vllm]"
```

## Quickstart

1. Run the smoke demo (mock generator, no GPU needed):

```bash
python -m scripts.demo_example
```

2. Download InfoQuest locally (optional):

```bash
python -m scripts.download_infoquest
```

3. Run one method:

```bash
python scripts/run_inference.py --method targeted_clarify --limit 100
```

4. Run multi-method evaluation:

```bash
python scripts/run_eval.py --methods direct_answer generic_hedge generic_clarify targeted_clarify
```

5. Run ablations:

```bash
python scripts/run_ablation.py \
  --ablation no_ambiguity_detection \
  --ablation no_intent_modeling \
  --ablation no_strategy_selection \
  --ablation no_targeted_question
```

For non-vLLM environments:

```bash
--backend transformers
```

## Repository layout

```
clarify/
  README.md
  pyproject.toml
  Makefile
  configs/
    default.yaml
    model/qwen25_7b.yaml
    method/{direct_answer,generic_hedge,generic_clarify,targeted_clarify}.yaml
  src/
    data/           # DialogueExample schema, InfoQuest loader
    llm/            # Generator backends, prompts, output schemas
    understand/     # Ambiguity detector, intent modeler, strategy selector, clarification generator
    methods/        # Four experimental conditions
    eval/           # Metrics and experiment runner
    utils/          # Config, I/O, logging, seeding
  scripts/          # CLI entry points
  tests/            # Pytest suite
  data/             # Cached datasets
  outputs/          # Experiment artifacts
```

## Configuration

YAML-based configs under `configs/`. The loader merges includes from `default.yaml` with method-specific overrides.

Key ablation flags in `configs/default.yaml`:

- `ambiguity_detection` — disable to skip detection (always answer)
- `intent_modeling` — disable to skip interpretation ranking
- `strategy_selection` — disable to use binary clarify/answer
- `targeted_question` — disable to use generic questions when clarifying

## Limitations

- Single-turn evaluation only (no multi-turn dialogue simulation yet)
- Evaluation of answer quality for open-ended responses is approximate
- Full model inference requires a GPU-friendly environment
- ProMISe dataset adapter is planned but not yet implemented (repo is unavailable)
