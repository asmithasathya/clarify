# clarify Status and Next Steps

This file tracks the repository as it exists now, plus the remaining work that would materially improve it.

## Current Status

The legal-QA verification codepath has already been removed. The repo is now centered on ambiguity-aware clarification with these implemented components:

- `src/data/infoquest.py`: InfoQuest normalization and local JSONL loading.
- `src/data/schema.py`: dialogue-centric schemas such as `DialogueExample` and `MethodResult`.
- `src/understand/`: ambiguity detection, intent modeling, strategy selection, and clarification generation.
- `src/methods/`: `direct_answer`, `generic_hedge`, `generic_clarify`, and `targeted_clarify`.
- `src/eval/`: clarification-oriented aggregate metrics and experiment runner.
- `scripts/`: evaluation, inference, ablation, dataset download, and demo entry points.
- `tests/`: smoke tests for the new pipeline plus unit tests for data loading, ambiguity detection, intent modeling, and runner evaluation.

## What Is Working Today

### Data

- InfoQuest examples can be loaded from Hugging Face or from a local JSONL cache.
- Each example carries hidden context, optional gold answer text, and ambiguity metadata.

### Runtime pipeline

- Ambiguity detection can use either an LLM or a heuristic fallback.
- Intent modeling produces ranked interpretations.
- Strategy selection supports:
  - `answer_directly`
  - `ask_clarification`
  - `narrow_and_answer`
  - `present_alternatives`
  - `abstain`
- Clarification generation now respects the `targeted_question` ablation and falls back to generic questions when requested.
- `targeted_clarify` now guards against invalid `most_likely_index` values instead of crashing on out-of-range outputs.

### Evaluation

- Per-example correctness is computed before metrics are aggregated.
- Current metrics cover action selection, ambiguity detection behavior, abstention, and a coarse answer-success proxy.
- `present_alternatives` is treated as a distinct strategy rather than being mislabeled as a clarification question.
- `narrow_and_answer` is treated as ambiguity handling rather than as a premature direct answer.

### Tooling

- `run_inference.py` runs a single method and writes method-specific artifacts.
- `run_eval.py` compares methods across the dataset.
- `run_ablation.py` runs `targeted_clarify` under four ablation switches.
- `download_infoquest.py` creates a local JSONL cache.
- `demo_example.py` exercises the targeted clarification flow with a mock generator.

## Known Gaps

These are the biggest mismatches between the current repo and a stronger research-ready version.

### 1. Multi-turn clarification is still missing

The pipeline can ask a clarification question, but the benchmark runner stops there. There is no turn-2 loop that consumes a user reply, updates the interpretation, and then answers.

### 2. Correctness remains a lightweight proxy

`src/eval/runner.py` currently uses normalized text overlap and a few strategy-specific proxies. That is enough to keep metrics non-broken, but it is not a high-confidence open-ended evaluator.

### 3. Alternatives need better evaluation

`present_alternatives` is now tracked correctly as its own strategy, but there is no dedicated metric for whether the alternatives were useful, well-targeted, or matched the hidden context.

### 4. Clarification quality is only weakly supervised

When the dataset lacks `gold_clarifying_question`, clarification correctness is proxied by whether clarification was needed at all. That does not tell us whether the question targeted the right missing variable.

### 5. No second dataset yet

InfoQuest is the only supported benchmark today. There is no implemented ProMISe adapter or multi-turn secondary dataset.

### 6. Local environment friction still exists

The repo requires Python 3.10+, but not every local environment is provisioned with matching dependencies. This especially affects quick test execution outside a dedicated virtualenv.

## Recommended Next Work

### High priority

- Add a multi-turn evaluation loop for clarification followed by final answering.
- Replace the current answer-match proxy with a stronger evaluator for open-ended responses.
- Add a dedicated metric for `present_alternatives`.
- Add a targeted-question quality metric that checks whether the missing variable was actually addressed.

### Medium priority

- Add a second benchmark adapter, ideally multi-turn.
- Support offline evaluation more cleanly with documented cached-data workflows.
- Tighten the output layout for `run_inference.py` so the default directory nesting is simpler.

### Low priority

- Expand the demo examples.
- Add richer artifact reports for strategy traces.
- Add more ablations around heuristic fallbacks and strategy-selection defaults.

## Validation Expectations

For meaningful local validation, use a Python 3.10+ environment with the project dependencies installed. In that environment, the minimum useful checks are:

```bash
python -m pytest
python scripts/run_eval.py --methods direct_answer generic_hedge generic_clarify targeted_clarify --backend transformers --limit 10
```

## Short Changelog for the Latest Fix Pass

The latest review-fix branch addressed these issues:

- metrics were being aggregated before `MethodResult.correct` was set,
- the `no_targeted_question` ablation was not changing behavior,
- `targeted_clarify` could crash on invalid `most_likely_index` values,
- `narrow_and_answer` was incorrectly counted as a direct answer,
- `present_alternatives` was incorrectly counted as a clarification question.
