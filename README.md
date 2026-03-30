# dont-bluff-verify-legal

Research-first repository for socially intelligent uncertainty handling in legal QA.

This project studies a specific failure mode in legal QA: an assistant gives a plausible answer, but one part of the explanation is weakly supported. Instead of only softening the language, this repository implements a verification-oriented revise pipeline that:

1. drafts an initial answer,
2. decomposes that answer into support claims,
3. finds the weakest claim,
4. follows a different retrieval trajectory for that claim,
5. revises the answer if support improves,
6. otherwise narrows or abstains.

## Important disclaimer

This repository is for research and class-project use only. It is not legal advice.

The primary benchmark, `reglab/housing_qa`, reflects housing-law questions and statutes framed around 2021 law. All prompts in this repo explicitly instruct the model to reason about the specified state as of 2021. Do not use outputs from this system as current legal guidance.

## Why this is not just calibration

This project is not a generic legal chatbot and it is not a pure confidence-calibration project.

Calibration-only baselines usually estimate confidence after producing an answer, or hedge their wording when confidence is low. That matters, but it does not directly test whether a system can identify the specific weak point in its reasoning and actively verify it through a different retrieval path.

The central hypothesis here is:

`verification-oriented revision beats naive hedging for reducing unsupported answers, while preserving useful answers.`

That is why the main comparison is `hedge` versus `revise_verify`, and why the repository tracks:

- `bad_acceptance_proxy = fraction of non-abstained answers that are incorrect`
- `useful_answer_rate = fraction of cases answered correctly and non-abstained`
- `selective_accuracy`
- weak-claim verification retrieval quality

## Primary dataset

Required primary benchmark:

- Questions:
  - `load_dataset("reglab/housing_qa", "questions", split="test")`
- Statute corpus:
  - `load_dataset("reglab/housing_qa", "statutes", split="corpus")`

The repository supports the required HousingQA fields:

- `question`
- `answer`
- `state`
- `statutes`
- `citation`
- `excerpt`

Optional secondary adapter:

- `legalqa_local`
- Supports local `.jsonl` or `.csv`
- Not required for the main HousingQA pipeline

## Implemented methods

The repository implements exactly four assistant conditions:

1. `closed_book`
   - No retrieval
   - Direct answer with `Yes` / `No` / `Abstain`
   - Explanation plus confidence score and confidence bucket

2. `rag_direct`
   - Dense retrieval over HousingQA statutes
   - Optional reranking
   - Direct statute-grounded answer

3. `hedge`
   - Same retrieval path as `rag_direct`
   - No extra verification
   - Only changes uncertainty language when confidence is low

4. `revise_verify`
   - Initial answer
   - Claim extraction
   - Weakest-claim support scoring
   - Alternative search trajectory
   - Second-pass retrieval and re-scoring
   - Minimal revision
   - Narrow-or-abstain policy when support remains low
   - Full trace output for debugging

## Model and retrieval stack

- Generator model:
  - `Qwen/Qwen2.5-7B-Instruct`
- Default inference backend:
  - vLLM
- Fallback inference backend:
  - Transformers
- Dense embeddings:
  - `BAAI/bge-base-en-v1.5`
- Dense index:
  - FAISS
- Reranker:
  - `BAAI/bge-reranker-base`

The code keeps these components modular so you can swap retrievers, rerankers, or generators later.

PEFT/LoRA hooks are included as future-facing scaffolding. Main experiments do not require fine-tuning.

## Repository layout

```text
dont-bluff-verify-legal/
  README.md
  pyproject.toml
  .gitignore
  Makefile
  configs/
  data/
  outputs/
  scripts/
  src/
  tests/
```

## Installation

Python 3.11 is required.

Base install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

If you want the default vLLM backend:

```bash
pip install -e ".[dev,vllm,peft]"
```

Why vLLM is an extra instead of a base dependency:

- it is large and environment-sensitive,
- many users will want the pure Transformers fallback,
- the default config still targets vLLM, but you can override with `--backend transformers`.

## Quickstart

1. Optional: cache HousingQA locally

```bash
python scripts/download_housingqa.py --config configs/default.yaml
```

2. Build the statute index

```bash
python scripts/build_statute_index.py --config configs/default.yaml
```

3. Run the lightweight smoke demo

```bash
python scripts/demo_example.py
```

The smoke demo uses a mocked generator and tiny in-memory retrieval setup so you can verify repository wiring without downloading the full model stack.

4. Run one real method

```bash
python scripts/run_inference.py --method revise_verify --limit 100
```

5. Run the multi-method evaluation

```bash
python scripts/run_eval.py --methods closed_book rag_direct hedge revise_verify
```

6. Run ablations

```bash
python scripts/run_ablation.py \
  --ablation no_query_rewrite \
  --ablation no_claim_decomp \
  --ablation no_reranker \
  --ablation no_abstain
```

If you do not want vLLM, run the inference and evaluation commands with:

```bash
--backend transformers
```

## Reproducible commands

Build the statute index:

```bash
python scripts/build_statute_index.py --config configs/default.yaml
```

Run a small smoke demo:

```bash
python scripts/demo_example.py
```

Run one method:

```bash
python scripts/run_inference.py --method revise_verify --limit 100
```

Run full evaluation:

```bash
python scripts/run_eval.py --methods closed_book rag_direct hedge revise_verify
```

Run ablations:

```bash
python scripts/run_ablation.py \
  --ablation no_query_rewrite \
  --ablation no_claim_decomp \
  --ablation no_reranker \
  --ablation no_second_pass_verification \
  --ablation no_abstain \
  --ablation retrieval_only_support_scorer \
  --ablation judge_assisted_support_scorer
```

## Configuration

Configs are YAML-based and live under `configs/`.

- `configs/default.yaml`
  - project defaults, paths, dataset settings, runtime, eval settings
- `configs/model/qwen25_7b.yaml`
  - generator backend and model settings
- `configs/retrieval/bge_faiss.yaml`
  - embedding, FAISS, reranking, and support-scoring settings
- `configs/method/*.yaml`
  - method-specific toggles

The config loader merges:

1. includes from `default.yaml`
2. the default file itself
3. a selected method override, when requested

## Outputs

Every run writes a timestamped folder under `outputs/` containing:

- per-example predictions in JSONL
- aggregate metrics JSON
- pretty summary CSV
- markdown report snippets
- case-study folders:
  - `improved_after_revision/`
  - `abstained_correctly/`
  - `failed_revision/`
  - `over-abstained/`

Important `revise_verify` trace fields:

- `initial_answer`
- `extracted_claims`
- `initial_support_scores`
- `weakest_claim`
- `rewrite_trajectory`
- `new_evidence`
- `revised_answer`
- `abstain_decision`
- `final_confidence`

## Evaluation metrics

Core answer metrics:

- exact match accuracy for `Yes` / `No`
- abstention rate
- selective accuracy on answered cases
- coverage
- risk-coverage curve data
- confusion matrix including `Abstain`

Retrieval metrics:

- Recall@k against gold statutes
- MRR against gold statutes
- hit@k against gold statutes

Support and attribution metrics:

- average claim support score
- fraction of explanation sentences with at least one supporting citation
- fraction of final answers whose cited passages overlap with gold statutes
- unsupported answer rate

Main project proxy metrics:

- `bad_acceptance_proxy`
- `useful_answer_rate`

## Design notes

### Claim extraction

- Structured JSON via Pydantic schema
- Deterministic prompting where possible
- Robust JSON parsing and retries
- Heuristic sentence-split fallback when no generator is available

### Support scoring

The v1 scorer is intentionally pragmatic and modular.

It combines:

- top retrieval or reranker score
- lexical overlap
- optional gold-statute overlap during HousingQA evaluation
- optional generator-as-judge support score

Both variants are implemented:

- retrieval-only
- judge-assisted

### Alternative retrieval trajectory

The weak-claim verification module supports:

- a single rewritten query
- a structured 2-3 query legal search plan
- multi-query retrieval fusion

### Abstention policy

The policy is intentionally simple and inspectable:

- answer normally if support is above threshold
- narrow the answer if polarity looks plausible but explanation support is weak
- abstain if support remains below threshold

## Limitations

- The repo is intentionally focused on HousingQA first.
- There is no web app or frontend.
- There is no browser tool use or general-purpose legal research agent.
- Retrieval metrics use HousingQA gold annotations as an offline proxy and do not replace legal validation.
- Full-corpus indexing can be expensive in time, storage, and RAM.
- Running `Qwen/Qwen2.5-7B-Instruct` plus reranking locally may require a GPU-friendly environment.
- vLLM is provided as the default intended backend, but not bundled in the base dependency set because installation is environment-specific.
- The local `legalqa_local` adapter is a file-format adapter only. It does not create gold statute annotations automatically.

## Ethics

Legal QA systems can sound authoritative even when they are not well supported. This repository is designed around that risk. The goal is not to make a more persuasive legal assistant. The goal is to make unsupported legal answers easier to detect, revise, narrow, or abstain from.

Again: outputs from this repository are not legal advice.

## Future extensions

- stronger claim-level attribution metrics
- better state-aware filtering and per-state indexes
- explicit sentence-to-citation alignment
- learned query rewriting or decomposition
- PEFT/LoRA fine-tuning experiments
- support for additional legal QA benchmarks
- richer analysis of over-abstention versus bad acceptance
