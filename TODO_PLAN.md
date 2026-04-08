# Migration Plan: Legal QA Verification → Ambiguity-Aware Clarification for MSI

## Summary

Transform this project from **"verify the answer's factual support"** (legal QA claim verification) to **"verify the assistant's understanding of the user"** (ambiguity detection → targeted clarification → answer only when user intent is resolved). The object of uncertainty shifts from *"is this claim evidence-supported?"* to *"do I understand what the user actually means/needs?"*

---

## Current Architecture (what exists)

```
User question → Draft answer → Extract claims → Score weakest claim
  → Retrieve alternative evidence → Revise or abstain
```

**Key modules:**
- `src/data/` — HousingQA dataset loaders, `LegalQAExample` / `StatutePassage` schemas
- `src/llm/` — Generator backends (vLLM/Transformers/Mock), legal-domain prompts, Pydantic output schemas
- `src/methods/` — 4 conditions: `closed_book`, `rag_direct`, `hedge`, `revise_verify`
- `src/verify/` — Claim extraction, support scoring, revision engine, abstention policy
- `src/retrieval/` — Dense retrieval + reranking + multi-query fusion + query rewriting
- `src/eval/` — Selective metrics, retrieval metrics, attribution metrics, experiment runner
- `configs/` — YAML configs for model, retrieval, methods, ablations
- `scripts/` — CLI entry points for inference, evaluation, ablation, demo

---

## Target Architecture (what we're building)

```
User request → Ambiguity detection → [if ambiguous] → Identify missing variable
  → Choose strategy (clarify / narrow / answer-with-caveats / abstain)
  → Generate clarification question OR narrowed response
  → [if clear enough] → Answer directly
```

**Core thesis:** *"User-model-oriented clarification beats premature answering and generic hedging."*

---

## Phase 1: Data Layer — Replace Legal QA with Ambiguity/Clarification Benchmarks

### 1.1 New dataset adapter: InfoQuest
**Files to change:** `src/data/schema.py`, new `src/data/infoquest.py`

- **Replace `LegalQAExample`** with a new `DialogueExample` schema:
  ```
  example_id, user_request, hidden_context, gold_clarification_needed (bool),
  gold_clarifying_question (optional), gold_answer (given full context),
  ambiguity_type, domain, metadata
  ```
- **Remove** `StatutePassage`, `BinaryAnswer`, `ClaimType`, `ClaimRecord`, `ClaimSupportScore` — these are legal-verification-specific
- **Add** `AmbiguityRecord` schema:
  ```
  ambiguity_type (lexical / referential / underspecified / missing_context),
  missing_variable, confidence_in_detection, source_span
  ```
- **New loader** `src/data/infoquest.py` — download/parse InfoQuest dataset, normalize into `DialogueExample`
- **Remove** `src/data/housingqa.py`, `src/data/legalqa_local.py`

### 1.2 Optional secondary: ProMISe
**New file:** `src/data/promise.py`

- Multi-turn intent-refinement dataset
- Adapter that normalizes ProMISe conversations into sequences of `DialogueExample` turns

### 1.3 Config updates
**Files to change:** `configs/default.yaml`, remove `configs/retrieval/bge_faiss.yaml`

- Replace `data.housingqa` / `data.legalqa_local` with `data.infoquest` and `data.promise`
- Remove statute-index paths, legal year constraints
- Add clarification-specific config: `ambiguity_threshold`, `max_clarification_turns`, `strategy_weights`

---

## Phase 2: Core Pipeline — Replace Claim Verification with User-Model Assessment

### 2.1 New module: `src/understand/` (replaces `src/verify/`)

#### `src/understand/ambiguity_detector.py` (replaces `claim_extraction.py`)
- **Purpose:** Given a user request, detect whether it is underspecified/ambiguous
- Outputs: `AmbiguityAssessment` — is_ambiguous (bool), ambiguity_type, missing_variables (list), confidence
- Two modes: LLM-based detection, heuristic fallback (question length, entity count, specificity signals)

#### `src/understand/intent_model.py` (replaces `claim_scoring.py`)
- **Purpose:** Build a lightweight model of what the user likely means/needs
- Given request + detected ambiguities → produce candidate interpretations ranked by plausibility
- Output: `IntentModel` — interpretations (list of `Interpretation`), entropy, most_likely, gap_description

#### `src/understand/strategy_selector.py` (replaces `abstain.py`)
- **Purpose:** Choose the best response strategy given the ambiguity assessment
- Strategies:
  - `answer_directly` — request is clear enough
  - `ask_clarification` — one targeted question about the most important missing variable
  - `narrow_and_answer` — state assumptions explicitly, then answer under those assumptions
  - `present_alternatives` — show 2-3 interpretations with answers for each
  - `abstain` — too ambiguous to help meaningfully
- Decision based on: ambiguity score, number of missing variables, stakes, domain

#### `src/understand/clarification_generator.py` (replaces `revise.py`)
- **Purpose:** Generate the actual clarification question or narrowed interpretation
- Takes the top missing variable + candidate interpretations → produces a natural, targeted question
- Must avoid generic "Can you be more specific?" — question should reference the specific ambiguity

### 2.2 Remove `src/verify/`
- Delete: `claim_extraction.py`, `claim_scoring.py`, `support_checker.py`, `revise.py`, `abstain.py`
- These are all legal-claim-verification-specific

---

## Phase 3: Methods — Redefine the Four Experimental Conditions

### 3.1 New methods (replace `src/methods/`)

**Files to change:** All four method files, `src/methods/__init__.py`

| Old method | New method | Description |
|---|---|---|
| `closed_book` | `direct_answer` | Answer immediately, no ambiguity checking. Baseline: "just answer the question" |
| `hedge` | `generic_hedge` | Detect low confidence → add generic hedging language ("I'm not sure exactly what you mean, but...") without targeted clarification |
| `rag_direct` | `generic_clarify` | Always ask a generic clarification question ("Could you provide more details?") regardless of actual ambiguity — tests whether any clarification helps vs. targeted |
| `revise_verify` | `targeted_clarify` | **Main method.** Detect ambiguity → identify specific missing variable → ask targeted clarification or present narrowed interpretation → answer only when user model is resolved |

### 3.2 `src/methods/targeted_clarify.py` (core new method)
Pipeline:
1. Receive user request
2. Run ambiguity detector → get ambiguity assessment
3. If clear: answer directly (fast path)
4. If ambiguous: run intent model → get candidate interpretations
5. Run strategy selector → choose response type
6. If `ask_clarification`: generate targeted question via clarification generator
7. If `narrow_and_answer`: state the assumed interpretation, then answer under it
8. If `present_alternatives`: generate 2-3 interpretation+answer pairs
9. Record full trace for analysis

### 3.3 `src/methods/direct_answer.py`
- Strip-down: just answer, no uncertainty handling at all

### 3.4 `src/methods/generic_hedge.py`
- Answer + add hedging language when model self-reports low confidence
- No ambiguity detection, no clarification

### 3.5 `src/methods/generic_clarify.py`
- Always produce a generic clarification question
- Tests: does *any* clarification help, or does it need to be *targeted*?

---

## Phase 4: Prompts — Replace Legal Prompts with Dialogue/Clarification Prompts

### 4.1 `src/llm/prompts.py` — full rewrite

**Remove all legal-specific prompts** (`LEGAL_DISCLAIMER`, `rag_answer_prompt`, `claim_extraction_prompt`, `claim_support_judging_prompt`, `query_rewrite_prompt`, `search_plan_prompt`, `minimal_revision_prompt`, `abstention_prompt`, `confidence_estimation_prompt`)

**New prompts:**
- `ambiguity_detection_prompt(request)` — "Is this request underspecified? What's missing?"
- `intent_modeling_prompt(request, ambiguities)` — "What are the most likely interpretations?"
- `strategy_selection_prompt(request, ambiguity_assessment, intent_model)` — "Should I clarify, narrow, or answer?"
- `clarification_question_prompt(request, missing_variable, interpretations)` — "Generate a targeted clarifying question"
- `narrowed_answer_prompt(request, assumed_interpretation)` — "Answer under this specific interpretation"
- `direct_answer_prompt(request)` — "Answer this clear request directly"

### 4.2 `src/llm/schemas.py` — replace legal schemas

**Remove:** `ClosedBookAnswerSchema`, `RagAnswerSchema`, `ClaimSchema`, `ClaimListSchema`, `SupportJudgmentSchema`, `QueryRewriteSchema`, `SearchPlanSchema`, `RevisionSchema`, `AbstentionSchema`

**New schemas:**
- `AmbiguityAssessmentSchema` — is_ambiguous, ambiguity_type, missing_variables, confidence
- `IntentModelSchema` — interpretations (list), entropy_estimate, most_likely_index
- `StrategyDecisionSchema` — strategy (enum), rationale
- `ClarificationQuestionSchema` — question, target_variable, why_this_helps
- `AnswerSchema` — answer, assumed_interpretation, confidence, caveats

---

## Phase 5: Evaluation — Replace Legal Metrics with Clarification/MSI Metrics

### 5.1 `src/eval/metrics.py` — new metric suite

**Remove:** retrieval metrics (Recall@k, MRR, hit@k), attribution metrics (claim support, citation overlap), legal-specific selective metrics

**New core metrics:**
- `task_success_rate` — did the final answer (after any clarification) match the gold answer?
- `unnecessary_clarification_rate` — asked a question when the request was actually clear
- `missed_ambiguity_rate` — answered directly when the request was actually ambiguous
- `wrong_answer_under_ambiguity` — answered without clarifying AND got it wrong
- `clarification_precision` — when a clarification was asked, did it target a real missing variable?
- `hidden_context_recovery` — after clarification, how much of the hidden context was recovered?
- `turns_to_resolution` — how many turns to reach a correct answer?
- `interpretation_accuracy` — did the system's assumed interpretation match the gold hidden context?

### 5.2 `src/eval/selective_metrics.py` — adapt
- Keep `coverage`, `abstention_rate` concepts but reframe:
  - `answer_rate` = fraction that answered (vs. clarified or abstained)
  - `clarification_rate` = fraction that asked a clarification question
  - `appropriate_action_rate` = answered when clear + clarified when ambiguous

### 5.3 Remove `src/eval/attribution_metrics.py`, `src/eval/retrieval_metrics.py`
- These are legal-retrieval-specific

---

## Phase 6: Retrieval Layer — Simplify or Remove

### 6.1 Decision: retrieval is mostly unnecessary for this pivot
- InfoQuest/ProMISe are dialogue benchmarks, not document-retrieval tasks
- The "alternative trajectory" concept maps to "alternative interpretation" not "alternative search query"
- **Remove:** `src/retrieval/` entirely (embedder, faiss_index, rerank, retrieve, query_rewrite)
- If a future extension needs retrieval (e.g., to look up domain knowledge while answering), it can be re-added later with a simpler interface

### 6.2 Simplify dependencies
- Remove from `pyproject.toml`: `faiss-cpu`, `sentence-transformers` (unless needed for embedding-based ambiguity detection)
- Keep: `torch`, `transformers`, `pydantic`, `datasets`, `pyyaml`, `typer`, `tqdm`

---

## Phase 7: Scripts and Configs

### 7.1 Scripts
- **Rewrite `scripts/run_inference.py`** — run one method on InfoQuest examples
- **Rewrite `scripts/run_eval.py`** — compare `direct_answer`, `generic_hedge`, `generic_clarify`, `targeted_clarify`
- **Rewrite `scripts/run_ablation.py`** — ablate components of targeted_clarify:
  - `no_ambiguity_detection` — always clarify or never clarify
  - `no_intent_modeling` — detect ambiguity but skip interpretation ranking
  - `no_strategy_selection` — always use the same strategy
  - `generic_question_only` — detect ambiguity but ask generic question instead of targeted
- **Rewrite `scripts/demo_example.py`** — demonstrate the clarification pipeline on example ambiguous requests
- **Remove `scripts/download_housingqa.py`**, **remove `scripts/build_statute_index.py`**
- **Add `scripts/download_infoquest.py`** — fetch and cache InfoQuest

### 7.2 Configs
- **Rewrite `configs/default.yaml`** — new dataset, new method defaults, new eval settings
- **Remove `configs/retrieval/bge_faiss.yaml`**
- **Rewrite `configs/model/qwen25_7b.yaml`** — keep model but update prompt-length settings
- **Rewrite `configs/method/*.yaml`** — one per new method

---

## Phase 8: Tests

### 8.1 Update tests
- **Rewrite `tests/test_data.py`** — test InfoQuest loader instead of HousingQA
- **Remove `tests/test_retrieval.py`**, **remove `tests/test_claim_extraction.py`**
- **Add `tests/test_ambiguity_detector.py`** — unit tests for ambiguity detection
- **Add `tests/test_intent_model.py`** — unit tests for intent modeling
- **Rewrite `tests/test_methods_smoke.py`** — smoke tests for all four new methods using MockGenerator

---

## Phase 9: Project Metadata

### 9.1 `pyproject.toml`
- Rename project: `"clarify"` (already the repo name)
- Update description: *"Research repository for studying when and how an assistant should clarify ambiguous user requests instead of answering prematurely."*
- Update dependencies (remove faiss-cpu, sentence-transformers if unused)

### 9.2 `README.md`
- Full rewrite reflecting new thesis, methods, datasets, metrics, and quickstart

### 9.3 `Makefile`
- Update targets to reflect new scripts

---

## File-Level Change Summary

### Delete entirely
| File | Reason |
|---|---|
| `src/data/housingqa.py` | Legal dataset adapter |
| `src/data/legalqa_local.py` | Legal dataset adapter |
| `src/verify/` (all 5 files + `__init__`) | Claim verification pipeline |
| `src/retrieval/` (all 6 files + `__init__`) | Dense retrieval for statutes |
| `src/eval/attribution_metrics.py` | Legal citation metrics |
| `src/eval/retrieval_metrics.py` | Retrieval metrics |
| `scripts/download_housingqa.py` | Legal data download |
| `scripts/build_statute_index.py` | FAISS index builder |
| `configs/retrieval/bge_faiss.yaml` | Retrieval config |
| `tests/test_retrieval.py` | Retrieval tests |
| `tests/test_claim_extraction.py` | Claim extraction tests |

### Create new
| File | Purpose |
|---|---|
| `src/data/infoquest.py` | InfoQuest dataset loader |
| `src/data/promise.py` | ProMISe dataset loader (optional/secondary) |
| `src/understand/__init__.py` | New core module |
| `src/understand/ambiguity_detector.py` | Detect underspecified requests |
| `src/understand/intent_model.py` | Model candidate user interpretations |
| `src/understand/strategy_selector.py` | Choose clarify/narrow/answer/abstain |
| `src/understand/clarification_generator.py` | Generate targeted clarification questions |
| `src/methods/direct_answer.py` | Baseline: answer immediately |
| `src/methods/generic_hedge.py` | Baseline: hedge when uncertain |
| `src/methods/generic_clarify.py` | Baseline: always ask generic question |
| `src/methods/targeted_clarify.py` | Main method: detect + targeted clarify |
| `scripts/download_infoquest.py` | Fetch InfoQuest data |
| `tests/test_ambiguity_detector.py` | Ambiguity detection tests |
| `tests/test_intent_model.py` | Intent modeling tests |
| `configs/method/direct_answer.yaml` | Config for direct answer baseline |
| `configs/method/generic_hedge.yaml` | Config for hedge baseline |
| `configs/method/generic_clarify.yaml` | Config for generic clarify baseline |
| `configs/method/targeted_clarify.yaml` | Config for main method |

### Rewrite substantially
| File | What changes |
|---|---|
| `src/data/schema.py` | Replace all legal schemas with dialogue/ambiguity schemas |
| `src/llm/prompts.py` | Replace all legal prompts with clarification prompts |
| `src/llm/schemas.py` | Replace all legal output schemas with clarification schemas |
| `src/eval/metrics.py` | Replace legal metrics with clarification metrics |
| `src/eval/selective_metrics.py` | Reframe for clarification rates |
| `src/eval/runner.py` | Wire up new methods, datasets, metrics |
| `scripts/run_eval.py` | Update CLI for new methods |
| `scripts/run_inference.py` | Update CLI for new methods |
| `scripts/run_ablation.py` | New ablation dimensions |
| `scripts/demo_example.py` | Demo the clarification pipeline |
| `configs/default.yaml` | New dataset/method/eval defaults |
| `configs/model/qwen25_7b.yaml` | Minor updates |
| `tests/test_methods_smoke.py` | Test new methods |
| `tests/test_data.py` | Test new dataset loaders |
| `pyproject.toml` | Rename, update deps/description |
| `README.md` | Full rewrite |
| `Makefile` | Update targets |

### Keep mostly unchanged
| File | Notes |
|---|---|
| `src/llm/generator.py` | Generator backends are model-agnostic — keep as-is |
| `src/utils/config.py` | Config loader is generic — keep as-is |
| `src/utils/io.py` | I/O helpers are generic — keep as-is |
| `src/utils/logging.py` | Keep as-is |
| `src/utils/seed.py` | Keep as-is |

---

## Recommended Implementation Order

1. **Phase 1** (Data) — Get InfoQuest loading and `DialogueExample` schema working first. Everything depends on this.
2. **Phase 4** (Prompts + Schemas) — Define new LLM prompts and output schemas before building logic that uses them.
3. **Phase 2** (Core pipeline / `src/understand/`) — Build ambiguity detection → intent modeling → strategy selection → clarification generation.
4. **Phase 3** (Methods) — Wire the four conditions using the new pipeline.
5. **Phase 5** (Evaluation) — New metrics so you can measure results.
6. **Phase 6** (Retrieval cleanup) — Delete retrieval code, simplify deps.
7. **Phase 7** (Scripts + Configs) — Update all CLI entry points and configs.
8. **Phase 8** (Tests) — Update test suite.
9. **Phase 9** (Metadata) — README, pyproject.toml, Makefile.

---

## What stays the same

- **`src/llm/generator.py`** — The entire generator infrastructure (BaseGenerator, TransformersGenerator, VLLMGenerator, MockGenerator, structured output parsing) is domain-agnostic and should be kept unchanged.
- **`src/utils/`** — All utility code (config loader, I/O, logging, seeding) is generic.
- **Overall architecture pattern** — The "methods dispatch to a pipeline of LLM calls with structured output" pattern is preserved. The pipeline *contents* change, but the scaffolding remains.
- **Evaluation runner structure** — The `run_experiment → run_method → compute_metrics → write_artifacts` pattern stays. Only the methods, metrics, and dataset plugged into it change.
