"""LLM-judge helpers for answer, clarification, and alternatives quality."""

from __future__ import annotations

import re
from typing import Any

from src.data.schema import DialogueExample, MethodResult
from src.llm.generator import BaseGenerator
from src.llm.prompts import (
    judge_alternatives_prompt,
    judge_answer_prompt,
    judge_clarification_prompt,
    schema_instruction,
)
from src.llm.schemas import (
    AlternativesEvaluationSchema,
    AnswerEvaluationSchema,
    ClarificationEvaluationSchema,
)


def _normalize_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def heuristic_answer_score(answer: str | None, example: DialogueExample) -> tuple[bool, float, str]:
    if not answer:
        return False, 0.0, "No answer provided."
    answer_tokens = _normalize_tokens(answer)
    gold_tokens = _normalize_tokens(example.gold_answer)
    if not gold_tokens:
        hidden_tokens = _normalize_tokens(example.hidden_context)
        overlap = len(answer_tokens & hidden_tokens)
        score = min(1.0, overlap / max(1, len(hidden_tokens)))
    else:
        overlap = len(answer_tokens & gold_tokens)
        score = overlap / max(1, len(gold_tokens))
    return score >= 0.6, score, "Token-overlap heuristic fallback."


def heuristic_clarification_score(question: str | None, example: DialogueExample) -> tuple[bool, float, str]:
    if not question:
        return False, 0.0, "No clarification question provided."
    question_tokens = _normalize_tokens(question)
    checklist_tokens = _normalize_tokens(" ".join(example.checklist))
    hidden_tokens = _normalize_tokens(example.hidden_context)
    overlap = len(question_tokens & (checklist_tokens | hidden_tokens))
    score = min(1.0, overlap / max(1, len(question_tokens)))
    return score >= 0.15, score, "Checklist/hidden-context overlap heuristic fallback."


def heuristic_alternatives_score(response_text: str | None, example: DialogueExample) -> tuple[bool, float, str]:
    if not response_text:
        return False, 0.0, "No alternatives response provided."
    response_tokens = _normalize_tokens(response_text)
    hidden_tokens = _normalize_tokens(example.hidden_context)
    overlap = len(response_tokens & hidden_tokens)
    numbered_branches = len(re.findall(r"\n\s*\d+\.", response_text))
    structure_bonus = 0.2 if numbered_branches >= 2 else 0.0
    score = min(1.0, structure_bonus + overlap / max(1, len(hidden_tokens)))
    return score >= 0.35, score, "Alternatives structure + overlap heuristic fallback."


class LLMJudge:
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator

    def evaluate_answer(self, example: DialogueExample, answer: str) -> AnswerEvaluationSchema:
        return self.generator.generate_structured(
            judge_answer_prompt(
                request=example.user_request,
                hidden_context=example.hidden_context,
                gold_answer=example.gold_answer,
                candidate_answer=answer,
                schema_text=schema_instruction(AnswerEvaluationSchema),
            ),
            AnswerEvaluationSchema,
            temperature=0.0,
        )

    def evaluate_clarification(
        self,
        example: DialogueExample,
        clarification_question: str,
    ) -> ClarificationEvaluationSchema:
        return self.generator.generate_structured(
            judge_clarification_prompt(
                request=example.user_request,
                hidden_context=example.hidden_context,
                clarification_question=clarification_question,
                schema_text=schema_instruction(ClarificationEvaluationSchema),
            ),
            ClarificationEvaluationSchema,
            temperature=0.0,
        )

    def evaluate_alternatives(
        self,
        example: DialogueExample,
        alternatives_response: str,
    ) -> AlternativesEvaluationSchema:
        return self.generator.generate_structured(
            judge_alternatives_prompt(
                request=example.user_request,
                hidden_context=example.hidden_context,
                alternatives_response=alternatives_response,
                schema_text=schema_instruction(AlternativesEvaluationSchema),
            ),
            AlternativesEvaluationSchema,
            temperature=0.0,
        )


def populate_quality_scores(
    result: MethodResult,
    example: DialogueExample,
    judge: LLMJudge | None = None,
) -> MethodResult:
    if result.final_answer:
        if judge is not None:
            evaluation = judge.evaluate_answer(example, result.final_answer)
            result.correct = evaluation.is_correct
            result.answer_score = evaluation.score
            result.trace.setdefault("judge", {})["answer"] = evaluation.model_dump()
        else:
            is_correct, score, rationale = heuristic_answer_score(result.final_answer, example)
            result.correct = is_correct
            result.answer_score = score
            result.trace.setdefault("judge", {})["answer"] = {"rationale": rationale, "score": score}

    if result.clarification_question:
        if judge is not None:
            evaluation = judge.evaluate_clarification(example, result.clarification_question)
            result.clarification_quality_score = evaluation.score
            result.trace.setdefault("judge", {})["clarification"] = evaluation.model_dump()
        else:
            _, score, rationale = heuristic_clarification_score(result.clarification_question, example)
            result.clarification_quality_score = score
            result.trace.setdefault("judge", {})["clarification"] = {"rationale": rationale, "score": score}

    if result.response_strategy == "present_alternatives":
        if judge is not None:
            evaluation = judge.evaluate_alternatives(example, result.response_text)
            result.alternatives_quality_score = evaluation.score
            result.trace.setdefault("judge", {})["alternatives"] = evaluation.model_dump()
            if not result.final_answer:
                result.correct = evaluation.is_useful
        else:
            is_useful, score, rationale = heuristic_alternatives_score(result.response_text, example)
            result.alternatives_quality_score = score
            result.trace.setdefault("judge", {})["alternatives"] = {"rationale": rationale, "score": score}
            if not result.final_answer:
                result.correct = is_useful

    return result
