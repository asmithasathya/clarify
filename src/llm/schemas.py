"""Structured output schemas for model-consumed prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ambiguity detection
# ---------------------------------------------------------------------------

class MissingVariableSchema(BaseModel):
    variable: str
    why_missing: str
    importance: float = Field(ge=0.0, le=1.0)


class AmbiguityAssessmentSchema(BaseModel):
    is_ambiguous: bool
    ambiguity_type: Literal["lexical", "referential", "underspecified", "missing_context", "none"]
    missing_variables: list[MissingVariableSchema] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


# ---------------------------------------------------------------------------
# Intent modeling
# ---------------------------------------------------------------------------

class InterpretationSchema(BaseModel):
    description: str
    assumed_context: str
    plausibility: float = Field(ge=0.0, le=1.0)


class IntentModelSchema(BaseModel):
    interpretations: list[InterpretationSchema] = Field(min_length=1, max_length=5)
    most_likely_index: int = 0
    entropy_estimate: Literal["low", "medium", "high"]
    gap_description: str


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

class StrategyDecisionSchema(BaseModel):
    strategy: Literal[
        "answer_directly",
        "ask_clarification",
        "narrow_and_answer",
        "present_alternatives",
        "abstain",
    ]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Clarification question generation
# ---------------------------------------------------------------------------

class ClarificationQuestionSchema(BaseModel):
    question: str
    target_variable: str
    why_this_helps: str


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

class AnswerSchema(BaseModel):
    answer: str
    assumed_interpretation: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: str | None = None


# ---------------------------------------------------------------------------
# Narrowed answer (answer under explicit assumptions)
# ---------------------------------------------------------------------------

class NarrowedAnswerSchema(BaseModel):
    stated_assumption: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: str | None = None


# ---------------------------------------------------------------------------
# Alternatives presentation
# ---------------------------------------------------------------------------

class AlternativeAnswerSchema(BaseModel):
    interpretation: str
    answer: str


class AlternativesResponseSchema(BaseModel):
    preamble: str
    alternatives: list[AlternativeAnswerSchema] = Field(min_length=2, max_length=4)


# ---------------------------------------------------------------------------
# Hedged answer
# ---------------------------------------------------------------------------

class HedgedAnswerSchema(BaseModel):
    answer: str
    hedge_reason: str
    confidence: float = Field(ge=0.0, le=1.0)
