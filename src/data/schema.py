"""Shared pydantic schemas used across the repository."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AmbiguityType = Literal["lexical", "referential", "underspecified", "missing_context"]
ConfidenceBand = Literal["low", "medium", "high"]
ResponseStrategy = Literal[
    "answer_directly",
    "ask_clarification",
    "narrow_and_answer",
    "present_alternatives",
    "abstain",
]


class DialogueExample(BaseModel):
    """A single evaluation instance: an ambiguous user request with hidden context."""

    example_id: str
    dataset_name: str = "unknown"
    split_name: str | None = None
    user_request: str
    hidden_context: str
    gold_clarification_needed: bool
    gold_clarifying_question: str | None = None
    gold_answer: str | None = None
    simulated_user_reply: str | None = None
    ambiguity_type: AmbiguityType = "underspecified"
    domain: str = "general"
    checklist: list[str] = Field(default_factory=list)
    personas: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AmbiguityRecord(BaseModel):
    """One detected ambiguity in a user request."""

    ambiguity_type: AmbiguityType
    missing_variable: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: str | None = None


class Interpretation(BaseModel):
    """A candidate interpretation of an ambiguous request."""

    description: str
    plausibility: float = Field(ge=0.0, le=1.0)
    assumed_context: str


class MethodResult(BaseModel):
    """Result of running one method on one example."""

    example_id: str
    method: str
    dataset_name: str = "unknown"
    split_name: str | None = None
    user_request: str
    hidden_context: str
    gold_clarification_needed: bool
    gold_answer: str | None = None
    task_model_id: str | None = None
    judge_model_id: str | None = None
    prompt_version: str | None = None
    judge_version: str | None = None

    # What the method produced
    response_strategy: ResponseStrategy
    response_text: str
    clarification_question: str | None = None
    assumed_interpretation: str | None = None
    final_answer: str | None = None
    simulated_user_reply: str | None = None
    answered_after_clarification: bool = False
    num_turns: int = 1

    # Assessment
    is_ambiguous_detected: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    intent_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_band: ConfidenceBand | None = None
    uncertainty_explanation: str | None = None
    weak_points: list[str] = Field(default_factory=list)
    resample_rounds: int = 0
    sample_count: int = 0
    task_model_calls: int = 0
    estimated_cost: float = Field(default=0.0, ge=0.0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    num_missing_variables: int = 0
    answer_score: float | None = Field(default=None, ge=0.0, le=1.0)
    clarification_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Evaluation flags (filled in by eval)
    asked_clarification: bool = False
    answered_directly: bool = False
    correct: bool = False

    # Full trace for debugging
    trace: dict[str, Any] = Field(default_factory=dict)
