"""Shared pydantic schemas used across the repository."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


BinaryAnswer = Literal["Yes", "No", "Abstain"]
ConfidenceBucket = Literal["low", "medium", "high"]
ClaimType = Literal["rule", "exception", "procedural", "factual"]


def normalize_answer_label(value: str | None) -> BinaryAnswer:
    if value is None:
        return "Abstain"
    cleaned = value.strip().lower()
    if cleaned.startswith("y"):
        return "Yes"
    if cleaned.startswith("n"):
        return "No"
    return "Abstain"


def confidence_bucket(score: float) -> ConfidenceBucket:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


class LegalQAExample(BaseModel):
    example_id: str
    question: str
    answer: BinaryAnswer
    state: str
    statutes: list[str] = Field(default_factory=list)
    citation: list[str] = Field(default_factory=list)
    excerpt: list[str] = Field(default_factory=list)
    year: int = 2021
    dataset_name: str = "housingqa"
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatutePassage(BaseModel):
    doc_id: str
    text: str
    state: str | None = None
    citation: str | None = None
    title: str | None = None
    source_dataset: str = "housingqa_statutes"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        chunks = [self.state or "", self.citation or "", self.title or "", self.text]
        return " ".join(chunk for chunk in chunks if chunk).strip()


class RetrievedPassage(BaseModel):
    doc_id: str
    text: str
    state: str | None = None
    citation: str | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    fused_score: float | None = None
    rank: int = 0


class ClaimRecord(BaseModel):
    claim_text: str
    claim_type: ClaimType
    importance_score: float = Field(ge=0.0, le=1.0)
    span_text_from_original_explanation: str


class ClaimSupportScore(BaseModel):
    claim: ClaimRecord
    retrieval_score: float = Field(ge=0.0, le=1.0)
    lexical_overlap_score: float = Field(ge=0.0, le=1.0)
    gold_match_score: float = Field(ge=0.0, le=1.0)
    judge_score: float | None = Field(default=None, ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    supported: bool
    evidence: list[RetrievedPassage] = Field(default_factory=list)
    rationale: str | None = None


class MethodResult(BaseModel):
    example_id: str
    method: str
    question: str
    gold_answer: BinaryAnswer
    predicted_answer: BinaryAnswer
    explanation: str
    state: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_bucket: ConfidenceBucket
    citations: list[str] = Field(default_factory=list)
    statute_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    narrowed: bool = False
    correct: bool = False
    support_score: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    claim_scores: list[ClaimSupportScore] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)

    @field_validator("predicted_answer", mode="before")
    @classmethod
    def _normalize_predicted_answer(cls, value: str | BinaryAnswer) -> BinaryAnswer:
        return normalize_answer_label(value)

    @field_validator("gold_answer", mode="before")
    @classmethod
    def _normalize_gold_answer(cls, value: str | BinaryAnswer) -> BinaryAnswer:
        return normalize_answer_label(value)


class AbstentionDecision(BaseModel):
    predicted_answer: BinaryAnswer
    explanation: str
    narrowed: bool = False
    abstained: bool = False
    requested_verification: str | None = None

