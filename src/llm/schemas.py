"""Structured output schemas for model-consumed prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.data.schema import ClaimType, confidence_bucket, normalize_answer_label


class ClosedBookAnswerSchema(BaseModel):
    answer: Literal["Yes", "No", "Abstain"]
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_bucket: Literal["low", "medium", "high"]

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, value: str) -> str:
        return normalize_answer_label(value)

    @field_validator("confidence_bucket", mode="before")
    @classmethod
    def _normalize_bucket(cls, value: str | None, info) -> str:
        if value is None and info.data.get("confidence") is not None:
            return confidence_bucket(float(info.data["confidence"]))
        return str(value).strip().lower()


class RagAnswerSchema(BaseModel):
    answer: Literal["Yes", "No", "Abstain"]
    explanation: str
    cited_statute_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_bucket: Literal["low", "medium", "high"]

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, value: str) -> str:
        return normalize_answer_label(value)

    @field_validator("confidence_bucket", mode="before")
    @classmethod
    def _normalize_bucket(cls, value: str | None, info) -> str:
        if value is None and info.data.get("confidence") is not None:
            return confidence_bucket(float(info.data["confidence"]))
        return str(value).strip().lower()


class ClaimSchema(BaseModel):
    claim_text: str
    claim_type: ClaimType
    importance_score: float = Field(ge=0.0, le=1.0)
    span_text_from_original_explanation: str


class ClaimListSchema(BaseModel):
    claims: list[ClaimSchema] = Field(min_length=1, max_length=5)


class SupportJudgmentSchema(BaseModel):
    supported: bool
    support_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class QueryRewriteSchema(BaseModel):
    rewritten_query: str
    justification: str


class SearchPlanStepSchema(BaseModel):
    query: str
    purpose: Literal["rule", "exception", "definition"]


class SearchPlanSchema(BaseModel):
    steps: list[SearchPlanStepSchema] = Field(min_length=2, max_length=3)


class RevisionSchema(BaseModel):
    answer: Literal["Yes", "No", "Abstain"]
    explanation: str
    cited_statute_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_bucket: Literal["low", "medium", "high"]
    revision_notes: str

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, value: str) -> str:
        return normalize_answer_label(value)

    @field_validator("confidence_bucket", mode="before")
    @classmethod
    def _normalize_bucket(cls, value: str | None, info) -> str:
        if value is None and info.data.get("confidence") is not None:
            return confidence_bucket(float(info.data["confidence"]))
        return str(value).strip().lower()


class AbstentionSchema(BaseModel):
    answer: Literal["Yes", "No", "Abstain"]
    explanation: str
    narrow_answer: bool
    requested_verification: str

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, value: str) -> str:
        return normalize_answer_label(value)

