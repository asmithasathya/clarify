"""Claim decomposition utilities for verification-oriented revision."""

from __future__ import annotations

import re
from typing import Any

from src.data.schema import ClaimRecord
from src.llm.generator import BaseGenerator
from src.llm.prompts import claim_extraction_prompt, schema_instruction
from src.llm.schemas import ClaimListSchema


def _guess_claim_type(sentence: str) -> str:
    lowered = sentence.lower()
    if "except" in lowered or "unless" in lowered:
        return "exception"
    if any(keyword in lowered for keyword in ["notice", "file", "serve", "written", "days"]):
        return "procedural"
    if any(keyword in lowered for keyword in ["because", "facts", "based on", "if the tenant"]):
        return "factual"
    return "rule"


def _sentence_split(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
    return chunks or [text.strip()]


def fallback_claims(explanation: str, min_claims: int = 2, max_claims: int = 5) -> list[ClaimRecord]:
    sentences = _sentence_split(explanation)
    selected = sentences[:max_claims]
    if len(selected) < min_claims and sentences:
        selected = (sentences * min_claims)[:min_claims]
    claims: list[ClaimRecord] = []
    total = max(len(selected), 1)
    for idx, sentence in enumerate(selected):
        claims.append(
            ClaimRecord(
                claim_text=sentence,
                claim_type=_guess_claim_type(sentence),
                importance_score=max(0.1, 1.0 - (idx / total)),
                span_text_from_original_explanation=sentence,
            )
        )
    return claims


class ClaimExtractor:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    def extract(self, question: str, explanation: str, state: str) -> list[ClaimRecord]:
        if not self.config.get("ablations", {}).get("claim_decomposition", True):
            return [
                ClaimRecord(
                    claim_text=explanation.strip(),
                    claim_type="rule",
                    importance_score=1.0,
                    span_text_from_original_explanation=explanation.strip(),
                )
            ]

        verify_cfg = self.config.get("verify", {})
        min_claims = verify_cfg.get("min_claims", 2)
        max_claims = verify_cfg.get("max_claims", 5)

        if self.generator is None:
            return fallback_claims(explanation, min_claims=min_claims, max_claims=max_claims)

        parsed = self.generator.generate_structured(
            claim_extraction_prompt(
                question=question,
                explanation=explanation,
                state=state,
                year=self.config.get("project", {}).get("year", 2021),
                min_claims=min_claims,
                max_claims=max_claims,
                schema_text=schema_instruction(ClaimListSchema),
            ),
            ClaimListSchema,
            temperature=0.0,
        )
        return [
            ClaimRecord(
                claim_text=claim.claim_text.strip(),
                claim_type=claim.claim_type,
                importance_score=claim.importance_score,
                span_text_from_original_explanation=claim.span_text_from_original_explanation.strip(),
            )
            for claim in parsed.claims[:max_claims]
        ]

