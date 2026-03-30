"""Support-scoring interfaces and baseline implementations."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from src.data.schema import ClaimRecord, ClaimSupportScore, RetrievedPassage
from src.verify.support_checker import LLMSupportJudge


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def _overlap_ratio(claim_text: str, passage_text: str) -> float:
    claim_tokens = _tokenize(claim_text)
    if not claim_tokens:
        return 0.0
    passage_tokens = _tokenize(passage_text)
    return len(claim_tokens & passage_tokens) / len(claim_tokens)


def _normalize_retrieval_score(passages: Sequence[RetrievedPassage]) -> float:
    if not passages:
        return 0.0
    top = passages[0]
    if top.rerank_score is not None:
        return 1.0 / (1.0 + math.exp(-top.rerank_score))
    dense = top.dense_score or 0.0
    return max(0.0, min(1.0, (dense + 1.0) / 2.0))


def _gold_overlap(passages: Sequence[RetrievedPassage], gold_statutes: Sequence[str]) -> float:
    if not gold_statutes:
        return 0.0
    gold = {value.strip().lower() for value in gold_statutes}
    for passage in passages:
        citation = (passage.citation or "").strip().lower()
        doc_id = passage.doc_id.strip().lower()
        if citation in gold or doc_id in gold:
            return 1.0
    return 0.0


class BaseSupportScorer(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def score(
        self,
        *,
        claim: ClaimRecord,
        evidence: Sequence[RetrievedPassage],
        question: str,
        state: str,
        gold_statutes: Sequence[str] | None = None,
    ) -> ClaimSupportScore:
        raise NotImplementedError


class RetrievalSupportScorer(BaseSupportScorer):
    def score(
        self,
        *,
        claim: ClaimRecord,
        evidence: Sequence[RetrievedPassage],
        question: str,
        state: str,
        gold_statutes: Sequence[str] | None = None,
    ) -> ClaimSupportScore:
        gold_statutes = gold_statutes or []
        retrieval_score = _normalize_retrieval_score(evidence)
        lexical_overlap = max((_overlap_ratio(claim.claim_text, passage.text) for passage in evidence), default=0.0)
        gold_match = _gold_overlap(evidence, gold_statutes)

        verify_cfg = self.config.get("verify", {})
        total = (
            verify_cfg.get("retrieval_weight", 0.55) * retrieval_score
            + verify_cfg.get("lexical_weight", 0.15) * lexical_overlap
            + verify_cfg.get("gold_weight", 0.05) * gold_match
        )
        max_weight = (
            verify_cfg.get("retrieval_weight", 0.55)
            + verify_cfg.get("lexical_weight", 0.15)
            + verify_cfg.get("gold_weight", 0.05)
        )
        final_score = total / max_weight if max_weight else 0.0
        threshold = verify_cfg.get("support_threshold", 0.62)
        return ClaimSupportScore(
            claim=claim,
            retrieval_score=retrieval_score,
            lexical_overlap_score=lexical_overlap,
            gold_match_score=gold_match,
            final_score=max(0.0, min(1.0, final_score)),
            supported=final_score >= threshold,
            evidence=list(evidence),
            rationale="Retrieval-only support score from reranker/dense, lexical overlap, and optional gold overlap.",
        )


class JudgeAssistedSupportScorer(RetrievalSupportScorer):
    def __init__(self, config: dict[str, Any], judge: LLMSupportJudge | None = None) -> None:
        super().__init__(config)
        self.judge = judge

    def score(
        self,
        *,
        claim: ClaimRecord,
        evidence: Sequence[RetrievedPassage],
        question: str,
        state: str,
        gold_statutes: Sequence[str] | None = None,
    ) -> ClaimSupportScore:
        baseline = super().score(
            claim=claim,
            evidence=evidence,
            question=question,
            state=state,
            gold_statutes=gold_statutes,
        )
        judgment = self.judge.judge(
            question=question,
            claim_text=claim.claim_text,
            state=state,
            evidence=evidence,
        ) if self.judge is not None else None

        if judgment is None:
            return baseline

        verify_cfg = self.config.get("verify", {})
        judge_weight = verify_cfg.get("judge_weight", 0.25)
        baseline_weight = 1.0 - judge_weight
        final_score = baseline_weight * baseline.final_score + judge_weight * judgment.support_score

        return ClaimSupportScore(
            claim=claim,
            retrieval_score=baseline.retrieval_score,
            lexical_overlap_score=baseline.lexical_overlap_score,
            gold_match_score=baseline.gold_match_score,
            judge_score=judgment.support_score,
            final_score=max(0.0, min(1.0, final_score)),
            supported=final_score >= verify_cfg.get("support_threshold", 0.62),
            evidence=list(evidence),
            rationale=judgment.rationale,
        )


def build_support_scorer(
    config: dict[str, Any],
    judge: LLMSupportJudge | None = None,
) -> BaseSupportScorer:
    scorer_name = config.get("ablations", {}).get("support_scorer", "judge_assisted")
    if scorer_name == "retrieval_only":
        return RetrievalSupportScorer(config)
    return JudgeAssistedSupportScorer(config, judge=judge)

