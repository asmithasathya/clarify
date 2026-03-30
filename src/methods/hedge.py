"""Naive hedging baseline: same retrieval path, softer language when confidence is low."""

from __future__ import annotations

from typing import Any

from src.data.schema import ClaimRecord, MethodResult, confidence_bucket
from src.llm.generator import BaseGenerator
from src.llm.prompts import rag_answer_prompt, schema_instruction
from src.llm.schemas import RagAnswerSchema
from src.verify.claim_scoring import BaseSupportScorer


def _filter_supported_citations(response: RagAnswerSchema, passages: list[Any]) -> tuple[list[str], list[str]]:
    valid_ids = {passage.doc_id for passage in passages}
    valid_citations = {passage.citation for passage in passages if passage.citation}
    statute_ids = [value for value in response.cited_statute_ids if value in valid_ids]
    citations = [value for value in response.citations if value in valid_citations]
    return statute_ids, citations


def run_hedge(
    example: Any,
    *,
    generator: BaseGenerator,
    retriever: Any,
    config: dict[str, Any],
    support_scorer: BaseSupportScorer | None = None,
) -> MethodResult:
    passages = retriever.retrieve(
        example.question,
        state=example.state,
        top_k=config.get("retrieval", {}).get("top_k", 8),
    )
    response = generator.generate_structured(
        rag_answer_prompt(
            question=example.question,
            state=example.state,
            year=config.get("project", {}).get("year", 2021),
            passages=passages,
            schema_text=schema_instruction(RagAnswerSchema),
            hedging=True,
        ),
        RagAnswerSchema,
        temperature=0.0,
    )
    statute_ids, citations = _filter_supported_citations(response, passages)

    claim_scores = []
    support_score = None
    if support_scorer is not None:
        pseudo_claim = ClaimRecord(
            claim_text=response.explanation,
            claim_type="rule",
            importance_score=1.0,
            span_text_from_original_explanation=response.explanation,
        )
        support = support_scorer.score(
            claim=pseudo_claim,
            evidence=passages,
            question=example.question,
            state=example.state,
            gold_statutes=example.statutes,
        )
        claim_scores = [support]
        support_score = support.final_score

    answer = response.answer
    conf = float(response.confidence)
    return MethodResult(
        example_id=example.example_id,
        method="hedge",
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=answer,
        explanation=response.explanation,
        state=example.state,
        confidence=conf,
        confidence_bucket=response.confidence_bucket or confidence_bucket(conf),
        citations=citations,
        statute_ids=statute_ids,
        abstained=(answer == "Abstain"),
        narrowed=False,
        correct=(answer == example.answer and answer != "Abstain"),
        support_score=support_score,
        retrieved_passages=list(passages),
        claim_scores=claim_scores,
        trace={"hedge_answer": response.model_dump()},
    )

