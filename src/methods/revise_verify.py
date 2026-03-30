"""Main verification-oriented revision method."""

from __future__ import annotations

from typing import Any, Sequence

from src.data.schema import ClaimSupportScore, MethodResult, RetrievedPassage, confidence_bucket
from src.llm.generator import BaseGenerator
from src.llm.prompts import rag_answer_prompt, schema_instruction
from src.llm.schemas import RagAnswerSchema
from src.verify.abstain import AbstentionPolicy
from src.verify.claim_extraction import ClaimExtractor
from src.verify.claim_scoring import BaseSupportScorer
from src.verify.revise import RevisionEngine


def _filter_supported_citations(
    response: RagAnswerSchema,
    passages: Sequence[RetrievedPassage],
) -> tuple[list[str], list[str]]:
    valid_ids = {passage.doc_id for passage in passages}
    valid_citations = {passage.citation for passage in passages if passage.citation}
    statute_ids = [value for value in response.cited_statute_ids if value in valid_ids]
    citations = [value for value in response.citations if value in valid_citations]
    return statute_ids, citations


def _score_claims(
    *,
    claims: Sequence[Any],
    example: Any,
    retriever: Any,
    support_scorer: BaseSupportScorer,
    top_k: int,
) -> list[ClaimSupportScore]:
    scored: list[ClaimSupportScore] = []
    for claim in claims:
        evidence = retriever.retrieve(claim.claim_text, state=example.state, top_k=top_k)
        scored.append(
            support_scorer.score(
                claim=claim,
                evidence=evidence,
                question=example.question,
                state=example.state,
                gold_statutes=example.statutes,
            )
        )
    return scored


def run_revise_verify(
    example: Any,
    *,
    generator: BaseGenerator,
    retriever: Any,
    config: dict[str, Any],
    claim_extractor: ClaimExtractor,
    support_scorer: BaseSupportScorer,
    trajectory_builder: Any,
    revision_engine: RevisionEngine,
    abstention_policy: AbstentionPolicy,
) -> MethodResult:
    top_k = config.get("retrieval", {}).get("top_k", 8)
    initial_passages = retriever.retrieve(example.question, state=example.state, top_k=top_k)
    initial_answer = generator.generate_structured(
        rag_answer_prompt(
            question=example.question,
            state=example.state,
            year=config.get("project", {}).get("year", 2021),
            passages=initial_passages,
            schema_text=schema_instruction(RagAnswerSchema),
            hedging=False,
        ),
        RagAnswerSchema,
        temperature=0.0,
    )
    extracted_claims = claim_extractor.extract(
        question=example.question,
        explanation=initial_answer.explanation,
        state=example.state,
    )
    initial_claim_scores = _score_claims(
        claims=extracted_claims,
        example=example,
        retriever=retriever,
        support_scorer=support_scorer,
        top_k=top_k,
    )
    weakest = min(initial_claim_scores, key=lambda item: item.final_score)

    trace: dict[str, Any] = {
        "initial_answer": initial_answer.model_dump(),
        "extracted_claims": [claim.model_dump() for claim in extracted_claims],
        "initial_support_scores": [score.model_dump() for score in initial_claim_scores],
        "weakest_claim": weakest.claim.model_dump(),
    }

    final_answer = initial_answer
    final_claim_scores = list(initial_claim_scores)
    verification_evidence = list(weakest.evidence)
    rewritten_query = weakest.claim.claim_text
    abstention_missing = weakest.claim.claim_text

    if config.get("ablations", {}).get("second_pass_verification", True):
        if config.get("ablations", {}).get("query_rewrite", True):
            trajectory = trajectory_builder.build_queries(
                question=example.question,
                weak_claim=weakest.claim.claim_text,
                state=example.state,
            )
            rewritten_query = trajectory["rewritten_query"]
            verification_queries = trajectory["queries"]
            trace["rewrite_trajectory"] = trajectory
        else:
            verification_queries = [weakest.claim.claim_text]
            trace["rewrite_trajectory"] = {
                "rewritten_query": weakest.claim.claim_text,
                "rewrite_justification": "Ablation disabled query rewriting.",
                "search_plan": None,
                "queries": verification_queries,
            }

        verification_evidence = retriever.retrieve_multi(
            verification_queries,
            state=example.state,
            top_k=top_k,
            rerank_query=weakest.claim.claim_text,
        )
        rescored_weakest = support_scorer.score(
            claim=weakest.claim,
            evidence=verification_evidence,
            question=example.question,
            state=example.state,
            gold_statutes=example.statutes,
        )
        final_claim_scores = [
            rescored_weakest if score.claim.claim_text == weakest.claim.claim_text else score
            for score in initial_claim_scores
        ]
        trace["new_evidence"] = [passage.model_dump() for passage in verification_evidence]
        trace["rescored_weakest_claim"] = rescored_weakest.model_dump()

        if rescored_weakest.final_score >= weakest.final_score:
            revised = revision_engine.revise(
                question=example.question,
                state=example.state,
                weak_claim=weakest.claim.claim_text,
                initial_answer=initial_answer,
                new_evidence=verification_evidence,
            )
            final_answer = RagAnswerSchema(
                answer=revised.answer,
                explanation=revised.explanation,
                cited_statute_ids=revised.cited_statute_ids,
                citations=revised.citations,
                confidence=revised.confidence,
                confidence_bucket=revised.confidence_bucket,
            )
            trace["revised_answer"] = revised.model_dump()
        else:
            trace["revised_answer"] = initial_answer.model_dump()
    else:
        trace["rewrite_trajectory"] = {
            "rewritten_query": rewritten_query,
            "rewrite_justification": "Ablation disabled second-pass verification.",
            "search_plan": None,
            "queries": [rewritten_query],
        }
        trace["new_evidence"] = [passage.model_dump() for passage in verification_evidence]
        trace["revised_answer"] = initial_answer.model_dump()

    final_support = min(score.final_score for score in final_claim_scores)
    abstention_decision = abstention_policy.decide(
        question=example.question,
        state=example.state,
        predicted_answer=final_answer.answer,
        current_explanation=final_answer.explanation,
        final_support_score=final_support,
        missing_verification=abstention_missing,
    )

    final_predicted_answer = abstention_decision.predicted_answer
    final_explanation = abstention_decision.explanation
    statute_ids, citations = _filter_supported_citations(final_answer, verification_evidence or initial_passages)
    trace["abstain_decision"] = abstention_decision.model_dump()
    trace["final_confidence"] = final_answer.confidence

    conf = float(final_answer.confidence)
    return MethodResult(
        example_id=example.example_id,
        method="revise_verify",
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=final_predicted_answer,
        explanation=final_explanation,
        state=example.state,
        confidence=conf,
        confidence_bucket=final_answer.confidence_bucket or confidence_bucket(conf),
        citations=citations,
        statute_ids=statute_ids,
        abstained=abstention_decision.abstained,
        narrowed=abstention_decision.narrowed,
        correct=(final_predicted_answer == example.answer and final_predicted_answer != "Abstain"),
        support_score=final_support,
        retrieved_passages=list(verification_evidence or initial_passages),
        claim_scores=final_claim_scores,
        trace=trace,
    )
