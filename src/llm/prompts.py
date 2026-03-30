"""Centralized prompts for answer generation, verification, and revision."""

from __future__ import annotations

import json
from typing import Sequence, Type

from pydantic import BaseModel

from src.data.schema import RetrievedPassage


LEGAL_DISCLAIMER = (
    "This question is about housing law in a specific U.S. state as of 2021. "
    "Use the provided state and year constraints, do not invent citations, and abstain when evidence is insufficient. "
    "This system is for research only and is not legal advice."
)


def schema_instruction(schema_type: Type[BaseModel]) -> str:
    return (
        "Return only valid JSON matching this schema.\n"
        f"{json.dumps(schema_type.model_json_schema(), indent=2)}"
    )


def format_passages(passages: Sequence[RetrievedPassage]) -> str:
    if not passages:
        return "No passages retrieved."
    blocks: list[str] = []
    for passage in passages:
        blocks.append(
            "\n".join(
                [
                    f"[DOC_ID] {passage.doc_id}",
                    f"[STATE] {passage.state or 'unknown'}",
                    f"[CITATION] {passage.citation or 'unknown'}",
                    f"[TEXT] {passage.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def closed_book_answer_prompt(question: str, state: str, year: int, schema_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                f"Answer the following housing-law question for {state} as of {year}.\n"
                "Return a binary answer of Yes, No, or Abstain; a short explanation; and calibrated confidence.\n"
                "If the law is unclear or you are unsure, answer Abstain.\n\n"
                f"Question: {question}\n\n"
                f"{schema_text}"
            ),
        },
    ]


def rag_answer_prompt(
    question: str,
    state: str,
    year: int,
    passages: Sequence[RetrievedPassage],
    schema_text: str,
    hedging: bool = False,
) -> list[dict[str, str]]:
    hedge_line = (
        "If confidence is low, keep the explanation cautious and explicitly narrow the claim, but do not perform extra verification beyond the provided evidence.\n"
        if hedging
        else ""
    )
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                f"Answer the following housing-law question for {state} as of {year} using only the supplied statute passages.\n"
                "Do not invent citations. If the evidence is insufficient, answer Abstain.\n"
                f"{hedge_line}"
                f"Question: {question}\n\n"
                f"Retrieved statute passages:\n{format_passages(passages)}\n\n"
                f"{schema_text}"
            ),
        },
    ]


def claim_extraction_prompt(
    question: str,
    explanation: str,
    state: str,
    year: int,
    min_claims: int,
    max_claims: int,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                f"Extract {min_claims}-{max_claims} atomic support claims from the answer explanation below.\n"
                "Each claim must be independently checkable against statute text.\n"
                "Use one of these claim types: rule, exception, procedural, factual.\n"
                f"State: {state}\n"
                f"Year: {year}\n"
                f"Question: {question}\n"
                f"Explanation: {explanation}\n\n"
                f"{schema_text}"
            ),
        },
    ]


def claim_support_judging_prompt(
    question: str,
    claim_text: str,
    state: str,
    year: int,
    evidence: Sequence[RetrievedPassage],
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                f"Judge whether the statute passages support the claim below for {state} housing law as of {year}.\n"
                "Score support from 0 to 1. A score near 1 means the claim is directly supported by the evidence.\n"
                "Be strict and avoid giving credit for partial matches that do not establish the legal rule.\n\n"
                f"Question: {question}\n"
                f"Claim: {claim_text}\n\n"
                f"Evidence:\n{format_passages(evidence)}\n\n"
                f"{schema_text}"
            ),
        },
    ]


def query_rewrite_prompt(
    question: str,
    weak_claim: str,
    state: str,
    year: int,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                "Rewrite the search query so it better verifies the weakest legal claim.\n"
                "Focus on statute language, exceptions, and definitions that bear directly on the claim.\n\n"
                f"Question: {question}\n"
                f"Weak claim: {weak_claim}\n"
                f"State: {state}\n"
                f"Year: {year}\n\n"
                f"{schema_text}"
            ),
        },
    ]


def search_plan_prompt(
    question: str,
    weak_claim: str,
    state: str,
    year: int,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                "Create a compact legal search plan of 2-3 subqueries that follow a different retrieval trajectory.\n"
                "One query should target the exact rule, one should target exceptions, and one may target definitions.\n\n"
                f"Question: {question}\n"
                f"Weak claim: {weak_claim}\n"
                f"State: {state}\n"
                f"Year: {year}\n\n"
                f"{schema_text}"
            ),
        },
    ]


def minimal_revision_prompt(
    question: str,
    state: str,
    year: int,
    initial_answer_json: str,
    weak_claim: str,
    new_evidence: Sequence[RetrievedPassage],
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                "Revise the answer minimally. Only change what is needed to address the weakest claim.\n"
                "Preserve supported parts of the answer, do not invent citations, and abstain if the new evidence still does not support the answer.\n\n"
                f"Question: {question}\n"
                f"State: {state}\n"
                f"Year: {year}\n"
                f"Initial answer JSON: {initial_answer_json}\n"
                f"Weak claim: {weak_claim}\n\n"
                f"New evidence:\n{format_passages(new_evidence)}\n\n"
                f"{schema_text}"
            ),
        },
    ]


def abstention_prompt(
    question: str,
    state: str,
    year: int,
    predicted_answer: str,
    missing_verification: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                "Draft either a narrow answer or an abstention message.\n"
                "Be concise, professional, and explicit about what still needs verification.\n\n"
                f"Question: {question}\n"
                f"State: {state}\n"
                f"Year: {year}\n"
                f"Current predicted answer polarity: {predicted_answer}\n"
                f"Missing verification: {missing_verification}\n\n"
                f"{schema_text}"
            ),
        },
    ]


def confidence_estimation_prompt(
    question: str,
    answer: str,
    explanation: str,
    state: str,
    year: int,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LEGAL_DISCLAIMER},
        {
            "role": "user",
            "content": (
                "Estimate confidence in the answer below. Be conservative.\n\n"
                f"Question: {question}\n"
                f"State: {state}\n"
                f"Year: {year}\n"
                f"Answer: {answer}\n"
                f"Explanation: {explanation}\n\n"
                f"{schema_text}"
            ),
        },
    ]

