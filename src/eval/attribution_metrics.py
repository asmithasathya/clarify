"""Support and attribution metrics."""

from __future__ import annotations

import re
from typing import Sequence

from src.data.schema import MethodResult


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def _sentence_split(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]


def average_claim_support_score(results: Sequence[MethodResult]) -> float:
    scores = [claim.final_score for result in results for claim in result.claim_scores]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def fraction_explanation_sentences_with_supporting_citation(results: Sequence[MethodResult]) -> float:
    sentence_total = 0
    sentence_supported = 0
    for result in results:
        sentences = _sentence_split(result.explanation)
        if not sentences:
            continue
        cited_passages = [
            passage for passage in result.retrieved_passages
            if passage.doc_id in result.statute_ids or (passage.citation and passage.citation in result.citations)
        ]
        sentence_total += len(sentences)
        for sentence in sentences:
            sentence_tokens = _tokenize(sentence)
            if any(len(sentence_tokens & _tokenize(passage.text)) >= 2 for passage in cited_passages):
                sentence_supported += 1
    if sentence_total == 0:
        return 0.0
    return sentence_supported / sentence_total


def fraction_final_answers_with_gold_overlap(
    results: Sequence[MethodResult],
    gold_lookup: dict[str, Sequence[str]],
) -> float:
    if not results:
        return 0.0
    hits = 0
    for result in results:
        gold = {value.strip().lower() for value in gold_lookup.get(result.example_id, [])}
        cited = {value.strip().lower() for value in (result.statute_ids + result.citations) if value}
        if gold and cited & gold:
            hits += 1
    return hits / len(results)


def unsupported_answer_rate(results: Sequence[MethodResult], threshold: float = 0.55) -> float:
    answered = [result for result in results if not result.abstained]
    if not answered:
        return 0.0
    bad = 0
    for result in answered:
        if not result.correct:
            bad += 1
            continue
        if result.support_score is not None and result.support_score < threshold:
            bad += 1
    return bad / len(answered)

