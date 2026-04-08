"""Centralized prompts for ambiguity detection, clarification, and answering."""

from __future__ import annotations

import json
from typing import Type

from pydantic import BaseModel


SYSTEM_PREAMBLE = (
    "You are a helpful assistant that pays close attention to what the user "
    "actually needs. Before answering, consider whether the request is clear "
    "enough for a useful response. If important details are missing, it is "
    "better to ask a targeted clarifying question than to guess."
)


def schema_instruction(schema_type: Type[BaseModel]) -> str:
    return (
        "Return only valid JSON matching this schema.\n"
        f"{json.dumps(schema_type.model_json_schema(), indent=2)}"
    )


# ---------------------------------------------------------------------------
# Ambiguity detection
# ---------------------------------------------------------------------------

def ambiguity_detection_prompt(
    request: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "Analyze the following user request and determine whether it is "
                "ambiguous or underspecified.\n\n"
                "Consider:\n"
                "- Is there missing context that would change the answer?\n"
                "- Could this request plausibly mean very different things?\n"
                "- Are there implicit assumptions that might be wrong?\n"
                "- What specific information is the user NOT providing that you "
                "would need to give a truly helpful response?\n\n"
                f"User request: {request}\n\n"
                f"{schema_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Intent modeling
# ---------------------------------------------------------------------------

def intent_modeling_prompt(
    request: str,
    ambiguity_rationale: str,
    missing_variables: list[str],
    schema_text: str,
) -> list[dict[str, str]]:
    vars_block = "\n".join(f"- {v}" for v in missing_variables) if missing_variables else "- (none identified)"
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "Given an ambiguous user request, generate 2-4 plausible "
                "interpretations of what the user might actually mean or need.\n\n"
                "Rank them by plausibility. For each interpretation, describe "
                "what context the user would need to have for that interpretation "
                "to be correct.\n\n"
                f"User request: {request}\n\n"
                f"Why it is ambiguous: {ambiguity_rationale}\n\n"
                f"Missing variables:\n{vars_block}\n\n"
                f"{schema_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def strategy_selection_prompt(
    request: str,
    is_ambiguous: bool,
    num_missing_variables: int,
    entropy: str,
    gap_description: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "Choose the best response strategy for this user request.\n\n"
                "Options:\n"
                "- answer_directly: the request is clear enough to answer well.\n"
                "- ask_clarification: the most important missing variable should "
                "be resolved first. Ask ONE targeted question.\n"
                "- narrow_and_answer: state your assumed interpretation explicitly, "
                "then answer under that assumption.\n"
                "- present_alternatives: the request has 2-3 distinctly different "
                "valid interpretations. Present each with a short answer.\n"
                "- abstain: the request is too vague or unclear to help.\n\n"
                f"User request: {request}\n"
                f"Ambiguous: {is_ambiguous}\n"
                f"Number of missing variables: {num_missing_variables}\n"
                f"Interpretation entropy: {entropy}\n"
                f"Gap description: {gap_description}\n\n"
                f"{schema_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Clarification question generation
# ---------------------------------------------------------------------------

def clarification_question_prompt(
    request: str,
    target_variable: str,
    interpretations_summary: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "Generate a single, targeted clarifying question for the user.\n\n"
                "The question must:\n"
                "- Reference the specific ambiguity, not be generic.\n"
                "- Help resolve the most important missing variable.\n"
                "- Be natural and conversational, not robotic.\n"
                "- NOT be something like 'Can you be more specific?' or "
                "'Could you provide more details?'\n\n"
                f"User request: {request}\n"
                f"Most important missing variable: {target_variable}\n"
                f"Possible interpretations:\n{interpretations_summary}\n\n"
                f"{schema_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Direct answer
# ---------------------------------------------------------------------------

def direct_answer_prompt(
    request: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "Answer the following user request directly and helpfully.\n\n"
                f"User request: {request}\n\n"
                f"{schema_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Narrowed answer (answer under stated assumptions)
# ---------------------------------------------------------------------------

def narrowed_answer_prompt(
    request: str,
    assumed_interpretation: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "The user's request is ambiguous. Answer it under the following "
                "assumed interpretation. State the assumption explicitly at the "
                "start of your response.\n\n"
                f"User request: {request}\n"
                f"Assumed interpretation: {assumed_interpretation}\n\n"
                f"{schema_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Alternatives presentation
# ---------------------------------------------------------------------------

def alternatives_prompt(
    request: str,
    interpretations_json: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "The user's request has multiple valid interpretations. "
                "Present each interpretation with a short answer.\n\n"
                "Start with a brief preamble acknowledging the ambiguity, then "
                "list each interpretation and its answer.\n\n"
                f"User request: {request}\n\n"
                f"Interpretations:\n{interpretations_json}\n\n"
                f"{schema_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Hedged answer (for generic_hedge baseline)
# ---------------------------------------------------------------------------

def hedged_answer_prompt(
    request: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "Answer the following user request. If you are not sure what "
                "the user means, use cautious hedging language but still provide "
                "an answer. Do NOT ask clarifying questions.\n\n"
                f"User request: {request}\n\n"
                f"{schema_text}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Generic clarification (for generic_clarify baseline)
# ---------------------------------------------------------------------------

def generic_clarification_prompt(
    request: str,
    schema_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE},
        {
            "role": "user",
            "content": (
                "Before answering the following user request, generate a "
                "clarifying question. The question can be generic \u2014 you do not "
                "need to identify specific ambiguities.\n\n"
                f"User request: {request}\n\n"
                f"{schema_text}"
            ),
        },
    ]
