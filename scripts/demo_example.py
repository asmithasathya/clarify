"""Small smoke demo for the targeted clarification pipeline."""

from __future__ import annotations

import json

import typer

from src.data.schema import DialogueExample
from src.llm.generator import MockGenerator
from src.methods.targeted_clarify import run_targeted_clarify
from src.understand.ambiguity_detector import AmbiguityDetector
from src.understand.clarification_generator import ClarificationGenerator
from src.understand.intent_model import IntentModeler
from src.understand.strategy_selector import StrategySelector


app = typer.Typer(add_completion=False)


def _demo_responder(messages):
    """Mock responder that returns canned JSON for each pipeline stage."""
    prompt = messages[-1]["content"]

    if "ambiguous or underspecified" in prompt:
        return json.dumps({
            "is_ambiguous": True,
            "ambiguity_type": "missing_context",
            "missing_variables": [
                {
                    "variable": "what kind of investment advice",
                    "why_missing": "The user could mean stocks, real estate, retirement planning, or crypto.",
                    "importance": 0.9,
                },
                {
                    "variable": "risk tolerance",
                    "why_missing": "Investment advice varies dramatically by risk tolerance.",
                    "importance": 0.7,
                },
            ],
            "confidence": 0.85,
            "rationale": "The request 'help me invest' is severely underspecified.",
        })

    if "plausible interpretations" in prompt or "2-4 plausible" in prompt:
        return json.dumps({
            "interpretations": [
                {
                    "description": "User wants to start investing in index funds for retirement",
                    "assumed_context": "Young professional, long time horizon, moderate risk tolerance",
                    "plausibility": 0.4,
                },
                {
                    "description": "User wants advice on investing a windfall in real estate",
                    "assumed_context": "Has received a large sum, interested in property",
                    "plausibility": 0.3,
                },
                {
                    "description": "User wants to learn about day-trading stocks",
                    "assumed_context": "Interested in active trading, higher risk tolerance",
                    "plausibility": 0.2,
                },
            ],
            "most_likely_index": 0,
            "entropy_estimate": "high",
            "gap_description": "Type of investment and risk tolerance are both unknown.",
        })

    if "best response strategy" in prompt or "Choose the best" in prompt:
        return json.dumps({
            "strategy": "ask_clarification",
            "rationale": "Two critical variables (investment type, risk tolerance) are missing. A targeted question is best.",
            "confidence": 0.82,
        })

    if "targeted clarifying question" in prompt:
        return json.dumps({
            "question": "Are you looking to invest for long-term goals like retirement, or are you interested in shorter-term opportunities? Also, how comfortable are you with the possibility of losing some of your investment?",
            "target_variable": "what kind of investment advice",
            "why_this_helps": "This resolves both the investment type and risk tolerance in one natural question.",
        })

    if "Continue the conversation below" in prompt:
        return json.dumps({
            "answer": "Given your five-year timeline and need for capital preservation, start with a mix of high-yield savings and conservative diversified funds rather than aggressive investments.",
            "assumed_interpretation": None,
            "confidence": 0.86,
            "caveats": None,
        })

    # Fallback: direct answer
    return json.dumps({
        "answer": "I'd recommend starting with a diversified index fund.",
        "assumed_interpretation": "General investment advice for a beginner",
        "confidence": 0.4,
        "caveats": "This is very generic without knowing your situation.",
    })


@app.command()
def main() -> None:
    typer.echo("=== Clarify Demo ===")
    typer.echo("Demonstrates the targeted clarification pipeline with a mock generator.\n")

    config = {
        "project": {"seed": 42},
        "understand": {"ambiguity_threshold": 0.5},
        "ablations": {
            "ambiguity_detection": True,
            "intent_modeling": True,
            "strategy_selection": True,
            "targeted_question": True,
        },
        "generation": {"json_max_retries": 0},
    }

    example = DialogueExample(
        example_id="demo-001",
        user_request="Help me invest my money",
        hidden_context="User is a 28-year-old teacher who just inherited $50k and wants safe, long-term growth for a house down payment in 5 years.",
        gold_clarification_needed=True,
        gold_answer="Consider a mix of high-yield savings and conservative index funds given your 5-year timeline and need for capital preservation.",
        ambiguity_type="missing_context",
        domain="personal finance",
        checklist=[
            "Did the assistant ask about the investment timeline?",
            "Did the assistant ask about risk tolerance?",
            "Did the assistant ask about the investment amount?",
            "Did the assistant ask about specific goals?",
        ],
    )

    generator = MockGenerator(responder=_demo_responder, config=config)

    result = run_targeted_clarify(
        example,
        config=config,
        ambiguity_detector=AmbiguityDetector(config, generator=generator),
        intent_modeler=IntentModeler(config, generator=generator),
        strategy_selector=StrategySelector(config, generator=generator),
        clarification_generator=ClarificationGenerator(config, generator=generator),
    )

    typer.echo(f"User request: {example.user_request}")
    typer.echo(f"Hidden context: {example.hidden_context}")
    typer.echo(f"\nStrategy chosen: {result.response_strategy}")
    typer.echo(f"Response: {result.response_text}")
    typer.echo(f"\nAmbiguity detected: {result.is_ambiguous_detected}")
    typer.echo(f"Asked clarification: {result.asked_clarification}")
    typer.echo(f"Missing variables: {result.num_missing_variables}")
    if result.asked_clarification:
        followup = "I want safe, long-term growth and I might need the money for a house down payment in about five years."
        final_answer = ClarificationGenerator(config, generator=generator).generate_conversation_answer(
            [
                {"role": "user", "content": example.user_request},
                {"role": "assistant", "content": result.response_text},
                {"role": "user", "content": followup},
            ]
        )
        typer.echo(f"\nSimulated follow-up: {followup}")
        typer.echo(f"Final answer after clarification: {final_answer.answer}")
    typer.echo(f"\nFull trace:")
    typer.echo(json.dumps(result.trace, indent=2, default=str))


if __name__ == "__main__":
    app()
