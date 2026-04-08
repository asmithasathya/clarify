import json

from src.data.schema import DialogueExample
from src.llm.generator import MockGenerator
from src.methods.direct_answer import run_direct_answer
from src.methods.generic_clarify import run_generic_clarify
from src.methods.generic_hedge import run_generic_hedge
from src.methods.targeted_clarify import run_targeted_clarify
from src.understand.ambiguity_detector import AmbiguityDetector
from src.understand.clarification_generator import ClarificationGenerator
from src.understand.intent_model import IntentModeler
from src.understand.strategy_selector import StrategySelector


def make_config():
    return {
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


def make_example():
    return DialogueExample(
        example_id="test-1",
        user_request="Help me invest my money",
        hidden_context="User is a teacher saving for a house down payment.",
        gold_clarification_needed=True,
        gold_answer="Consider index funds for a 5-year horizon.",
        ambiguity_type="missing_context",
        domain="personal finance",
    )


def _responder(messages):
    prompt = messages[-1]["content"]

    if "ambiguous or underspecified" in prompt:
        return json.dumps({
            "is_ambiguous": True,
            "ambiguity_type": "missing_context",
            "missing_variables": [
                {"variable": "investment type", "why_missing": "not specified", "importance": 0.9},
            ],
            "confidence": 0.85,
            "rationale": "The request is underspecified.",
        })

    if "2-4 plausible" in prompt:
        return json.dumps({
            "interpretations": [
                {"description": "Retire savings", "assumed_context": "Long-term", "plausibility": 0.5},
                {"description": "Day trading", "assumed_context": "Short-term", "plausibility": 0.3},
            ],
            "most_likely_index": 0,
            "entropy_estimate": "high",
            "gap_description": "Investment type unknown.",
        })

    if "best response strategy" in prompt or "Choose the best" in prompt:
        return json.dumps({
            "strategy": "ask_clarification",
            "rationale": "Key variable missing.",
            "confidence": 0.8,
        })

    if "targeted clarifying question" in prompt:
        return json.dumps({
            "question": "What type of investing are you interested in?",
            "target_variable": "investment type",
            "why_this_helps": "Resolves the main ambiguity.",
        })

    if "Answer the following user request directly" in prompt:
        return json.dumps({
            "answer": "Consider a diversified index fund.",
            "assumed_interpretation": None,
            "confidence": 0.4,
            "caveats": None,
        })

    if "cautious hedging" in prompt:
        return json.dumps({
            "answer": "I'm not entirely sure what you mean, but index funds are generally a good start.",
            "hedge_reason": "The request is vague.",
            "confidence": 0.35,
        })

    if "Before answering" in prompt and "clarifying question" in prompt:
        return json.dumps({
            "question": "Could you tell me more about what you need?",
            "target_variable": "general",
            "why_this_helps": "More info would help.",
        })

    raise AssertionError(f"Unexpected prompt: {prompt[:120]}")


def test_direct_answer():
    config = make_config()
    generator = MockGenerator(responder=_responder, config=config)
    result = run_direct_answer(make_example(), generator=generator, config=config)
    assert result.method == "direct_answer"
    assert result.answered_directly is True
    assert result.final_answer is not None


def test_generic_hedge():
    config = make_config()
    generator = MockGenerator(responder=_responder, config=config)
    result = run_generic_hedge(make_example(), generator=generator, config=config)
    assert result.method == "generic_hedge"
    assert result.answered_directly is True
    assert result.final_answer is not None


def test_generic_clarify():
    config = make_config()
    generator = MockGenerator(responder=_responder, config=config)
    result = run_generic_clarify(make_example(), generator=generator, config=config)
    assert result.method == "generic_clarify"
    assert result.asked_clarification is True
    assert result.clarification_question is not None


def test_targeted_clarify():
    config = make_config()
    generator = MockGenerator(responder=_responder, config=config)
    result = run_targeted_clarify(
        make_example(),
        config=config,
        ambiguity_detector=AmbiguityDetector(config, generator=generator),
        intent_modeler=IntentModeler(config, generator=generator),
        strategy_selector=StrategySelector(config, generator=generator),
        clarification_generator=ClarificationGenerator(config, generator=generator),
    )
    assert result.method == "targeted_clarify"
    assert result.is_ambiguous_detected is True
    assert result.asked_clarification is True
    assert result.clarification_question is not None
    assert "investment" in result.clarification_question.lower() or "invest" in result.clarification_question.lower()


def test_targeted_clarify_clear_request():
    """When ambiguity detection says 'not ambiguous', should answer directly."""
    config = make_config()

    def clear_responder(messages):
        prompt = messages[-1]["content"]
        if "ambiguous or underspecified" in prompt:
            return json.dumps({
                "is_ambiguous": False,
                "ambiguity_type": "none",
                "missing_variables": [],
                "confidence": 0.9,
                "rationale": "Request is clear.",
            })
        if "Answer the following user request directly" in prompt:
            return json.dumps({
                "answer": "The capital of France is Paris.",
                "assumed_interpretation": None,
                "confidence": 0.95,
                "caveats": None,
            })
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")

    generator = MockGenerator(responder=clear_responder, config=config)
    example = DialogueExample(
        example_id="test-clear",
        user_request="What is the capital of France?",
        hidden_context="None",
        gold_clarification_needed=False,
    )
    result = run_targeted_clarify(
        example,
        config=config,
        ambiguity_detector=AmbiguityDetector(config, generator=generator),
        intent_modeler=IntentModeler(config, generator=generator),
        strategy_selector=StrategySelector(config, generator=generator),
        clarification_generator=ClarificationGenerator(config, generator=generator),
    )
    assert result.answered_directly is True
    assert result.is_ambiguous_detected is False
    assert "Paris" in result.response_text
