from src.llm.generator import parse_structured_output
from src.llm.schemas import (
    AnswerSchema,
    ClarificationQuestionSchema,
    IntentModelSchema,
    StrategyDecisionSchema,
)


def test_parse_structured_output_recovers_intent_model_from_list():
    raw = """
    [
      {
        "description": "The user wants a vegetarian dinner for four people",
        "assumed_context": "Needs something quick",
        "plausibility": 0.6
      },
      {
        "description": "The user wants a gluten-free dinner idea",
        "plausibility": 0.3
      }
    ]
    """

    parsed = parse_structured_output(raw, IntentModelSchema)

    assert len(parsed.interpretations) == 2
    assert parsed.most_likely_index == 0
    assert parsed.entropy_estimate == "medium"


def test_parse_structured_output_recovers_strategy_from_string():
    parsed = parse_structured_output('"ask_clarification"', StrategyDecisionSchema)

    assert parsed.strategy == "ask_clarification"
    assert parsed.confidence == 0.5


def test_parse_structured_output_recovers_clarification_from_string():
    parsed = parse_structured_output(
        '"What kind of trip are you planning and what constraints do you have?"',
        ClarificationQuestionSchema,
    )

    assert "What kind of trip" in parsed.question


def test_parse_structured_output_recovers_partial_clarification_object():
    raw = """
    {
      "question": "What specific issue in the paragraph do you want feedback on?",
      "target_variable": "feedback focus"
    }
    """

    parsed = parse_structured_output(raw, ClarificationQuestionSchema)

    assert parsed.target_variable == "feedback focus"
    assert "missing feedback focus" in parsed.why_this_helps


def test_parse_structured_output_recovers_answer_from_string():
    parsed = parse_structured_output(
        '"Plan a three-day Shenandoah hiking trip with budget lodging."',
        AnswerSchema,
    )

    assert "Shenandoah" in parsed.answer


def test_parse_structured_output_recovers_truncated_answer_object():
    raw = """
    {
      "answer": "Connect the branch office over a VPN with failover for network outages.",
      "ass
    """

    parsed = parse_structured_output(raw, AnswerSchema)

    assert "branch office" in parsed.answer
    assert parsed.confidence == 0.5


def test_parse_structured_output_recovers_plain_text_answer():
    parsed = parse_structured_output(
        "Use a high-yield savings account for the emergency fund.",
        AnswerSchema,
    )

    assert "high-yield" in parsed.answer


def test_parse_structured_output_recovers_answer_from_properties_wrapper():
    raw = """
    {
      "properties": {
        "answer": "Use the latest sales totals and segment by region.",
        "confidence": 0.98,
        "caveats": null
      }
    }
    """

    parsed = parse_structured_output(raw, AnswerSchema)

    assert "sales totals" in parsed.answer
    assert parsed.confidence == 0.98


def test_parse_structured_output_recovers_partial_strategy_object():
    raw = """
    {
      "strategy": "ask_clarification"
    }
    """

    parsed = parse_structured_output(raw, StrategyDecisionSchema)

    assert parsed.strategy == "ask_clarification"
    assert parsed.confidence == 0.5


def test_parse_structured_output_recovers_json_like_intent_object():
    raw = """
    {
      interpretations: [
        {
          description: 'The user wants a quick vegetarian dinner',
          assumed_context: 'Dinner for four on a weekday',
          plausibility: 0.7,
        },
        {
          description: 'The user wants a gluten-free dinner',
          assumed_context: 'Dietary restriction',
          plausibility: 0.2,
        },
      ],
      most_likely_index: 0,
      entropy_estimate: 'medium',
      gap_description: 'Dietary constraints and timeline are underspecified.',
    }
    """

    parsed = parse_structured_output(raw, IntentModelSchema)

    assert parsed.most_likely_index == 0
    assert len(parsed.interpretations) == 2
    assert parsed.interpretations[0].description.startswith("The user wants")
