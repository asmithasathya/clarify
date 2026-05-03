from src.data.schema import MethodResult
from src.train.distillation import build_preference_pair, build_sft_records


def test_build_sft_records_from_resample_rollout():
    result = MethodResult(
        example_id="infoquest-1-s1",
        dataset_name="infoquest",
        split_name="train",
        method="resample_clarify",
        user_request="Help me invest my money",
        hidden_context="Saving for retirement.",
        gold_clarification_needed=True,
        gold_answer="Use diversified long-term investing.",
        response_strategy="ask_clarification",
        response_text="What time horizon are you investing for?",
        clarification_question="What time horizon are you investing for?",
        intent_confidence=0.72,
        confidence_band="medium",
        trace={
            "stabilized_ambiguity": {
                "is_ambiguous": True,
                "ambiguity_type": "missing_context",
                "missing_variables": [
                    {"variable": "time horizon", "why_missing": "not provided", "importance": 0.9}
                ],
                "confidence": 0.8,
                "rationale": "Time horizon is missing.",
            },
            "stabilized_intent": {
                "interpretations": [
                    {"description": "Retirement investing", "assumed_context": "long term", "plausibility": 0.7},
                    {"description": "Short-term speculation", "assumed_context": "high risk", "plausibility": 0.2},
                ],
                "most_likely_index": 0,
                "entropy_estimate": "medium",
                "gap_description": "Time horizon is missing.",
            },
            "stabilized_strategy": {
                "strategy": "ask_clarification",
                "rationale": "Need the missing time horizon.",
                "confidence": 0.8,
            },
            "stabilized_clarification_target": {
                "question": "What time horizon are you investing for?",
                "target_variable": "time horizon",
                "why_this_helps": "It resolves the main ambiguity.",
            },
        },
    )

    records = build_sft_records(result, rollout_id="rollout-1")
    stage_names = {record["stage_name"] for record in records}
    assert "ambiguity_detection" in stage_names
    assert "intent_modeling" in stage_names
    assert "strategy_selection" in stage_names
    assert "clarification_generation" in stage_names


def test_build_preference_pair_uses_teacher_and_rejected_outputs():
    teacher = MethodResult(
        example_id="clarifybench-v1-001",
        dataset_name="clarifybench",
        split_name="test",
        method="resample_clarify",
        user_request="Book my flight for next Friday",
        hidden_context="Need the flight for Friday next week, not this week.",
        gold_clarification_needed=True,
        response_strategy="ask_clarification",
        response_text="Do you mean next Friday as in April 24?",
        correct=True,
    )
    rejected = MethodResult(
        example_id="clarifybench-v1-001",
        dataset_name="clarifybench",
        split_name="test",
        method="direct_answer",
        user_request="Book my flight for next Friday",
        hidden_context="Need the flight for Friday next week, not this week.",
        gold_clarification_needed=True,
        response_strategy="answer_directly",
        response_text="Booked for this Friday.",
        correct=False,
    )

    pair = build_preference_pair(teacher, rejected, rollout_id="rollout-2")
    assert pair is not None
    assert pair["chosen_strategy"] == "ask_clarification"
    assert pair["rejected_strategy"] == "answer_directly"
