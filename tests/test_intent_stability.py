from src.llm.schemas import (
    AmbiguityAssessmentSchema,
    ClarificationQuestionSchema,
    IntentModelSchema,
    StrategyDecisionSchema,
)
from src.understand.confidence_calibrator import ConfidenceExample, IntentConfidenceCalibrator
from src.understand.intent_stability import (
    aggregate_ambiguity_samples,
    aggregate_clarification_targets,
    aggregate_intent_samples,
    aggregate_strategy_samples,
    build_stability_report,
    select_weakest_stage,
)


def test_intent_stability_pipeline_units():
    ambiguity = aggregate_ambiguity_samples(
        [
            AmbiguityAssessmentSchema(
                is_ambiguous=True,
                ambiguity_type="missing_context",
                missing_variables=[{"variable": "time horizon", "why_missing": "not provided", "importance": 0.9}],
                confidence=0.8,
                rationale="Need the user's time horizon.",
            ),
            AmbiguityAssessmentSchema(
                is_ambiguous=True,
                ambiguity_type="missing_context",
                missing_variables=[{"variable": "time horizon", "why_missing": "not provided", "importance": 0.8}],
                confidence=0.7,
                rationale="Need the user's time horizon.",
            ),
        ]
    )
    intent = aggregate_intent_samples(
        [
            IntentModelSchema(
                interpretations=[
                    {"description": "Long-term investing", "assumed_context": "retirement", "plausibility": 0.7},
                    {"description": "Short-term trading", "assumed_context": "speculation", "plausibility": 0.2},
                ],
                most_likely_index=0,
                entropy_estimate="medium",
                gap_description="Time horizon is missing.",
            ),
            IntentModelSchema(
                interpretations=[
                    {"description": "Long-term investing", "assumed_context": "retirement", "plausibility": 0.6},
                    {"description": "Saving for a house", "assumed_context": "medium term", "plausibility": 0.25},
                ],
                most_likely_index=0,
                entropy_estimate="medium",
                gap_description="Time horizon is missing.",
            ),
        ]
    )
    strategy = aggregate_strategy_samples(
        [
            StrategyDecisionSchema(strategy="ask_clarification", rationale="Need the time horizon.", confidence=0.8),
            StrategyDecisionSchema(strategy="ask_clarification", rationale="Need the time horizon.", confidence=0.7),
        ]
    )
    clarification = aggregate_clarification_targets(
        [
            ClarificationQuestionSchema(
                question="What time horizon are you investing for?",
                target_variable="time horizon",
                why_this_helps="It resolves the main ambiguity.",
            ),
            ClarificationQuestionSchema(
                question="What time horizon are you investing for?",
                target_variable="time horizon",
                why_this_helps="It resolves the main ambiguity.",
            ),
        ]
    )

    calibrator = IntentConfidenceCalibrator().fit(
        [
            ConfidenceExample(raw_confidence=0.2, success=False),
            ConfidenceExample(raw_confidence=0.8, success=True),
            ConfidenceExample(raw_confidence=0.9, success=True),
        ]
    )
    report = build_stability_report(
        [
            ambiguity.stage_report,
            intent.stage_report,
            strategy.stage_report,
            clarification.stage_report,
        ],
        weak_point_threshold=0.75,
        calibrator=calibrator,
    )

    assert report.overall_confidence > 0.0
    assert report.confidence_band in {"low", "medium", "high"}
    assert select_weakest_stage(report, allowed_stages=["ambiguity_detection", "intent_modeling"]) in {
        None,
        "ambiguity_detection",
        "intent_modeling",
    }


def test_calibrator_round_trip(tmp_path):
    calibrator = IntentConfidenceCalibrator().fit(
        [
            ConfidenceExample(raw_confidence=0.1, success=False),
            ConfidenceExample(raw_confidence=0.9, success=True),
        ]
    )
    path = tmp_path / "calibrator.json"
    calibrator.save(path)
    loaded = IntentConfidenceCalibrator.from_path(path)

    assert loaded.predict(0.9) >= loaded.predict(0.1)
    assert loaded.band(loaded.predict(0.9)) in {"medium", "high"}
