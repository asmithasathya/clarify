from src.llm.schemas import AmbiguityAssessmentSchema
from src.llm.generator import MockGenerator
from src.understand.intent_model import IntentModeler, heuristic_intent_model


def _make_assessment(is_ambiguous=True):
    return AmbiguityAssessmentSchema(
        is_ambiguous=is_ambiguous,
        ambiguity_type="underspecified" if is_ambiguous else "none",
        missing_variables=[
            {"variable": "user goal", "why_missing": "not stated", "importance": 0.8}
        ] if is_ambiguous else [],
        confidence=0.7,
        rationale="test",
    )


def test_heuristic_intent_model_ambiguous():
    assessment = _make_assessment(is_ambiguous=True)
    result = heuristic_intent_model("Help me with something", assessment)
    assert len(result.interpretations) >= 1
    assert result.entropy_estimate == "high"
    assert "user goal" in result.gap_description


def test_heuristic_intent_model_clear():
    assessment = _make_assessment(is_ambiguous=False)
    result = heuristic_intent_model("What is 2+2?", assessment)
    assert result.entropy_estimate == "low"


def test_modeler_ablation_disabled():
    config = {"ablations": {"intent_modeling": False}}
    modeler = IntentModeler(config, generator=None)
    assessment = _make_assessment()
    result = modeler.model("Help me", assessment)
    assert len(result.interpretations) >= 1


def test_modeler_heuristic_fallback():
    config = {"ablations": {"intent_modeling": True}}
    modeler = IntentModeler(config, generator=None)
    assessment = _make_assessment()
    result = modeler.model("Help me", assessment)
    assert result.gap_description


def test_modeler_falls_back_when_generation_fails():
    config = {"ablations": {"intent_modeling": True}, "generation": {"json_max_retries": 0}}

    def _raising_responder(_messages):
        raise RuntimeError("bad structured output")

    modeler = IntentModeler(config, generator=MockGenerator(responder=_raising_responder, config=config))
    assessment = _make_assessment()
    result = modeler.model("Help me choose a plan", assessment)
    assert len(result.interpretations) >= 1
    assert result.gap_description
