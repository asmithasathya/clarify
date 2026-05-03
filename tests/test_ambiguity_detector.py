from src.understand.ambiguity_detector import AmbiguityDetector, heuristic_ambiguity


def test_heuristic_detects_short_vague_request():
    result = heuristic_ambiguity("Help me with this")
    assert result.is_ambiguous is True
    assert len(result.missing_variables) >= 1


def test_heuristic_passes_specific_request():
    result = heuristic_ambiguity(
        "What is the capital of France and what is its population as of 2023?"
    )
    assert result.is_ambiguous is False


def test_detector_ablation_disabled():
    config = {"ablations": {"ambiguity_detection": False}}
    detector = AmbiguityDetector(config, generator=None)
    result = detector.detect("Help me with something")
    assert result.is_ambiguous is False
    assert "Ablation" in result.rationale


def test_detector_heuristic_fallback():
    config = {"ablations": {"ambiguity_detection": True}}
    detector = AmbiguityDetector(config, generator=None)
    result = detector.detect("Fix it")
    assert result.is_ambiguous is True


def test_to_records():
    config = {"ablations": {"ambiguity_detection": True}}
    detector = AmbiguityDetector(config, generator=None)
    assessment = detector.detect("Help me with stuff")
    records = detector.to_records(assessment)
    assert len(records) >= 1
    assert records[0].missing_variable
