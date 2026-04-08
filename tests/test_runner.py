import json

from src.data.schema import DialogueExample, MethodResult
from src.eval.metrics import compute_all_metrics
from src.eval.runner import run_method
from src.llm.generator import MockGenerator


def test_run_method_scores_direct_answer_before_metrics(tmp_path):
    config = {
        "generation": {"json_max_retries": 0},
        "evaluation": {"answer_match_threshold": 0.5},
    }
    example = DialogueExample(
        example_id="capital-1",
        user_request="What is the capital of France?",
        hidden_context="None",
        gold_clarification_needed=False,
        gold_answer="Paris",
    )

    def responder(messages):
        prompt = messages[-1]["content"]
        if "Answer the following user request directly" not in prompt:
            raise AssertionError(f"Unexpected prompt: {prompt[:120]}")
        return json.dumps({
            "answer": "Paris",
            "assumed_interpretation": None,
            "confidence": 0.95,
            "caveats": None,
        })

    payload = run_method(
        "direct_answer",
        config=config,
        examples=[example],
        resources={"generator": MockGenerator(responder=responder, config=config)},
        output_dir=tmp_path,
    )

    assert payload["results"][0].correct is True
    assert payload["metrics"]["task_success_rate"] == 1.0
    assert payload["metrics"]["wrong_answer_under_ambiguity"] == 0.0


def test_present_alternatives_counts_as_appropriate_action():
    result = MethodResult(
        example_id="alt-1",
        method="targeted_clarify",
        user_request="Help me invest my money",
        hidden_context="User wants long-term savings advice.",
        gold_clarification_needed=True,
        gold_answer="Use a diversified long-term strategy.",
        response_strategy="present_alternatives",
        response_text="This request could mean a few different things.",
        is_ambiguous_detected=True,
        asked_clarification=False,
    )

    metrics = compute_all_metrics([result])

    assert metrics["appropriate_action_rate"] == 1.0
    assert metrics["clarification_rate"] == 0.0
