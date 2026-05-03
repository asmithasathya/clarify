import json

from src.data.schema import DialogueExample, MethodResult
from src.data.report_data import sha256_file
from src.eval.metrics import compute_all_metrics
from src.eval.runner import run_method
from src.llm.generator import MockGenerator
from src.understand.clarification_generator import ClarificationGenerator
from src.utils.io import read_csv


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


def test_run_method_completes_multi_turn_after_clarification(tmp_path):
    config = {
        "generation": {"json_max_retries": 0},
        "understand": {"max_clarification_turns": 2},
        "evaluation": {
            "answer_match_threshold": 0.5,
            "user_reply_simulation": {"use_generator": False},
            "llm_judge": {"enabled": False},
        },
    }
    example = DialogueExample(
        example_id="travel-1",
        user_request="Help me plan a trip",
        hidden_context="Three-day hiking trip in Shenandoah next month, budget under $500, no camping.",
        simulated_user_reply="I want a three-day Shenandoah hiking trip next month, under $500, and no camping.",
        gold_clarification_needed=True,
        gold_answer="Plan a three-day Shenandoah hiking trip next month with budget lodging and day hikes instead of camping.",
    )

    def responder(messages):
        prompt = messages[-1]["content"]
        if "Before answering the following user request" in prompt:
            return json.dumps({
                "question": "What kind of trip are you planning and what constraints do you have?",
                "target_variable": "trip type and constraints",
                "why_this_helps": "It reveals the destination, budget, and lodging needs.",
            })
        if "Continue the conversation below" in prompt:
            return json.dumps({
                "answer": "Plan a three-day Shenandoah hiking trip next month with budget lodging and day hikes instead of camping.",
                "assumed_interpretation": None,
                "confidence": 0.92,
                "caveats": None,
            })
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")

    payload = run_method(
        "generic_clarify",
        config=config,
        examples=[example],
        resources={
            "generator": MockGenerator(responder=responder, config=config),
            "clarification_generator": ClarificationGenerator(
                config,
                generator=MockGenerator(responder=responder, config=config),
            ),
            "llm_judge": None,
        },
        output_dir=tmp_path,
        nest_method_dir=False,
    )

    result = payload["results"][0]
    assert result.asked_clarification is True
    assert result.answered_after_clarification is True
    assert result.final_answer is not None
    assert result.num_turns == 3
    assert payload["metrics"]["multi_turn_completion_rate"] == 1.0
    assert (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "generic_clarify").exists()


def test_run_experiment_skips_completed_method_outputs(tmp_path, monkeypatch):
    import src.eval.runner as runner_module

    config = {
        "project": {"seed": 42, "prompt_version": "baseline-v1"},
        "generation": {"json_max_retries": 0},
        "understand": {"max_clarification_turns": 1},
        "paths": {"outputs_dir": str(tmp_path)},
        "data": {
            "primary_dataset": "clarifybench",
            "split": "test",
            "clarifybench": {"local_path": str(tmp_path / "clarifybench.jsonl")},
        },
        "evaluation": {
            "answer_match_threshold": 0.5,
            "llm_judge": {"enabled": False, "version": "judge-v1"},
        },
        "model": {"generator": {"backend": "mock", "base_model": "mock-model"}},
    }
    example = DialogueExample(
        example_id="resume-1",
        dataset_name="clarifybench",
        split_name="test",
        user_request="What is the capital of France?",
        hidden_context="None",
        gold_clarification_needed=False,
        gold_answer="Paris",
    )
    (tmp_path / "clarifybench.jsonl").write_text(example.model_dump_json() + "\n", encoding="utf-8")

    completed_dir = tmp_path / "run" / "direct_answer"
    completed_dir.mkdir(parents=True)
    (completed_dir / "metrics.json").write_text(
        json.dumps(
            {
                "task_success_rate": 1.0,
                "appropriate_action_rate": 1.0,
                "clarification_rate": 0.0,
                "answer_rate": 1.0,
                "final_answer_rate": 1.0,
                "abstention_rate": 0.0,
                "clarification_precision": 0.0,
                "clarification_recall": 0.0,
                "ambiguity_detection_accuracy": 1.0,
                "unnecessary_clarification_rate": 0.0,
                "missed_ambiguity_rate": 0.0,
                "wrong_answer_under_ambiguity": 0.0,
                "multi_turn_completion_rate": 0.0,
                "average_answer_score": 1.0,
                "average_clarification_quality": 0.0,
                "average_alternatives_quality": 0.0,
                "strategy_distribution": {"answer_directly": 1},
                "n_examples": 1,
            }
        ),
        encoding="utf-8",
    )
    (completed_dir / "manifest.json").write_text(
        json.dumps(
            {
                    "dataset_name": "clarifybench",
                    "split_name": "test",
                    "dataset_sha256": sha256_file(tmp_path / "clarifybench.jsonl"),
                    "task_model_id": "mock-model",
                    "judge_model_id": None,
                    "prompt_version": "baseline-v1",
                "judge_version": "judge-v1",
            }
        ),
        encoding="utf-8",
    )

    def responder(messages):
        prompt = messages[-1]["content"]
        if "cautious hedging" not in prompt:
            raise AssertionError(f"Unexpected prompt: {prompt[:100]}")
        return json.dumps(
            {
                "answer": "Paris",
                "hedge_reason": "Low ambiguity in this example.",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(
        runner_module,
        "build_resources",
        lambda config: {"generator": MockGenerator(responder=responder, config=config), "llm_judge": None},
    )

    payload = runner_module.run_experiment(
        config=config,
        methods=["direct_answer", "generic_hedge"],
        output_root=tmp_path / "run",
    )

    assert payload["metrics"]["direct_answer"]["task_success_rate"] == 1.0
    assert (tmp_path / "run" / "generic_hedge" / "metrics.json").exists()


def test_run_experiment_resume_writes_union_metric_columns(tmp_path, monkeypatch):
    import src.eval.runner as runner_module

    config = {
        "project": {"seed": 42, "prompt_version": "baseline-v1"},
        "generation": {"json_max_retries": 0},
        "understand": {"max_clarification_turns": 1},
        "paths": {"outputs_dir": str(tmp_path)},
        "data": {
            "primary_dataset": "clarifybench",
            "split": "test",
            "clarifybench": {"local_path": str(tmp_path / "clarifybench.jsonl")},
        },
        "evaluation": {
            "answer_match_threshold": 0.5,
            "llm_judge": {"enabled": False, "version": "judge-v1"},
        },
        "model": {"generator": {"backend": "mock", "base_model": "mock-model"}},
    }
    example = DialogueExample(
        example_id="resume-2",
        dataset_name="clarifybench",
        split_name="test",
        user_request="Book me a table for Friday.",
        hidden_context="The user did not specify the restaurant or party size.",
        gold_clarification_needed=True,
        gold_answer="Need the restaurant and party size before booking.",
    )
    (tmp_path / "clarifybench.jsonl").write_text(example.model_dump_json() + "\n", encoding="utf-8")

    completed_dir = tmp_path / "run" / "direct_answer"
    completed_dir.mkdir(parents=True)
    (completed_dir / "metrics.json").write_text(
        json.dumps(
            {
                "task_success_rate": 0.0,
                "appropriate_action_rate": 0.0,
                "clarification_rate": 0.0,
                "answer_rate": 1.0,
                "final_answer_rate": 1.0,
                "abstention_rate": 0.0,
                "clarification_precision": 0.0,
                "clarification_recall": 0.0,
                "ambiguity_detection_accuracy": 0.0,
                "unnecessary_clarification_rate": 0.0,
                "missed_ambiguity_rate": 1.0,
                "wrong_answer_under_ambiguity": 1.0,
                "multi_turn_completion_rate": 0.0,
                "average_answer_score": 0.0,
                "average_clarification_quality": 0.0,
                "average_alternatives_quality": 0.0,
                "strategy_distribution": {"answer_directly": 1},
                "n_examples": 1,
            }
        ),
        encoding="utf-8",
    )
    (completed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "clarifybench",
                "split_name": "test",
                "dataset_sha256": sha256_file(tmp_path / "clarifybench.jsonl"),
                "task_model_id": "mock-model",
                "judge_model_id": None,
                "prompt_version": "baseline-v1",
                "judge_version": "judge-v1",
            }
        ),
        encoding="utf-8",
    )

    def responder(messages):
        prompt = messages[-1]["content"]
        if "generate a clarifying question" not in prompt:
            raise AssertionError(f"Unexpected prompt: {prompt[:100]}")
        return json.dumps(
            {
                "question": "Which restaurant and for how many people?",
                "target_variable": "restaurant and party size",
                "why_this_helps": "Those details are needed before the booking can be completed.",
            }
        )

    monkeypatch.setattr(
        runner_module,
        "build_resources",
        lambda config: {"generator": MockGenerator(responder=responder, config=config), "llm_judge": None},
    )

    runner_module.run_experiment(
        config=config,
        methods=["direct_answer", "generic_clarify"],
        output_root=tmp_path / "run",
    )

    all_metrics = read_csv(tmp_path / "run" / "all_metrics.csv")
    assert len(all_metrics) == 2
    assert "strategy_distribution.answer_directly" in all_metrics[0]
    assert "strategy_distribution.ask_clarification" in all_metrics[0]


def test_run_method_resumes_from_partial_predictions(tmp_path):
    config = {
        "project": {"prompt_version": "baseline-v1"},
        "generation": {"json_max_retries": 0},
        "evaluation": {"answer_match_threshold": 0.5, "llm_judge": {"enabled": False, "version": "judge-v1"}},
        "model": {"generator": {"backend": "mock", "base_model": "mock-model"}},
    }
    example_one = DialogueExample(
        example_id="resume-partial-1",
        dataset_name="clarifybench",
        split_name="test",
        user_request="What is the capital of France?",
        hidden_context="None",
        gold_clarification_needed=False,
        gold_answer="Paris",
    )
    example_two = DialogueExample(
        example_id="resume-partial-2",
        dataset_name="clarifybench",
        split_name="test",
        user_request="What is the capital of Spain?",
        hidden_context="None",
        gold_clarification_needed=False,
        gold_answer="Madrid",
    )
    method_dir = tmp_path / "direct_answer"
    method_dir.mkdir(parents=True)
    run_manifest = {
        "dataset_name": "clarifybench",
        "split_name": "test",
        "dataset_sha256": None,
        "task_model_id": "mock-model",
        "judge_model_id": None,
        "prompt_version": "baseline-v1",
        "judge_version": "judge-v1",
    }
    (method_dir / "manifest.json").write_text(
        json.dumps({**run_manifest, "method": "direct_answer"}),
        encoding="utf-8",
    )
    partial = MethodResult(
        example_id=example_one.example_id,
        method="direct_answer",
        dataset_name="clarifybench",
        split_name="test",
        user_request=example_one.user_request,
        hidden_context=example_one.hidden_context,
        gold_clarification_needed=example_one.gold_clarification_needed,
        gold_answer=example_one.gold_answer,
        task_model_id="mock-model",
        judge_model_id=None,
        prompt_version="baseline-v1",
        judge_version="judge-v1",
        response_strategy="answer_directly",
        response_text="Paris",
        final_answer="Paris",
        answered_directly=True,
        confidence=0.95,
        correct=True,
    )
    (method_dir / "predictions.jsonl").write_text(partial.model_dump_json() + "\n", encoding="utf-8")

    calls: list[str] = []

    def responder(messages):
        calls.append(messages[-1]["content"])
        return json.dumps(
            {
                "answer": "Madrid",
                "assumed_interpretation": None,
                "confidence": 0.95,
                "caveats": None,
            }
        )

    payload = run_method(
        "direct_answer",
        config=config,
        examples=[example_one, example_two],
        resources={"generator": MockGenerator(responder=responder, config=config), "llm_judge": None},
        output_dir=tmp_path,
        run_manifest=run_manifest,
    )

    assert len(calls) == 1
    assert payload["metrics"]["n_examples"] == 2
    predictions = read_csv(method_dir / "summary.csv")
    assert predictions[0]["n_examples"] == "2"
    saved = (method_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(saved) == 2
