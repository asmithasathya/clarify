import json

from src.data.schema import DialogueExample, MethodResult
from src.eval.metrics import compute_all_metrics
from src.eval.runner import build_run_manifest
from src.eval.sharding import aggregate_sharded_experiment, partition_examples, shard_dir
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl


def _example(example_id: str, request: str, answer: str) -> DialogueExample:
    return DialogueExample(
        example_id=example_id,
        dataset_name="clarifybench",
        split_name="test",
        user_request=request,
        hidden_context="None",
        gold_clarification_needed=False,
        gold_answer=answer,
    )


def test_partition_examples_is_deterministic_and_balanced():
    examples = [
        _example(f"ex-{index}", f"Request {index}", f"Answer {index}")
        for index in range(7)
    ]

    shards = partition_examples(examples, 3)

    assert [[example.example_id for example in shard] for shard in shards] == [
        ["ex-0", "ex-3", "ex-6"],
        ["ex-1", "ex-4"],
        ["ex-2", "ex-5"],
    ]


def test_aggregate_sharded_experiment_writes_standard_leaf_outputs(tmp_path):
    config = {
        "project": {"seed": 42, "prompt_version": "baseline-v1"},
        "evaluation": {"case_study_limit": 2, "llm_judge": {"enabled": False, "version": "judge-v1"}},
        "data": {
            "primary_dataset": "clarifybench",
            "split": "test",
            "clarifybench": {"local_path": str(tmp_path / "clarifybench.jsonl")},
        },
        "model": {"generator": {"backend": "mock", "base_model": "mock-model"}},
    }
    examples = [
        _example("ex-0", "Capital of France?", "Paris"),
        _example("ex-1", "Capital of Spain?", "Madrid"),
        _example("ex-2", "Capital of Italy?", "Rome"),
        _example("ex-3", "Capital of Portugal?", "Lisbon"),
    ]
    (tmp_path / "clarifybench.jsonl").write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )

    methods = ["direct_answer"]
    leaf_root = tmp_path / "leaf"
    shard_sets = partition_examples(examples, 2)

    for shard_index, shard_examples in enumerate(shard_sets):
        shard_root = shard_dir(leaf_root, shard_index, 2)
        run_manifest = build_run_manifest(config, shard_examples, methods)
        write_json(run_manifest, shard_root / "run_manifest.json")

        results = []
        for example in shard_examples:
            results.append(
                MethodResult(
                    example_id=example.example_id,
                    method="direct_answer",
                    dataset_name="clarifybench",
                    split_name="test",
                    user_request=example.user_request,
                    hidden_context=example.hidden_context,
                    gold_clarification_needed=False,
                    gold_answer=example.gold_answer,
                    task_model_id="mock-model",
                    judge_model_id=None,
                    prompt_version="baseline-v1",
                    judge_version="judge-v1",
                    response_strategy="answer_directly",
                    response_text=example.gold_answer or "",
                    final_answer=example.gold_answer,
                    answered_directly=True,
                    confidence=0.95,
                    answer_score=1.0,
                    correct=True,
                )
            )
        method_dir = shard_root / "direct_answer"
        method_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(results, method_dir / "predictions.jsonl")
        write_json(compute_all_metrics(results), method_dir / "metrics.json")
        write_json({**run_manifest, "method": "direct_answer"}, method_dir / "manifest.json")

    payload = aggregate_sharded_experiment(
        config=config,
        methods=methods,
        examples=examples,
        leaf_root=leaf_root,
        shard_count=2,
    )

    assert payload["complete"] is True
    metrics = read_json(leaf_root / "direct_answer" / "metrics.json")
    assert metrics["n_examples"] == 4
    assert metrics["task_success_rate"] == 1.0

    merged = read_jsonl(leaf_root / "direct_answer" / "predictions.jsonl")
    assert [row["example_id"] for row in merged] == ["ex-0", "ex-1", "ex-2", "ex-3"]

    sharding_status = read_json(leaf_root / "sharding.json")
    assert sharding_status["status"] == "complete"
    assert sharding_status["shard_count"] == 2
