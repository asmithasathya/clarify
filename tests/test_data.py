from src.data.clarifybench_v1 import build_clarifybench_v1
from src.data.infoquest import normalize_infoquest_example
from src.data.report_data import split_infoquest_examples, validate_examples
from src.data.schema import DialogueExample


def test_normalize_infoquest_example():
    seed = {
        "id": 42,
        "seed_message": "Help me plan a trip",
        "persona1": {"id": 1, "persona": "A frequent traveler"},
        "persona2": {"id": 2, "persona": "A homebody"},
        "persona3": {"id": 3, "persona": "A chef"},
    }
    setting = {
        "goal": "Plan a hiking trip in Patagonia",
        "obstacle": "Budget is limited to $3000",
        "constraints": ["Must be in December", "Needs gear rental", "Solo trip"],
        "solution": "Book a guided group trek with gear included.",
        "checklist": ["Did the assistant ask about budget?", "Did the assistant ask about timing?"],
        "description": "Adventure travel planning",
        "persona": "A frequent traveler",
    }

    example = normalize_infoquest_example(seed, setting, 1)

    assert isinstance(example, DialogueExample)
    assert example.example_id == "infoquest-42-s1"
    assert example.user_request == "Help me plan a trip"
    assert "Patagonia" in example.hidden_context
    assert example.gold_clarification_needed is True
    assert example.gold_answer == "Book a guided group trek with gear included."
    assert len(example.checklist) == 2
    assert len(example.personas) == 3


def test_normalize_infoquest_example_minimal():
    seed = {"id": 1, "seed_message": "I need help"}
    setting = {"description": "general request"}

    example = normalize_infoquest_example(seed, setting, 1)

    assert example.example_id == "infoquest-1-s1"
    assert example.user_request == "I need help"
    assert example.gold_clarification_needed is True


def test_load_infoquest_local(tmp_path):
    data_file = tmp_path / "test.jsonl"
    ex = DialogueExample(
        example_id="local-1",
        user_request="Fix my code",
        hidden_context="Python indentation error",
        gold_clarification_needed=True,
    )
    data_file.write_text(ex.model_dump_json() + "\n")

    from src.data.infoquest import load_infoquest_local

    loaded = load_infoquest_local(data_file)
    assert len(loaded) == 1
    assert loaded[0].example_id == "local-1"
    assert loaded[0].user_request == "Fix my code"


def test_load_clarifybench_local():
    from src.data.clarifybench import load_clarifybench_local

    loaded = load_clarifybench_local("data/clarifybench_v1_full.jsonl", limit=2)
    assert len(loaded) == 2
    assert loaded[0].example_id == "clarifybench-v1-001"
    assert loaded[0].simulated_user_reply is not None


def test_build_clarifybench_v1_counts():
    rows = build_clarifybench_v1()

    assert len(rows) == 120
    assert sum(1 for row in rows if row.split_name == "dev") == 30
    assert sum(1 for row in rows if row.split_name == "test") == 90
    stats = validate_examples(rows, expected_dataset="clarifybench", require_split=True)
    assert set(stats["by_ambiguity_type"]) == {"lexical", "missing_context", "referential", "underspecified"}
    assert len(stats["by_domain"]) == 5


def test_split_infoquest_examples_keeps_seed_groups_together():
    examples = []
    for seed_id in range(5):
        for setting_index in (1, 2):
            examples.append(
                DialogueExample(
                    example_id=f"infoquest-{seed_id}-s{setting_index}",
                    dataset_name="infoquest",
                    user_request="Help me decide",
                    hidden_context="Need a clarifying question",
                    gold_clarification_needed=True,
                    metadata={"seed_id": seed_id},
                )
            )

    splits = split_infoquest_examples(examples, train_fraction=0.4, dev_fraction=0.2, seed=7)
    assert len(splits["train"]) + len(splits["dev"]) + len(splits["test"]) == 10

    split_by_seed: dict[int, set[str]] = {}
    for split_name, rows in splits.items():
        for row in rows:
            split_by_seed.setdefault(row.metadata["seed_id"], set()).add(split_name)
    assert all(len(split_names) == 1 for split_names in split_by_seed.values())
