from src.data.infoquest import normalize_infoquest_example
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
