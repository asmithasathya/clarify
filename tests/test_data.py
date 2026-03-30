from src.data.housingqa import load_housingqa_questions, normalize_housingqa_question


def test_normalize_housingqa_question():
    record = {
        "idx": 1,
        "question": "Can a landlord enter without notice?",
        "answer": "no",
        "state": "California",
        "statutes": [
            {
                "statute_idx": 101,
                "citation": "CA Civ. Code 1954",
                "excerpt": "Entry generally requires notice.",
            }
        ],
    }
    example = normalize_housingqa_question(record, 0)
    assert example.example_id == "1"
    assert example.answer == "No"
    assert example.state == "California"
    assert example.statutes == ["101"]
    assert example.citation == ["CA Civ. Code 1954"]


def test_load_housingqa_questions_with_injected_loader():
    def fake_loader(*args, **kwargs):
        return [
            {
                "question_id": "q-2",
                "question": "Can the landlord shut off utilities?",
                "answer": "No",
                "state": "New York",
                "statutes": ["NY-002"],
                "citation": ["NY Real Prop. Law 235"],
                "excerpt": ["Utility shutoffs are restricted."],
            }
        ]

    examples = load_housingqa_questions(loader=fake_loader)
    assert len(examples) == 1
    assert examples[0].example_id == "q-2"
    assert examples[0].answer == "No"
