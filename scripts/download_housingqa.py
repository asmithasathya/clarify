"""Download and cache HousingQA questions and statutes locally."""

from __future__ import annotations

import typer

from src.data.housingqa import load_housingqa_questions, load_housingqa_statutes
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_jsonl


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option("configs/default.yaml", help="Base config path."),
    question_limit: int | None = typer.Option(None, help="Optional limit for questions."),
    statute_limit: int | None = typer.Option(None, help="Optional limit for statute passages."),
) -> None:
    resolved = load_config(config)
    housing_cfg = resolved["data"]["housingqa"]
    output_dir = ensure_dir(housing_cfg["local_cache_dir"])

    questions = load_housingqa_questions(
        dataset_name=housing_cfg["questions_dataset"],
        config_name=housing_cfg["questions_config"],
        split=housing_cfg["questions_split"],
        limit=question_limit,
    )
    statutes = load_housingqa_statutes(
        dataset_name=housing_cfg["statutes_dataset"],
        config_name=housing_cfg["statutes_config"],
        split=housing_cfg["statutes_split"],
        limit=statute_limit,
    )

    write_jsonl(questions, output_dir / "questions_test.jsonl")
    write_jsonl(statutes, output_dir / "statutes_corpus.jsonl")
    typer.echo(f"Saved {len(questions)} questions and {len(statutes)} statutes to {output_dir}")


if __name__ == "__main__":
    app()

