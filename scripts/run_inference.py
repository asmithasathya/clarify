"""Run one method on HousingQA and save outputs."""

from __future__ import annotations

from pathlib import Path

import typer

from src.eval.runner import run_experiment
from src.utils.config import load_config


app = typer.Typer(add_completion=False)


@app.command()
def main(
    method: str = typer.Option(..., help="Method name: closed_book, rag_direct, hedge, revise_verify"),
    config: str = typer.Option("configs/default.yaml", help="Base config path."),
    limit: int | None = typer.Option(None, help="Optional example limit."),
    output_dir: str | None = typer.Option(None, help="Optional output directory."),
    backend: str | None = typer.Option(None, help="Optional generator backend override."),
) -> None:
    resolved = load_config(config, method_name=method)
    if backend is not None:
        resolved["model"]["generator"]["backend"] = backend
    payload = run_experiment(
        config=resolved,
        methods=[method],
        limit=limit,
        output_root=Path(output_dir) if output_dir else None,
    )
    typer.echo(f"Completed {method}. Outputs saved to {payload['output_dir']}")


if __name__ == "__main__":
    app()

