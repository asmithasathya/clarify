"""Run a single method on InfoQuest examples."""

from __future__ import annotations

from pathlib import Path

import typer

from src.eval.runner import build_resources, load_examples, run_method
from src.utils.config import load_config
from src.utils.io import ensure_dir


app = typer.Typer(add_completion=False)


@app.command()
def main(
    method: str = typer.Option(..., help="Method: direct_answer, generic_hedge, generic_clarify, targeted_clarify"),
    config: str = typer.Option("configs/default.yaml", help="Base config path."),
    limit: int | None = typer.Option(None, help="Optional example limit."),
    output_dir: str | None = typer.Option(None, help="Optional output directory."),
    backend: str | None = typer.Option(None, help="Optional generator backend override."),
) -> None:
    resolved = load_config(config)
    if backend is not None:
        resolved["model"]["generator"]["backend"] = backend

    examples = load_examples(resolved, limit=limit)
    resources = build_resources(resolved)
    out = ensure_dir(Path(output_dir) if output_dir else Path("outputs") / method)

    payload = run_method(method, config=resolved, examples=examples, resources=resources, output_dir=out)
    typer.echo(f"Completed {method}. Outputs saved to {out}")
    for key, val in payload["metrics"].items():
        if isinstance(val, float):
            typer.echo(f"  {key}: {val:.3f}")


if __name__ == "__main__":
    app()
