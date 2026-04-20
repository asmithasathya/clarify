"""Run a single method on InfoQuest examples."""

from __future__ import annotations

from pathlib import Path

import typer

from src.eval.runner import build_resources, build_run_manifest, load_examples, run_method
from src.utils.config import load_config
from src.utils.io import ensure_dir


app = typer.Typer(add_completion=False)


@app.command()
def main(
    method: str = typer.Option(..., help="Method: direct_answer, generic_hedge, generic_clarify, targeted_clarify, resample_clarify"),
    config: str = typer.Option("configs/default.yaml", help="Base config path."),
    dataset: str | None = typer.Option(None, help="Optional dataset override."),
    split: str | None = typer.Option(None, help="Optional split override."),
    limit: int | None = typer.Option(None, help="Optional example limit."),
    output_dir: str | None = typer.Option(None, help="Optional output directory."),
) -> None:
    resolved = load_config(config)
    if dataset is not None:
        resolved["data"]["primary_dataset"] = dataset
    if split is not None:
        resolved["data"]["split"] = split

    examples = load_examples(resolved, limit=limit)
    resources = build_resources(resolved)
    out = ensure_dir(Path(output_dir) if output_dir else Path("outputs") / method)
    run_manifest = build_run_manifest(resolved, examples, [method])

    payload = run_method(
        method,
        config=resolved,
        examples=examples,
        resources=resources,
        output_dir=out,
        nest_method_dir=False,
        run_manifest=run_manifest,
    )
    typer.echo(f"Completed {method}. Outputs saved to {out}")
    for key, val in payload["metrics"].items():
        if isinstance(val, float):
            typer.echo(f"  {key}: {val:.3f}")


if __name__ == "__main__":
    app()
