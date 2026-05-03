"""Run method ablations and save separate output folders."""

from __future__ import annotations

import copy
from pathlib import Path

import typer

from src.eval.runner import run_experiment
from src.utils.config import apply_ablation, load_config


app = typer.Typer(add_completion=False)

@app.command()
def main(
    ablation: list[str] = typer.Option(..., help="Ablation names to run."),
    config: str = typer.Option("configs/default.yaml", help="Base config path."),
    dataset: str | None = typer.Option(None, help="Optional dataset override."),
    split: str | None = typer.Option(None, help="Optional split override."),
    limit: int | None = typer.Option(None, help="Optional example limit."),
    output_dir: str | None = typer.Option(None, help="Optional ablation output root."),
    method: str = typer.Option("resample_clarify", help="Method to ablate."),
) -> None:
    base_config = load_config(config)
    if dataset is not None:
        base_config["data"]["primary_dataset"] = dataset
    if split is not None:
        base_config["data"]["split"] = split

    root = Path(output_dir) if output_dir else Path(base_config["paths"]["outputs_dir"]) / "ablations"
    root.mkdir(parents=True, exist_ok=True)

    for name in ablation:
        resolved = apply_ablation(copy.deepcopy(base_config), name)
        payload = run_experiment(
            config=resolved,
            methods=[method],
            limit=limit,
            output_root=root / name,
        )
        typer.echo(f"Completed ablation {name}. Outputs saved to {payload['output_dir']}")


if __name__ == "__main__":
    app()
