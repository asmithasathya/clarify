"""Run targeted_clarify ablations and save separate output folders."""

from __future__ import annotations

import copy
from pathlib import Path

import typer

from src.eval.runner import run_experiment
from src.utils.config import load_config


app = typer.Typer(add_completion=False)

ABLATION_MAP: dict[str, tuple[str, bool]] = {
    "no_ambiguity_detection": ("ambiguity_detection", False),
    "no_intent_modeling": ("intent_modeling", False),
    "no_strategy_selection": ("strategy_selection", False),
    "no_targeted_question": ("targeted_question", False),
}


def apply_ablation(config: dict, name: str) -> dict:
    result = copy.deepcopy(config)
    if name not in ABLATION_MAP:
        raise ValueError(f"Unknown ablation: {name}. Choices: {list(ABLATION_MAP)}")
    key, value = ABLATION_MAP[name]
    result.setdefault("ablations", {})[key] = value
    return result


@app.command()
def main(
    ablation: list[str] = typer.Option(..., help="Ablation names to run."),
    config: str = typer.Option("configs/default.yaml", help="Base config path."),
    limit: int | None = typer.Option(None, help="Optional example limit."),
    output_dir: str | None = typer.Option(None, help="Optional ablation output root."),
    backend: str | None = typer.Option(None, help="Optional generator backend override."),
) -> None:
    base_config = load_config(config)
    if backend is not None:
        base_config["model"]["generator"]["backend"] = backend

    root = Path(output_dir) if output_dir else Path(base_config["paths"]["outputs_dir"]) / "ablations"
    root.mkdir(parents=True, exist_ok=True)

    for name in ablation:
        resolved = apply_ablation(base_config, name)
        payload = run_experiment(
            config=resolved,
            methods=["targeted_clarify"],
            limit=limit,
            output_root=root / name,
        )
        typer.echo(f"Completed ablation {name}. Outputs saved to {payload['output_dir']}")


if __name__ == "__main__":
    app()
