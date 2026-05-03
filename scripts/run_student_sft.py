"""Train a modular LoRA student on the exported SFT corpus."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import tinker

from src.train.sft import train_lora_sft
from src.utils.config import load_config
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


def _load_api_key(env_var: str = "TINKER_API_KEY") -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise EnvironmentError(f"Missing {env_var}. Export your Tinker API key before running student SFT.")
    return api_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA SFT for the distilled student.")
    parser.add_argument("--config", default="configs/report_baselines.yaml")
    parser.add_argument("--train-corpus", required=True, help="Path to aggregated sft_corpus.jsonl for train.")
    parser.add_argument("--dev-corpus", default=None, help="Optional path to a dev SFT corpus.")
    parser.add_argument("--output-dir", default="outputs/student_sft")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    distill_cfg = config.get("distillation", {})
    output_dir = ensure_dir(args.output_dir)

    train_records = read_jsonl(args.train_corpus)
    if args.limit is not None:
        train_records = train_records[: args.limit]
    dev_records = read_jsonl(args.dev_corpus) if args.dev_corpus else []
    if args.limit is not None and dev_records:
        dev_records = dev_records[: max(1, args.limit // 4)]

    service_client = tinker.ServiceClient(api_key=_load_api_key())
    payload = train_lora_sft(
        service_client=service_client,
        base_model=distill_cfg.get("student_model", "Qwen/Qwen3-4B-Instruct-2507"),
        train_records=train_records,
        dev_records=dev_records,
        rank=int(distill_cfg.get("lora_rank", 32)),
        seed=int(config.get("project", {}).get("seed", 42)),
        batch_size=int(distill_cfg.get("batch_size", 16)),
        learning_rate=float(distill_cfg.get("learning_rate", 1.0e-4)),
        max_steps=int(distill_cfg.get("max_steps", 200)),
        eval_interval=max(1, int(distill_cfg.get("max_steps", 200)) // 5),
        checkpoint_name=distill_cfg.get("publish_alias") or "final_student_sft",
    )

    write_json(payload, output_dir / "student_sft_manifest.json")
    write_jsonl(payload["history"], output_dir / "student_sft_history.jsonl")
    print(f"Saved student SFT artifacts to {output_dir}")
    print(f"checkpoint_path={payload['checkpoint_path']}")


if __name__ == "__main__":
    main()
