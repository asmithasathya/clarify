"""Prepare or run the optional DPO stage for the distilled student."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import ensure_dir, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional DPO stage for the distilled student.")
    parser.add_argument("--config", default="configs/report_baselines.yaml")
    parser.add_argument("--preference-corpus", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/student_dpo")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    pairs = read_jsonl(args.preference_corpus)
    manifest = {
        "preference_corpus": args.preference_corpus,
        "sft_checkpoint": args.sft_checkpoint,
        "n_pairs": len(pairs),
        "dpo_beta": config.get("post_training", {}).get("beta", 0.1),
        "status": "prepared",
    }
    write_json(manifest, output_dir / "student_dpo_manifest.json")

    if args.prepare_only:
        print(f"Prepared DPO manifest at {output_dir / 'student_dpo_manifest.json'}")
        return

    try:
        import tinker_cookbook  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "DPO execution requires the optional 'tinker-cookbook' package. "
            "Install it and rerun this script, or use --prepare-only to generate the manifest now."
        ) from exc

    raise SystemExit(
        "DPO preparation is implemented, but execution currently requires a local tinker-cookbook workflow. "
        "Use the generated manifest and preference corpus to launch the cookbook-based DPO run."
    )


if __name__ == "__main__":
    main()
