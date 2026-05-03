"""Prepare frozen datasets for report runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.clarifybench_v1 import write_clarifybench_v1
from src.data.infoquest import load_infoquest, load_infoquest_local
from src.data.report_data import split_infoquest_examples, write_dataset_split, write_manifest
from src.utils.io import write_json
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def prepare_infoquest(
    output_dir: Path,
    *,
    refresh: bool = False,
    limit: int | None = None,
    both_settings: bool = True,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
    split_seed: int = 42,
) -> dict[str, object]:
    snapshot_path = output_dir / "infoquest_snapshot.jsonl"
    if snapshot_path.exists() and not refresh:
        examples = load_infoquest_local(snapshot_path)
    else:
        LOGGER.info("Downloading InfoQuest from Hugging Face.")
        examples = load_infoquest(limit=limit, both_settings=both_settings)
        snapshot_info = write_dataset_split(
            examples,
            snapshot_path,
            expected_dataset="infoquest",
            require_split=False,
        )
        LOGGER.info("Wrote InfoQuest snapshot: %s", snapshot_info["path"])

    if limit is not None:
        examples = examples[:limit]

    splits = split_infoquest_examples(
        examples,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
        seed=split_seed,
    )
    manifest_splits = {
        "snapshot": write_dataset_split(
            examples,
            snapshot_path,
            expected_dataset="infoquest",
            require_split=False,
        ),
        "train": write_dataset_split(
            splits["train"],
            output_dir / "infoquest_train.jsonl",
            expected_dataset="infoquest",
            require_split=True,
        ),
        "dev": write_dataset_split(
            splits["dev"],
            output_dir / "infoquest_dev.jsonl",
            expected_dataset="infoquest",
            require_split=True,
        ),
        "test": write_dataset_split(
            splits["test"],
            output_dir / "infoquest_test.jsonl",
            expected_dataset="infoquest",
            require_split=True,
        ),
    }
    stats = {
        "snapshot": manifest_splits["snapshot"]["stats"],
        "train": manifest_splits["train"]["stats"],
        "dev": manifest_splits["dev"]["stats"],
        "test": manifest_splits["test"]["stats"],
    }
    write_json(stats, output_dir / "infoquest_stats.json")
    write_manifest(
        output_dir / "infoquest_manifest.json",
        dataset_name="infoquest",
        splits=manifest_splits,
        extra={
            "both_settings": both_settings,
            "split_seed": split_seed,
            "train_fraction": train_fraction,
            "dev_fraction": dev_fraction,
        },
    )
    return {
        "n_examples": len(examples),
        "train_examples": len(splits["train"]),
        "dev_examples": len(splits["dev"]),
        "test_examples": len(splits["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen report datasets.")
    parser.add_argument("--output-dir", default="data", help="Directory to write dataset files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for InfoQuest snapshot generation.")
    parser.add_argument("--refresh-infoquest", action="store_true", help="Re-download the InfoQuest snapshot even if it exists.")
    parser.add_argument("--single-setting", action="store_true", help="Use only the first InfoQuest setting per seed.")
    parser.add_argument("--skip-infoquest", action="store_true", help="Skip the InfoQuest preparation step.")
    parser.add_argument("--skip-clarifybench", action="store_true", help="Skip the ClarifyBench preparation step.")
    parser.add_argument("--train-fraction", type=float, default=0.6, help="Fraction of seed groups assigned to the train split.")
    parser.add_argument("--dev-fraction", type=float, default=0.2, help="Fraction of seed groups assigned to the dev split.")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for deterministic split assignment.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_infoquest:
        infoquest_summary = prepare_infoquest(
            output_dir,
            refresh=args.refresh_infoquest,
            limit=args.limit,
            both_settings=not args.single_setting,
            train_fraction=args.train_fraction,
            dev_fraction=args.dev_fraction,
            split_seed=args.split_seed,
        )
        LOGGER.info(
            "Prepared InfoQuest with %s examples (%s train / %s dev / %s test).",
            infoquest_summary["n_examples"],
            infoquest_summary["train_examples"],
            infoquest_summary["dev_examples"],
            infoquest_summary["test_examples"],
        )

    if not args.skip_clarifybench:
        clarifybench_summary = write_clarifybench_v1(output_dir)
        LOGGER.info(
            "Prepared ClarifyBench v1 with %s examples (%s dev / %s test).",
            clarifybench_summary["n_examples"],
            clarifybench_summary["dev_examples"],
            clarifybench_summary["test_examples"],
        )


if __name__ == "__main__":
    main()
