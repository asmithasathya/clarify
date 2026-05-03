"""Validate frozen dataset files and print summary stats."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.report_data import load_examples_jsonl, sha256_file, validate_examples
from src.utils.io import read_json


def _summarize(counter_map: dict[str, int], limit: int = 5) -> str:
    items = sorted(counter_map.items(), key=lambda item: (-item[1], item[0]))
    head = ", ".join(f"{key}:{value}" for key, value in items[:limit])
    if len(items) <= limit:
        return head
    return f"{head}, +{len(items) - limit} more"


def _validate_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = read_json(manifest_path)
    dataset_name = manifest["dataset_name"]
    payload: dict[str, object] = {"dataset_name": dataset_name, "splits": {}}
    for split_name, split_info in manifest["splits"].items():
        path = Path(split_info["path"])
        examples = load_examples_jsonl(path)
        stats = validate_examples(
            examples,
            expected_dataset=dataset_name,
            require_split=(split_name != "snapshot" and split_name != "full"),
        )
        payload["splits"][split_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "n_examples": stats["n_examples"],
            "by_split": stats["by_split"],
            "by_ambiguity_type": stats["by_ambiguity_type"],
            "by_domain": stats["by_domain"],
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate report datasets from their manifests.")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to a dataset manifest JSON file. Repeat the flag to validate multiple datasets.",
    )
    args = parser.parse_args()

    for manifest_str in args.manifest:
        payload = _validate_manifest(Path(manifest_str))
        print(payload["dataset_name"])
        for split_name, split_payload in payload["splits"].items():
            print(
                f"  {split_name}: {split_payload['n_examples']} examples "
                f"sha256={split_payload['sha256']}"
            )
            print(f"    ambiguity={_summarize(split_payload['by_ambiguity_type'])}")
            print(f"    domain={_summarize(split_payload['by_domain'])}")


if __name__ == "__main__":
    main()
