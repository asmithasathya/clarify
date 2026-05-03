"""Download and cache InfoQuest dataset locally as JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.infoquest import load_infoquest
from src.utils.io import write_jsonl
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download InfoQuest and cache as JSONL.")
    parser.add_argument("--output", default="data/infoquest.jsonl", help="Output path.")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to download.")
    parser.add_argument("--single-setting", action="store_true", help="Use only setting1 per seed.")
    args = parser.parse_args()

    LOGGER.info("Loading InfoQuest from HuggingFace...")
    examples = load_infoquest(limit=args.limit, both_settings=not args.single_setting)
    LOGGER.info("Loaded %d examples.", len(examples))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(examples, out_path)
    LOGGER.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
