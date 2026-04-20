"""Print a top-level manifest value for shell command substitution."""

from __future__ import annotations

import argparse

from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a top-level value from a JSON manifest.")
    parser.add_argument("--path", required=True, help="Path to the JSON manifest.")
    parser.add_argument("--key", required=True, help="Top-level key to print.")
    args = parser.parse_args()

    payload = read_json(args.path)
    if args.key not in payload:
        raise SystemExit(f"Key '{args.key}' not found in {args.path}")
    value = payload[args.key]
    if value is None:
        raise SystemExit(f"Key '{args.key}' is null in {args.path}")
    print(value)


if __name__ == "__main__":
    main()
