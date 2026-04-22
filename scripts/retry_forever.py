"""Run a command until it succeeds, sleeping between failed attempts."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restart a command until it exits successfully.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=30.0,
        help="Seconds to sleep between failed attempts.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Optional maximum number of attempts. Zero means retry forever.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Prefix with -- to separate wrapper flags from the wrapped command.",
    )
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("No command provided. Use: retry_forever.py -- <command> ...")

    attempt = 0
    while True:
        attempt += 1
        printable = " ".join(command)
        print(f"[retry_forever] attempt {attempt}: {printable}", flush=True)
        completed = subprocess.run(command)
        if completed.returncode == 0:
            print(f"[retry_forever] command succeeded on attempt {attempt}", flush=True)
            return 0

        if args.max_attempts > 0 and attempt >= args.max_attempts:
            print(
                f"[retry_forever] command failed after {attempt} attempts with exit code {completed.returncode}",
                flush=True,
            )
            return completed.returncode

        print(
            f"[retry_forever] command failed with exit code {completed.returncode}; "
            f"sleeping {args.retry_delay:.1f}s before retry",
            flush=True,
        )
        time.sleep(args.retry_delay)


if __name__ == "__main__":
    raise SystemExit(main())
