"""Environment checks for report runs."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from src.llm.generator import _load_env_file, _looks_like_jwt, _tinker_auth_guidance
from src.utils.config import load_config


def _status(ok: bool, label: str, detail: str) -> tuple[bool, str]:
    prefix = "OK" if ok else "FAIL"
    return ok, f"[{prefix}] {label}: {detail}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run environment preflight checks.")
    parser.add_argument("--config", default="configs/report_baselines.yaml", help="Config path to validate against.")
    parser.add_argument("--output-dir", default="outputs", help="Directory that should be writable.")
    parser.add_argument("--skip-tinker", action="store_true", help="Skip live Tinker capability checks.")
    args = parser.parse_args()

    config = load_config(args.config)
    _load_env_file()

    checks: list[tuple[bool, str]] = []
    checks.append(
        _status(
            sys.version_info >= (3, 11),
            "python",
            f"{sys.version.split()[0]} (requires 3.11+)",
        )
    )

    for module_name in ("datasets", "numpy", "pydantic", "yaml", "tinker", "typer", "tqdm"):
        try:
            importlib.import_module(module_name)
            checks.append(_status(True, f"import:{module_name}", "available"))
        except Exception as exc:
            checks.append(_status(False, f"import:{module_name}", str(exc)))

    api_key_env = config.get("model", {}).get("generator", {}).get("api_key_env_var", "TINKER_API_KEY")
    api_key_value = os.getenv(api_key_env)
    checks.append(
        _status(
            bool(api_key_value) and not _looks_like_jwt(api_key_value),
            "credentials",
            (
                f"{api_key_env} is set"
                if api_key_value and not _looks_like_jwt(api_key_value)
                else (
                    f"{api_key_env} is set but looks like a JWT/session token"
                    if api_key_value
                    else f"{api_key_env} is missing"
                )
            ),
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".preflight_write_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        checks.append(_status(True, "output-dir", f"writable: {output_dir}"))
    except Exception as exc:
        checks.append(_status(False, "output-dir", str(exc)))

    if not args.skip_tinker:
        try:
            import tinker

            service_client = tinker.ServiceClient(api_key=api_key_value)
            capabilities = service_client.get_server_capabilities()
            available = {
                getattr(model, "model_name", str(model))
                for model in getattr(capabilities, "supported_models", [])
            }
            requested = {
                model["base_model"]
                for model in config.get("report", {}).get("task_models", [])
            }
            judge_model = config.get("report", {}).get("judge_model", {}).get("base_model")
            if judge_model:
                requested.add(judge_model)
            missing = sorted(requested - available)
            detail = f"{len(available)} models visible"
            if missing:
                checks.append(_status(False, "tinker-models", f"missing requested models: {missing}"))
            else:
                checks.append(_status(True, "tinker-models", detail))
        except Exception as exc:
            guidance = _tinker_auth_guidance(api_key_env, api_key_value)
            detail = str(exc)
            if "Invalid JWT" in detail or "401" in detail:
                detail = f"{detail}. {guidance}"
            checks.append(_status(False, "tinker-models", detail))

    failures = 0
    for ok, line in checks:
        print(line)
        if not ok:
            failures += 1
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
