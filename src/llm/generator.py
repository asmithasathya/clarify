"""LLM backends with structured-output helpers."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, Type, TypeVar

from pydantic import BaseModel

from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _load_env_file(path: str = ".env") -> None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except FileNotFoundError:
        return


def _looks_like_jwt(value: str | None) -> bool:
    if not value:
        return False
    parts = value.split(".")
    if len(parts) != 3:
        return False
    return all(part and re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts)


def _tinker_auth_guidance(api_key_env_var: str, api_key_value: str | None) -> str:
    if _looks_like_jwt(api_key_value):
        return (
            f"The value in {api_key_env_var} looks like a JWT/session token. "
            "Use a long-lived Tinker API key instead of a browser or temporary JWT."
        )
    return (
        f"Tinker rejected the credential in {api_key_env_var}. "
        "Make sure it is a valid Tinker API key and rerun preflight before starting a long job."
    )


def _extract_json_snippet(text: str) -> str:
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    raise ValueError("No JSON object or array found in model output.")


def _quote_unquoted_object_keys(text: str) -> str:
    key_pattern = re.compile(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)')
    previous = None
    updated = text
    while updated != previous:
        previous = updated
        updated = key_pattern.sub(r'\1"\2"\3', updated)
    return updated


def _try_lenient_json_loads(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        raise ValueError("No JSON-like content found in model output.")

    python_like_candidates = [candidate, _quote_unquoted_object_keys(candidate)]
    for python_like in python_like_candidates:
        try:
            return ast.literal_eval(python_like)
        except Exception:
            pass

        normalized = re.sub(r"\bnull\b", "None", python_like, flags=re.IGNORECASE)
        normalized = re.sub(r"\btrue\b", "True", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
        try:
            return ast.literal_eval(normalized)
        except Exception:
            continue

    raise ValueError("Could not parse JSON-like model output.")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _looks_like_schema_echo(payload: Any) -> bool:
    return isinstance(payload, dict) and "properties" in payload and payload.get("type") == "object"


def _unwrap_properties_payload(payload: Any, schema_type: Type[SchemaT]) -> Any:
    if not isinstance(payload, dict):
        return payload
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        return payload

    extracted: dict[str, Any] = {}
    for field_name in schema_type.model_fields:
        if field_name not in properties:
            continue
        raw_value = properties[field_name]
        if isinstance(raw_value, dict):
            for key in ("value", "default", "example", "const"):
                if key in raw_value:
                    extracted[field_name] = raw_value[key]
                    break
            else:
                enum_values = raw_value.get("enum")
                if isinstance(enum_values, list) and enum_values:
                    extracted[field_name] = enum_values[0]
        else:
            extracted[field_name] = raw_value
    return extracted or payload


def _extract_string_field(raw_text: str, field_name: str) -> str | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"', raw_text)
    if not match:
        return None
    cursor = match.end()
    chars: list[str] = []
    escaped = False
    while cursor < len(raw_text):
        char = raw_text[cursor]
        if escaped:
            chars.append(char)
            escaped = False
        elif char == "\\":
            chars.append(char)
            escaped = True
        elif char == '"':
            candidate = "".join(chars)
            try:
                return json.loads(f'"{candidate}"')
            except Exception:
                return candidate
        else:
            chars.append(char)
        cursor += 1
    candidate = "".join(chars).strip()
    if candidate:
        for boundary in ('",\n', '",\r\n', '"\n', '", "'):
            if boundary in candidate:
                candidate = candidate.split(boundary, 1)[0]
        try:
            return json.loads(f'"{candidate}"')
        except Exception:
            return candidate
    return None


def _extract_optional_string_field(raw_text: str, field_name: str) -> str | None:
    null_match = re.search(rf'"{re.escape(field_name)}"\s*:\s*null', raw_text)
    if null_match:
        return None
    return _extract_string_field(raw_text, field_name)


def _extract_float_field(raw_text: str, field_name: str, default: float = 0.5) -> float:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)', raw_text)
    if not match:
        return default
    return _clamp_unit_interval(match.group(1), default=default)


def _extract_bool_field(raw_text: str, field_name: str, default: bool = False) -> bool:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*(true|false)', raw_text, flags=re.IGNORECASE)
    if not match:
        return default
    return match.group(1).lower() == "true"


def _extract_all_string_fields(raw_text: str, field_name: str) -> list[str]:
    pattern = re.compile(
        rf'(?:["\']?{re.escape(field_name)}["\']?)\s*:\s*(["\'])(.*?)\1',
        flags=re.DOTALL,
    )
    values: list[str] = []
    for match in pattern.finditer(raw_text):
        value = match.group(2).strip()
        if not value:
            continue
        value = value.replace("\\'", "'").replace('\\"', '"')
        values.append(value)
    return values


def _extract_all_float_fields(raw_text: str, field_name: str) -> list[float]:
    pattern = re.compile(rf'(?:["\']?{re.escape(field_name)}["\']?)\s*:\s*(-?\d+(?:\.\d+)?)')
    return [_clamp_unit_interval(match.group(1), default=0.5) for match in pattern.finditer(raw_text)]


def _recover_partial_payload(raw_text: str, schema_type: Type[SchemaT]) -> Any:
    schema_name = schema_type.__name__
    stripped = raw_text.strip()

    if schema_name == "AmbiguityAssessmentSchema":
        is_ambiguous = _extract_bool_field(stripped, "is_ambiguous", default=False)
        ambiguity_type = _extract_string_field(stripped, "ambiguity_type") or (
            "underspecified" if is_ambiguous else "none"
        )
        rationale = _extract_string_field(stripped, "rationale") or "Recovered from malformed JSON output."
        missing_variable = _extract_string_field(stripped, "variable")
        why_missing = _extract_string_field(stripped, "why_missing") or rationale
        missing_variables = []
        if is_ambiguous and missing_variable:
            missing_variables.append(
                {
                    "variable": missing_variable,
                    "why_missing": why_missing,
                    "importance": _extract_float_field(stripped, "importance", default=0.8),
                }
            )
        return {
            "is_ambiguous": is_ambiguous,
            "ambiguity_type": ambiguity_type,
            "missing_variables": missing_variables,
            "confidence": _extract_float_field(stripped, "confidence", default=0.5),
            "rationale": rationale,
        }

    if schema_name == "IntentModelSchema":
        descriptions = (
            _extract_all_string_fields(stripped, "description")
            or _extract_all_string_fields(stripped, "interpretation")
            or _extract_all_string_fields(stripped, "answer")
        )
        contexts = (
            _extract_all_string_fields(stripped, "assumed_context")
            or _extract_all_string_fields(stripped, "context")
            or _extract_all_string_fields(stripped, "rationale")
        )
        plausibilities = (
            _extract_all_float_fields(stripped, "plausibility")
            or _extract_all_float_fields(stripped, "probability")
        )

        if not descriptions:
            for line in stripped.splitlines():
                candidate = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip(" ,")
                if candidate and not candidate.startswith(("{", "[", '"', "'")):
                    descriptions.append(candidate)

        if descriptions:
            interpretations: list[dict[str, Any]] = []
            default_plausibility = 1.0 / max(1, min(len(descriptions), 5))
            for idx, description in enumerate(descriptions[:5]):
                interpretations.append(
                    {
                        "description": description,
                        "assumed_context": contexts[idx] if idx < len(contexts) else description,
                        "plausibility": plausibilities[idx] if idx < len(plausibilities) else default_plausibility,
                    }
                )
            return {
                "interpretations": interpretations,
                "most_likely_index": 0,
                "entropy_estimate": _entropy_label_from_count(len(interpretations)),
                "gap_description": _extract_string_field(stripped, "gap_description")
                or "Recovered from malformed JSON output.",
            }

    if schema_name in {"AnswerSchema", "HedgedAnswerSchema"}:
        answer = _extract_string_field(stripped, "answer")
        if not answer and stripped and not stripped.startswith(("{", "[")) and not stripped.startswith('"properties"'):
            answer = stripped
        if answer:
            payload: dict[str, Any] = {
                "answer": answer,
                "confidence": _extract_float_field(stripped, "confidence", default=0.5),
            }
            if schema_name == "HedgedAnswerSchema":
                payload["hedge_reason"] = _extract_string_field(stripped, "hedge_reason") or "Recovered from malformed JSON output."
            else:
                payload["assumed_interpretation"] = _extract_optional_string_field(stripped, "assumed_interpretation")
                payload["caveats"] = _extract_optional_string_field(stripped, "caveats")
            return payload

    if schema_name == "ClarificationQuestionSchema":
        question = _extract_string_field(stripped, "question")
        if not question and stripped and not stripped.startswith(("{", "[")):
            question = stripped
        if question:
            return {
                "question": question,
                "target_variable": _extract_string_field(stripped, "target_variable") or "general context",
                "why_this_helps": _extract_string_field(stripped, "why_this_helps") or "Recovered from malformed JSON output.",
            }

    if schema_name == "UserReplySchema":
        user_reply = _extract_string_field(stripped, "user_reply")
        if not user_reply and stripped and not stripped.startswith(("{", "[")):
            user_reply = stripped
        if user_reply:
            return {
                "user_reply": user_reply,
                "grounded_in_hidden_context": _extract_bool_field(stripped, "grounded_in_hidden_context", default=True),
            }

    if schema_name == "StrategyDecisionSchema":
        strategy = _extract_string_field(stripped, "strategy")
        if strategy:
            return {
                "strategy": strategy,
                "rationale": _extract_string_field(stripped, "rationale") or "Recovered from malformed JSON output.",
                "confidence": _extract_float_field(stripped, "confidence", default=0.5),
            }

    if schema_name == "NarrowedAnswerSchema":
        answer = _extract_string_field(stripped, "answer")
        if not answer and stripped and not stripped.startswith(("{", "[")):
            answer = stripped
        if answer:
            return {
                "stated_assumption": _extract_string_field(stripped, "stated_assumption") or "Recovered assumption",
                "answer": answer,
                "confidence": _extract_float_field(stripped, "confidence", default=0.5),
                "caveats": _extract_optional_string_field(stripped, "caveats")
                or "Recovered from malformed JSON output.",
            }

    if schema_name == "AnswerEvaluationSchema":
        rationale = _extract_string_field(stripped, "rationale")
        if rationale:
            return {
                "is_correct": _extract_bool_field(stripped, "is_correct", default=False),
                "score": _extract_float_field(stripped, "score", default=0.0),
                "rationale": rationale,
            }

    if schema_name == "ClarificationEvaluationSchema":
        rationale = _extract_string_field(stripped, "rationale")
        if rationale:
            return {
                "is_targeted": _extract_bool_field(stripped, "is_targeted", default=False),
                "score": _extract_float_field(stripped, "score", default=0.0),
                "missing_variable": _extract_string_field(stripped, "missing_variable") or "unknown",
                "rationale": rationale,
            }

    if schema_name == "AlternativesEvaluationSchema":
        rationale = _extract_string_field(stripped, "rationale")
        if rationale:
            return {
                "is_useful": _extract_bool_field(stripped, "is_useful", default=False),
                "score": _extract_float_field(stripped, "score", default=0.0),
                "rationale": rationale,
            }

    raise ValueError("Could not recover malformed structured output.")


def _is_truncation_error(raw_text: str, exc: Exception) -> bool:
    message = str(exc)
    if any(token in message for token in ("EOF while parsing", "Unterminated string", "json_invalid")):
        return True
    return raw_text.strip().startswith(("{", "[")) and not raw_text.strip().endswith(("}", "]"))


def _repair_instruction(schema_type: Type[SchemaT], raw_text: str, exc: Exception) -> str:
    details = "Your last response was not valid JSON."
    if _is_truncation_error(raw_text, exc):
        details = "Your last response was truncated before the JSON finished."
    elif '"properties"' in raw_text and '"type"' in raw_text:
        details = "Your last response returned a JSON schema instead of a JSON object with actual values."
    return (
        f"{details} "
        f"Return only one compact JSON object that matches the {schema_type.__name__} schema exactly. "
        "Do not return the schema itself. Do not include markdown, explanations, or code fences."
    )


def _next_max_new_tokens(
    current_max_new_tokens: int | None,
    config: dict[str, Any],
    raw_text: str,
    exc: Exception,
) -> int | None:
    if not _is_truncation_error(raw_text, exc):
        return current_max_new_tokens
    base = current_max_new_tokens or config.get("generation", {}).get("max_new_tokens", 512)
    return min(int(base * 1.5), 2048)


def _clamp_unit_interval(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def _entropy_label_from_count(count: int) -> str:
    if count <= 1:
        return "low"
    if count == 2:
        return "medium"
    return "high"


def _coerce_payload_for_schema(payload: Any, schema_type: Type[SchemaT]) -> Any:
    payload = _unwrap_properties_payload(payload, schema_type)
    schema_name = schema_type.__name__

    if schema_name == "IntentModelSchema":
        if isinstance(payload, dict) and "interpretations" in payload:
            return payload

        raw_interpretations: Any = payload
        if isinstance(payload, dict):
            for key in ("items", "candidates", "options", "results"):
                if isinstance(payload.get(key), list):
                    raw_interpretations = payload[key]
                    break

        if isinstance(raw_interpretations, list):
            interpretations: list[dict[str, Any]] = []
            for item in raw_interpretations[:5]:
                if isinstance(item, dict):
                    description = str(
                        item.get("description")
                        or item.get("interpretation")
                        or item.get("answer")
                        or ""
                    ).strip()
                    if not description:
                        continue
                    interpretations.append(
                        {
                            "description": description,
                            "assumed_context": str(
                                item.get("assumed_context")
                                or item.get("context")
                                or item.get("rationale")
                                or description
                            ).strip(),
                            "plausibility": _clamp_unit_interval(
                                item.get("plausibility", item.get("probability", 0.5))
                            ),
                        }
                    )
                elif isinstance(item, str) and item.strip():
                    interpretations.append(
                        {
                            "description": item.strip(),
                            "assumed_context": item.strip(),
                            "plausibility": 0.5,
                        }
                    )

            if interpretations:
                return {
                    "interpretations": interpretations,
                    "most_likely_index": 0,
                    "entropy_estimate": _entropy_label_from_count(len(interpretations)),
                    "gap_description": "Recovered from list-style intent model output.",
                }

    if schema_name == "StrategyDecisionSchema":
        if isinstance(payload, dict) and "strategy" in payload:
            return {
                "strategy": payload["strategy"],
                "rationale": payload.get("rationale") or "Recovered from partial strategy output.",
                "confidence": _clamp_unit_interval(payload.get("confidence", 0.5)),
            }
        if isinstance(payload, str):
            strategy = payload.strip().strip('"').strip("'")
            return {
                "strategy": strategy,
                "rationale": "Recovered from string-only strategy output.",
                "confidence": 0.5,
            }

    if schema_name == "ClarificationQuestionSchema":
        if isinstance(payload, dict):
            question = str(
                payload.get("question")
                or payload.get("clarification_question")
                or payload.get("prompt")
                or ""
            ).strip()
            if question:
                target_variable = str(
                    payload.get("target_variable")
                    or payload.get("missing_variable")
                    or payload.get("focus")
                    or "general context"
                ).strip()
                why_this_helps = str(payload.get("why_this_helps") or "").strip()
                if not why_this_helps:
                    why_this_helps = f"This helps resolve the missing {target_variable} needed to answer accurately."
                return {
                    "question": question,
                    "target_variable": target_variable,
                    "why_this_helps": why_this_helps,
                }
        if isinstance(payload, str):
            question = payload.strip()
            return {
                "question": question,
                "target_variable": "general context",
                "why_this_helps": "Recovered from string-only clarification output.",
            }

    if schema_name in {"AnswerSchema", "HedgedAnswerSchema"}:
        if isinstance(payload, dict) and "answer" in payload:
            answer = str(payload["answer"]).strip()
            coerced = {
                "answer": answer,
                "confidence": _clamp_unit_interval(payload.get("confidence", 0.5)),
            }
            if schema_name == "HedgedAnswerSchema":
                coerced["hedge_reason"] = payload.get("hedge_reason") or "Recovered from partial hedged answer output."
            else:
                coerced["assumed_interpretation"] = payload.get("assumed_interpretation")
                coerced["caveats"] = payload.get("caveats")
            return coerced
        if isinstance(payload, str):
            answer = payload.strip()
            coerced = {"answer": answer, "confidence": 0.5}
            if schema_name == "HedgedAnswerSchema":
                coerced["hedge_reason"] = "Recovered from string-only hedged answer output."
            else:
                coerced["assumed_interpretation"] = None
                coerced["caveats"] = None
            return coerced

    if schema_name == "NarrowedAnswerSchema":
        if isinstance(payload, dict) and "answer" in payload:
            return {
                "stated_assumption": payload.get("stated_assumption") or "Recovered assumption",
                "answer": str(payload["answer"]).strip(),
                "confidence": _clamp_unit_interval(payload.get("confidence", 0.5)),
                "caveats": payload.get("caveats") or "Recovered from partial narrowed answer output.",
            }
        if isinstance(payload, str):
            answer = payload.strip()
            return {
                "stated_assumption": "Recovered assumption",
                "answer": answer,
                "confidence": 0.5,
                "caveats": "Recovered from string-only narrowed answer output.",
            }

    return payload


def parse_structured_output(raw_text: str, schema_type: Type[SchemaT]) -> SchemaT:
    raw_text = _strip_code_fences(raw_text)
    try:
        return schema_type.model_validate_json(raw_text)
    except Exception:
        payload: Any | None = None
        last_error: Exception | None = None
        candidates = [raw_text]
        try:
            snippet = _extract_json_snippet(raw_text)
            if snippet != raw_text:
                candidates.append(snippet)
        except Exception as exc:
            last_error = exc

        for candidate in candidates:
            for loader in (json.loads, _try_lenient_json_loads):
                try:
                    payload = loader(candidate)
                    break
                except Exception as exc:
                    last_error = exc
            if payload is not None:
                break

        if payload is None:
            for candidate in candidates:
                try:
                    payload = _recover_partial_payload(candidate, schema_type)
                    break
                except Exception as exc:
                    last_error = exc
            if payload is None:
                raise last_error if last_error is not None else ValueError(
                    "Could not recover malformed structured output."
                )
        payload = _coerce_payload_for_schema(payload, schema_type)
        if _looks_like_schema_echo(payload):
            raise ValueError("Model returned a JSON schema instead of a schema instance.")
        return schema_type.model_validate(payload)


class BaseGenerator(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        schema_type: Type[SchemaT],
        *,
        max_retries: int | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> SchemaT:
        retries = (
            self.config.get("generation", {}).get("json_max_retries", 2)
            if max_retries is None
            else max_retries
        )
        latest_output = ""
        working_messages = list(messages)
        current_max_new_tokens = max_new_tokens

        for attempt in range(retries + 1):
            latest_output = self.generate(
                working_messages,
                temperature=temperature,
                max_new_tokens=current_max_new_tokens,
            )
            try:
                return parse_structured_output(latest_output, schema_type)
            except Exception as exc:
                LOGGER.warning(
                    "Structured parse failed on attempt %s/%s: %s",
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                if attempt == retries:
                    raise
                current_max_new_tokens = _next_max_new_tokens(
                    current_max_new_tokens,
                    self.config,
                    latest_output,
                    exc,
                )
                working_messages = list(messages) + [
                    {"role": "assistant", "content": latest_output},
                    {
                        "role": "user",
                        "content": _repair_instruction(schema_type, latest_output, exc),
                    },
                ]

        raise RuntimeError("Unreachable structured generation state.")

    def generate_n(
        self,
        messages: Sequence[dict[str, str]],
        *,
        n: int,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if n <= 0:
            return []
        if n == 1:
            return [self.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)]

        max_workers = min(
            n,
            int(self.config.get("runtime", {}).get("parallelism", {}).get("max_concurrency", 4)),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.generate,
                    messages,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                for _ in range(n)
            ]
            return [future.result() for future in futures]

    def generate_structured_n(
        self,
        messages: Sequence[dict[str, str]],
        schema_type: Type[SchemaT],
        *,
        n: int,
        max_retries: int | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[SchemaT]:
        if n <= 0:
            return []
        if n == 1:
            return [
                self.generate_structured(
                    messages,
                    schema_type,
                    max_retries=max_retries,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            ]

        max_workers = min(
            n,
            int(self.config.get("runtime", {}).get("parallelism", {}).get("max_concurrency", 4)),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.generate_structured,
                    messages,
                    schema_type,
                    max_retries=max_retries,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                for _ in range(n)
            ]
            return [future.result() for future in futures]


class TinkerGenerator(BaseGenerator):
    """Remote inference via the native Tinker SDK."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        model_cfg = config.get("model", {}).get("generator", {})
        self.api_key_env_var = model_cfg.get("api_key_env_var", "TINKER_API_KEY")
        self.api_key = self._load_api_key()
        resilience_cfg = config.get("runtime", {}).get("resilience", {})
        self.request_max_attempts = max(1, int(resilience_cfg.get("tinker_request_max_attempts", 6)))
        self.auth_retry_attempts = max(0, int(resilience_cfg.get("tinker_auth_retry_attempts", 1)))
        self.retry_base_delay_seconds = max(0.0, float(resilience_cfg.get("tinker_retry_base_delay_seconds", 5.0)))
        self.retry_max_delay_seconds = max(
            self.retry_base_delay_seconds,
            float(resilience_cfg.get("tinker_retry_max_delay_seconds", 60.0)),
        )
        if resilience_cfg.get("disable_tinker_telemetry", True):
            os.environ.setdefault("TINKER_TELEMETRY", "0")
        if resilience_cfg.get("enable_tinker_subprocess_sampling", False):
            os.environ.setdefault("TINKER_SUBPROCESS_SAMPLING", "1")

        try:
            import tinker
        except ImportError as exc:
            raise ImportError(
                "Tinker backend requested but the 'tinker' package is not installed."
            ) from exc

        self._tinker = tinker
        self._types = tinker.types

        self.model_path = model_cfg.get("model_path")
        self.base_model = model_cfg.get("base_model")
        if not self.model_path and not self.base_model:
            raise ValueError("Tinker backend requires either model_path or base_model.")

        self._refresh_client()

    def _load_api_key(self) -> str:
        _load_env_file()
        api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            raise EnvironmentError(
                f"Missing Tinker credentials. Set the {self.api_key_env_var} environment variable."
            )
        return api_key

    def _build_service_client(self) -> Any:
        return self._tinker.ServiceClient(api_key=self.api_key)

    def _refresh_client(self) -> None:
        self.api_key = self._load_api_key()
        service_client = self._build_service_client()
        if self.model_path:
            self.sampling_client = service_client.create_sampling_client(model_path=self.model_path)
        else:
            self.sampling_client = service_client.create_sampling_client(base_model=self.base_model)
        self.tokenizer = self.sampling_client.get_tokenizer()
        self.stop_sequences = self._infer_stop_sequences()
        self._prompt_cache: dict[str, Any] = {}

    def _is_auth_error(self, exc: Exception) -> bool:
        auth_cls = getattr(self._tinker, "AuthenticationError", None)
        if auth_cls is not None and isinstance(exc, auth_cls):
            return True
        message = str(exc).lower()
        return "invalid jwt" in message or ("401" in message and "auth" in exc.__class__.__name__.lower())

    def _raise_auth_error(self, exc: Exception) -> None:
        guidance = _tinker_auth_guidance(self.api_key_env_var, self.api_key)
        raise EnvironmentError(f"Tinker authentication failed: {exc}. {guidance}") from exc

    def _is_transient_error(self, exc: Exception) -> bool:
        message = f"{exc.__class__.__name__}: {exc}".lower()
        markers = (
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection closed",
            "broken pipe",
            "network is unreachable",
            "temporary failure",
            "temporarily unavailable",
            "remote disconnected",
            "server disconnected",
            "service unavailable",
            "too many requests",
            "rate limit",
            "429",
            "502",
            "503",
            "504",
        )
        return any(marker in message for marker in markers)

    def _retry_delay_seconds(self, attempt_index: int) -> float:
        delay = self.retry_base_delay_seconds * (2 ** attempt_index)
        return min(delay, self.retry_max_delay_seconds)

    def _infer_stop_sequences(self) -> list[int] | None:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return None
        if isinstance(eos_token_id, (list, tuple)):
            return [int(token_id) for token_id in eos_token_id]
        return [int(eos_token_id)]

    @staticmethod
    def _normalize_prompt_tokens(prompt_tokens: Any) -> list[int]:
        if hasattr(prompt_tokens, "tolist"):
            prompt_tokens = prompt_tokens.tolist()

        if isinstance(prompt_tokens, dict):
            prompt_tokens = prompt_tokens.get("input_ids", prompt_tokens)

        if hasattr(prompt_tokens, "get") and not isinstance(prompt_tokens, dict):
            input_ids = prompt_tokens.get("input_ids")
            if input_ids is not None:
                prompt_tokens = input_ids

        if hasattr(prompt_tokens, "tolist"):
            prompt_tokens = prompt_tokens.tolist()

        if isinstance(prompt_tokens, (list, tuple)) and prompt_tokens:
            first = prompt_tokens[0]
            if isinstance(first, (list, tuple)):
                prompt_tokens = first

        if not isinstance(prompt_tokens, (list, tuple)):
            raise TypeError(
                "Tokenizer.apply_chat_template() did not return token ids in a supported shape."
            )

        return [int(token_id) for token_id in prompt_tokens]

    def _prompt_from_messages(self, messages: Sequence[dict[str, str]]) -> Any:
        cache_enabled = bool(self.config.get("intent_resampling", {}).get("cache_prompt_renders", True))
        cache_key = json.dumps(list(messages), sort_keys=True)
        if cache_enabled and cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "The tokenizer returned by Tinker does not expose apply_chat_template()."
            )

        prompt_tokens = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt = self._types.ModelInput.from_ints(
            self._normalize_prompt_tokens(prompt_tokens)
        )
        if cache_enabled:
            self._prompt_cache[cache_key] = prompt
        return prompt

    @staticmethod
    def _extract_generated_tokens(result: Any) -> list[int]:
        sequences = getattr(result, "sequences", None)
        if sequences:
            return list(sequences[0].tokens)

        samples = getattr(result, "samples", None)
        if samples:
            return list(samples[0].tokens)

        raise RuntimeError("Tinker sample() returned no sequences/samples to decode.")

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        generation_cfg = self.config.get("generation", {})
        sampling_kwargs: dict[str, Any] = {
            "temperature": (
                temperature if temperature is not None else generation_cfg.get("temperature", 0.0)
            ),
            "top_p": generation_cfg.get("top_p", 0.95),
            "max_tokens": max_new_tokens or generation_cfg.get("max_new_tokens", 512),
        }
        if self.stop_sequences:
            sampling_kwargs["stop"] = self.stop_sequences

        auth_attempts = 0
        for attempt in range(self.request_max_attempts):
            try:
                future = self.sampling_client.sample(
                    prompt=self._prompt_from_messages(messages),
                    num_samples=1,
                    sampling_params=self._types.SamplingParams(**sampling_kwargs),
                )
                result = future.result()
                generated_tokens = self._extract_generated_tokens(result)
                return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            except Exception as exc:
                is_auth = self._is_auth_error(exc)
                is_transient = self._is_transient_error(exc)
                if is_auth:
                    if auth_attempts >= self.auth_retry_attempts:
                        LOGGER.error(
                            "Tinker auth failure persisted after %s in-process retries; failing fast so the outer retry wrapper can restart a fresh process: %s",
                            self.auth_retry_attempts,
                            exc,
                        )
                        self._raise_auth_error(exc)
                    auth_attempts += 1
                    delay = self._retry_delay_seconds(auth_attempts - 1)
                    LOGGER.warning(
                        "Tinker auth failure during generation; rebuilding client and retrying in %.1fs (%s/%s): %s",
                        delay,
                        auth_attempts,
                        self.auth_retry_attempts,
                        exc,
                    )
                    try:
                        self._refresh_client()
                    except Exception as refresh_exc:
                        LOGGER.warning("Tinker client rebuild failed during auth retry (%s/%s): %s", auth_attempts, self.auth_retry_attempts, refresh_exc)
                    time.sleep(delay)
                    continue

                has_retry = attempt < self.request_max_attempts - 1
                if is_transient:
                    if not has_retry:
                        raise
                    delay = self._retry_delay_seconds(attempt)
                    LOGGER.warning(
                        "Tinker transient failure during generation; rebuilding client and retrying in %.1fs (%s/%s): %s",
                        delay,
                        attempt + 1,
                        self.request_max_attempts,
                        exc,
                    )
                    try:
                        self._refresh_client()
                    except Exception as refresh_exc:
                        LOGGER.warning("Tinker client rebuild failed during transient retry (%s/%s): %s", attempt + 1, self.request_max_attempts, refresh_exc)
                    time.sleep(delay)
                    continue
                raise

        raise RuntimeError("Unreachable Tinker generation state.")


class MockGenerator(BaseGenerator):
    def __init__(
        self,
        responses: list[str] | None = None,
        responder: Callable[[Sequence[dict[str, str]]], str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config or {"generation": {"json_max_retries": 0}})
        self.responses = responses or []
        self.responder = responder

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        if self.responder is not None:
            return self.responder(messages)
        if not self.responses:
            raise RuntimeError("MockGenerator has no remaining responses.")
        return self.responses.pop(0)


def build_generator(config: dict[str, Any]) -> BaseGenerator:
    backend = config.get("model", {}).get("generator", {}).get("backend", "tinker").lower()
    if backend == "tinker":
        return TinkerGenerator(config)
    if backend == "mock":
        return MockGenerator(config=config)
    raise ValueError(f"Unsupported generator backend: {backend}")
