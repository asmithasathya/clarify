"""LLM backends with structured-output helpers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, Type, TypeVar

from pydantic import BaseModel

from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


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


def parse_structured_output(raw_text: str, schema_type: Type[SchemaT]) -> SchemaT:
    try:
        return schema_type.model_validate_json(raw_text)
    except Exception:
        snippet = _extract_json_snippet(raw_text)
        return schema_type.model_validate(json.loads(snippet))


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

        for attempt in range(retries + 1):
            latest_output = self.generate(
                working_messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
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
                working_messages = list(messages) + [
                    {"role": "assistant", "content": latest_output},
                    {
                        "role": "user",
                        "content": (
                            "Your last response was not valid JSON. "
                            "Return only corrected JSON that matches the required schema exactly."
                        ),
                    },
                ]

        raise RuntimeError("Unreachable structured generation state.")


class TransformersGenerator(BaseGenerator):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        model_cfg = config.get("model", {}).get("generator", {})
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Transformers backend requested but transformers is not installed."
            ) from exc

        self.model_name = model_cfg["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=model_cfg.get("transformers_device_map", "auto"),
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )

        adapter_path = model_cfg.get("adapter_path")
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "A PEFT adapter path was provided but peft is not installed."
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def _prompt_from_messages(self, messages: Sequence[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        import torch

        generation_cfg = self.config.get("generation", {})
        prompt = self._prompt_from_messages(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **encoded,
            do_sample=(temperature or generation_cfg.get("temperature", 0.0)) > 0,
            temperature=temperature if temperature is not None else generation_cfg.get("temperature", 0.0),
            top_p=generation_cfg.get("top_p", 0.95),
            max_new_tokens=max_new_tokens or generation_cfg.get("max_new_tokens", 512),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class VLLMGenerator(BaseGenerator):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        model_cfg = config.get("model", {}).get("generator", {})
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vLLM backend requested but vllm is not installed.") from exc

        if model_cfg.get("adapter_path"):
            raise NotImplementedError(
                "LoRA/PEFT adapters are not wired into the vLLM path in v1. "
                "Use the transformers backend if you want to test adapters."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["model_name"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        self.llm = LLM(
            model=model_cfg["model_name"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.9),
            dtype=config.get("runtime", {}).get("dtype", "auto"),
            max_model_len=model_cfg.get("max_model_len", 4096),
        )

    def _prompt_from_messages(self, messages: Sequence[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        from vllm import SamplingParams

        generation_cfg = self.config.get("generation", {})
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else generation_cfg.get("temperature", 0.0),
            top_p=generation_cfg.get("top_p", 0.95),
            max_tokens=max_new_tokens or generation_cfg.get("max_new_tokens", 512),
        )
        prompt = self._prompt_from_messages(messages)
        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()


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
    backend = config.get("model", {}).get("generator", {}).get("backend", "vllm").lower()
    if backend == "vllm":
        return VLLMGenerator(config)
    if backend == "transformers":
        return TransformersGenerator(config)
    if backend == "mock":
        return MockGenerator(config=config)
    raise ValueError(f"Unsupported generator backend: {backend}")
