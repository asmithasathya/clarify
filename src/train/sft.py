"""Low-level LoRA SFT helpers built on the Tinker SDK."""

from __future__ import annotations

import random
from typing import Any, Iterable

from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


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
    if isinstance(prompt_tokens, (list, tuple)) and prompt_tokens and isinstance(prompt_tokens[0], (list, tuple)):
        prompt_tokens = prompt_tokens[0]
    if not isinstance(prompt_tokens, (list, tuple)):
        raise TypeError("Tokenizer.apply_chat_template() did not return token ids in a supported shape.")
    return [int(token_id) for token_id in prompt_tokens]


def build_supervised_datum(record: dict[str, Any], *, tokenizer: Any, types: Any) -> Any:
    prompt_messages = list(record["messages"])
    assistant_target = str(record["assistant_target"])

    prompt_ids = _normalize_prompt_tokens(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = _normalize_prompt_tokens(
        tokenizer.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": assistant_target}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    prefix_len = len(prompt_ids)
    while prefix_len > 0 and full_ids[:prefix_len] != prompt_ids[:prefix_len]:
        prefix_len -= 1
    if prefix_len == 0:
        raise ValueError("Could not align prompt tokens with full supervised example.")

    weights = [0.0] * prefix_len + [1.0] * (len(full_ids) - prefix_len)
    return types.Datum(
        model_input=types.ModelInput.from_ints(full_ids),
        loss_fn_inputs={
            "target_tokens": full_ids,
            "weights": weights,
        },
    )


def build_supervised_dataset(records: Iterable[dict[str, Any]], *, tokenizer: Any, types: Any) -> list[Any]:
    return [build_supervised_datum(record, tokenizer=tokenizer, types=types) for record in records]


def batchify(rows: list[Any], batch_size: int) -> list[list[Any]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def average_metric(results: list[dict[str, Any]], key: str) -> float:
    values = [float(result.get(key, 0.0)) for result in results if key in result]
    if not values:
        return 0.0
    return sum(values) / len(values)


def train_lora_sft(
    *,
    service_client: Any,
    base_model: str,
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    rank: int,
    seed: int,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
    eval_interval: int,
    checkpoint_name: str,
) -> dict[str, Any]:
    import tinker

    sampling_client = service_client.create_sampling_client(base_model=base_model)
    tokenizer = sampling_client.get_tokenizer()
    train_client = service_client.create_lora_training_client(base_model=base_model, rank=rank, seed=seed)

    train_data = build_supervised_dataset(train_records, tokenizer=tokenizer, types=tinker.types)
    dev_data = build_supervised_dataset(dev_records, tokenizer=tokenizer, types=tinker.types) if dev_records else []

    rng = random.Random(seed)
    optimizer = tinker.types.AdamParams(learning_rate=learning_rate)
    history: list[dict[str, Any]] = []

    if not train_data:
        raise ValueError("No training data provided for SFT.")

    steps = max(1, int(max_steps))
    index = 0
    for step in range(1, steps + 1):
        if index == 0:
            rng.shuffle(train_data)
        batch = train_data[index : index + batch_size]
        if not batch:
            index = 0
            rng.shuffle(train_data)
            batch = train_data[:batch_size]
        index = (index + batch_size) % len(train_data)

        fb_future = train_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = train_client.optim_step(optimizer)
        fb_result = fb_future.result()
        optim_result = optim_future.result()
        row = {
            "step": step,
            "train_metrics": dict(fb_result.metrics),
            "optim_metrics": dict(optim_result.metrics or {}),
        }
        if dev_data and step % max(1, eval_interval) == 0:
            dev_batches = batchify(dev_data, batch_size)
            eval_metrics: list[dict[str, Any]] = []
            for dev_batch in dev_batches:
                eval_result = train_client.forward(dev_batch, loss_fn="cross_entropy").result()
                eval_metrics.append(dict(eval_result.metrics))
            row["dev_metrics"] = {
                "loss": average_metric(eval_metrics, "loss"),
                "n_batches": len(eval_metrics),
            }
        history.append(row)
        LOGGER.info("SFT step %s/%s train_metrics=%s", step, steps, row["train_metrics"])

    checkpoint = train_client.save_state(checkpoint_name).result()
    return {
        "checkpoint_path": checkpoint.path,
        "history": history,
        "n_train_records": len(train_records),
        "n_dev_records": len(dev_records),
        "rank": rank,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_steps": steps,
    }
