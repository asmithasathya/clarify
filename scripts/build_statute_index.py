"""Build a FAISS dense statute index for HousingQA."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import typer

from src.data.housingqa import iter_housingqa_statutes, load_housingqa_statutes
from src.retrieval.embedder import build_embedder
from src.retrieval.faiss_index import DOCS_FILENAME, create_index, save_index
from src.utils.config import load_config
from src.utils.io import append_jsonl, ensure_dir
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
app = typer.Typer(add_completion=False)


def _batched(items: Iterable, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@app.command()
def main(
    config: str = typer.Option("configs/default.yaml", help="Base config path."),
    limit: int | None = typer.Option(None, help="Optional corpus limit for smoke builds."),
) -> None:
    resolved = load_config(config)
    housing_cfg = resolved["data"]["housingqa"]
    retrieval_cfg = resolved["retrieval"]
    index_dir = ensure_dir(resolved["paths"]["index_dir"])
    docs_path = Path(index_dir) / DOCS_FILENAME
    if docs_path.exists():
        docs_path.unlink()

    embedder = build_embedder(resolved)
    train_size = retrieval_cfg.get("train_size", 20000)
    index_type = retrieval_cfg.get("index_type", "ivfflat")

    warmup_docs = load_housingqa_statutes(
        dataset_name=housing_cfg["statutes_dataset"],
        config_name=housing_cfg["statutes_config"],
        split=housing_cfg["statutes_split"],
        limit=min(train_size, limit) if limit else train_size,
    )
    if not warmup_docs:
        raise RuntimeError("No statute documents were loaded for index construction.")

    warmup_embeddings = embedder.encode(
        [doc.searchable_text for doc in warmup_docs],
        batch_size=retrieval_cfg.get("batch_size", 64),
    )
    dim = warmup_embeddings.shape[1]
    index = create_index(
        dim,
        index_type=index_type,
        metric=retrieval_cfg.get("metric", "ip"),
        nlist=retrieval_cfg.get("nlist", 2048),
    )
    if hasattr(index, "is_trained") and not index.is_trained:
        LOGGER.info("Training FAISS %s index on %s passages", index_type, len(warmup_docs))
        index.train(warmup_embeddings)

    count = 0
    for batch in _batched(
        iter_housingqa_statutes(
            dataset_name=housing_cfg["statutes_dataset"],
            config_name=housing_cfg["statutes_config"],
            split=housing_cfg["statutes_split"],
        ),
        retrieval_cfg.get("batch_size", 64),
    ):
        if limit is not None and count >= limit:
            break
        if limit is not None:
            batch = batch[: max(0, limit - count)]
        texts = [doc.searchable_text for doc in batch]
        embeddings = embedder.encode(texts, batch_size=retrieval_cfg.get("batch_size", 64))
        index.add(embeddings)
        for doc in batch:
            append_jsonl(doc, docs_path)
        count += len(batch)
        LOGGER.info("Indexed %s statutes", count)

    metadata = {
        "built_at": datetime.now().isoformat(),
        "statute_count": count,
        "embedding_model": retrieval_cfg["embedder_model"],
        "index_type": index_type,
        "dimension": dim,
    }
    save_index(index, index_dir, metadata)
    typer.echo(f"Built index with {count} statute passages at {index_dir}")


if __name__ == "__main__":
    app()

