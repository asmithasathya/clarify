import numpy as np
import pytest

from src.data.schema import StatutePassage
from src.retrieval.faiss_index import create_index
from src.retrieval.retrieve import Retriever


pytest.importorskip("faiss")


class FakeEmbedder:
    def encode(self, texts, batch_size=None):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "california" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype="float32")


def test_faiss_retrieval_returns_passages():
    docs = [
        StatutePassage(doc_id="CA-1", state="California", citation="CA Civ. Code 1", text="California notice rule."),
        StatutePassage(doc_id="NY-1", state="New York", citation="NY RPL 1", text="New York notice rule."),
    ]
    index = create_index(dim=2, index_type="flatip", metric="ip")
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))

    retriever = Retriever(
        config={"project": {"year": 2021}, "retrieval": {"top_k": 1, "initial_fetch_k": 2}},
        embedder=FakeEmbedder(),
        documents=docs,
        index=index,
        reranker=None,
    )
    hits = retriever.retrieve("written notice", state="California", top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "CA-1"
    assert isinstance(hits[0].text, str)

