"""Phase 3.75 task 3: allocate_top_k_by_weights and weighted query retrieval."""

import numpy as np
import pytest

from app.services.query_intent_service import allocate_top_k_by_weights, DOCUMENT_TYPE_VALUES
from app.services.query_service import QueryService


def test_allocate_sums_to_top_k_uniform():
    w = {t: 0.2 for t in DOCUMENT_TYPE_VALUES}
    for k in range(1, 21):
        a = allocate_top_k_by_weights(w, k)
        assert sum(a.values()) == k


def test_allocate_heavy_type_gets_more():
    w = {
        "statement of work": 0.7,
        "solution description document": 0.3,
        "proposal document": 0.0,
        "risk register": 0.0,
        "issue log": 0.0,
    }
    a = allocate_top_k_by_weights(w, 10)
    assert sum(a.values()) == 10
    assert a["statement of work"] >= a["solution description document"]


def test_allocate_zero_total_k():
    assert allocate_top_k_by_weights({"statement of work": 1.0}, 0) == {}


class _FakeEmb:
    def generate_embedding(self, text: str):
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


class _NoPast:
    async def find_similar_projects(self, **kwargs):
        return []


@pytest.mark.asyncio
async def test_weighted_retrieval_issues_per_type_searches(monkeypatch):
    search_calls = []

    class VS:
        async def search(self, **kwargs):
            search_calls.append(
                {
                    "top_k": kwargs["top_k"],
                    "document_type": kwargs.get("document_type"),
                    "project_name": kwargs.get("project_name"),
                }
            )
            dt = kwargs.get("document_type")
            k = kwargs["top_k"]
            pn = kwargs.get("project_name") or "p"
            label = dt.value if dt is not None else "any"
            out = []
            for i in range(k):
                out.append(
                    {
                        "text": f"{label}-{i}",
                        "metadata": {
                            "chunk_id": f"{pn}-{label}-{i}",
                            "document_type": label,
                            "project_name": pn,
                        },
                        "distance": 0.05 + 0.01 * i,
                    }
                )
            return out

    async def fake_get_vector_store():
        return VS()

    monkeypatch.setattr("app.services.query_service.get_vector_store", fake_get_vector_store)

    weights = {t: 1.0 / len(DOCUMENT_TYPE_VALUES) for t in DOCUMENT_TYPE_VALUES}
    qs = QueryService(embedding_service=_FakeEmb(), project_similarity_service=_NoPast())
    await qs.retrieve_with_past_projects(
        query="q",
        project_name="current",
        top_k=10,
        current_project_weight=1.0,
        past_projects_weight=0.0,
        document_weights=weights,
    )

    typed = [c for c in search_calls if c["document_type"] is not None]
    assert len(typed) >= 5
    assert sum(c["top_k"] for c in typed) == 10
    assert {c["document_type"].value for c in search_calls if c["document_type"]} == set(
        DOCUMENT_TYPE_VALUES
    )


@pytest.mark.asyncio
async def test_unweighted_single_search_current(monkeypatch):
    calls = []

    class VS:
        async def search(self, **kwargs):
            calls.append(kwargs)
            k = kwargs["top_k"]
            return [
                {
                    "text": f"x-{i}",
                    "metadata": {"chunk_id": f"id-{i}"},
                    "distance": 0.1,
                }
                for i in range(k)
            ]

    async def fake_get_vector_store():
        return VS()

    monkeypatch.setattr("app.services.query_service.get_vector_store", fake_get_vector_store)

    qs = QueryService(embedding_service=_FakeEmb(), project_similarity_service=_NoPast())
    await qs.retrieve_with_past_projects(
        query="q",
        project_name="cur",
        top_k=5,
        document_weights=None,
    )
    typed = [c for c in calls if c.get("document_type") is not None]
    assert len(typed) == 0
