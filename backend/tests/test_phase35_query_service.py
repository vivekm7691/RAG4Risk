"""Phase 3.5: query service past-project allocation and backfill."""

import numpy as np
import pytest

from app.services.query_service import QueryService


class _FakeEmb:
    def generate_embedding(self, text: str):
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


class _PastSim:
    async def find_similar_projects(self, **kwargs):
        return [
            {"project_name": "past_a", "customer": "C1", "similarity_score": 0.9, "metadata": None},
            {"project_name": "past_b", "customer": "C2", "similarity_score": 0.8, "metadata": None},
        ]


@pytest.mark.asyncio
async def test_past_retrieval_respects_n_past_budget(monkeypatch):
    """With top_k=10 and 70/30 split, at most 3 chunks should come from past projects total."""
    search_calls = []

    class VS:
        async def search(self, **kwargs):
            search_calls.append(kwargs)
            pn = kwargs.get("project_name")
            k = kwargs["top_k"]
            out = []
            for i in range(k):
                out.append(
                    {
                        "text": f"{pn}-{i}",
                        "metadata": {"chunk_id": f"{pn}-{i}", "project_name": pn},
                        "distance": 0.1 * (i + 1),
                    }
                )
            return out

    async def fake_get_vector_store():
        return VS()

    monkeypatch.setattr("app.services.query_service.get_vector_store", fake_get_vector_store)

    qs = QueryService(embedding_service=_FakeEmb(), project_similarity_service=_PastSim())
    result = await qs.retrieve_with_past_projects(
        query="q",
        project_name="current",
        top_k=10,
        current_project_weight=0.7,
        past_projects_weight=0.3,
    )
    chunks = result["chunks"]
    past_n = sum(1 for c in chunks if c.get("is_past_project"))
    assert past_n <= 3
    assert len(chunks) == 10


@pytest.mark.asyncio
async def test_backfill_from_current_when_short(monkeypatch):
    """If initial current search yields few chunks, backfill pulls more from current project."""
    class VS:
        def __init__(self):
            self._calls = 0

        async def search(self, **kwargs):
            pn = kwargs.get("project_name")
            k = kwargs["top_k"]
            if pn != "current":
                return []
            self._calls += 1
            if self._calls == 1:
                return [
                    {
                        "text": "only-one",
                        "metadata": {"chunk_id": "c-0", "project_name": "current"},
                        "distance": 0.05,
                    }
                ]
            return [
                {
                    "text": f"extra-{i}",
                    "metadata": {"chunk_id": f"c-{i}", "project_name": "current"},
                    "distance": 0.1 + i * 0.01,
                }
                for i in range(k)
            ]

    async def fake_get_vector_store():
        return VS()

    monkeypatch.setattr("app.services.query_service.get_vector_store", fake_get_vector_store)

    class NoPast(_PastSim):
        async def find_similar_projects(self, **kwargs):
            return []

    qs = QueryService(embedding_service=_FakeEmb(), project_similarity_service=NoPast())
    result = await qs.retrieve_with_past_projects(
        query="q",
        project_name="current",
        top_k=5,
        current_project_weight=0.7,
        past_projects_weight=0.3,
    )
    assert len(result["chunks"]) == 5
    assert all(not c.get("is_past_project") for c in result["chunks"])
