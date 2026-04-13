"""Phase 3.75 task 6: intent JSON, normalization, allocation, sparse-type retrieval."""

import json

import numpy as np
import pytest

from app.models.document import DocumentType
from app.services.query_intent_service import (
    DOCUMENT_TYPE_VALUES,
    QueryIntentService,
    allocate_top_k_by_weights,
    extract_json_object,
    mask_weights_to_project_types,
    normalize_weights_all_types,
    strip_reasoning_traces,
)
from app.services.query_service import QueryService


def test_normalize_weights_renormalizes_when_sum_not_one():
    raw = {t: 2.0 for t in DOCUMENT_TYPE_VALUES}
    out = normalize_weights_all_types(raw)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert all(out[t] == pytest.approx(0.2) for t in DOCUMENT_TYPE_VALUES)


def test_normalize_weights_all_zero_is_uniform():
    out = normalize_weights_all_types({t: 0.0 for t in DOCUMENT_TYPE_VALUES})
    assert abs(sum(out.values()) - 1.0) < 1e-9
    u = 1.0 / len(DOCUMENT_TYPE_VALUES)
    assert all(out[t] == pytest.approx(u) for t in DOCUMENT_TYPE_VALUES)


def test_normalize_coerces_string_numbers():
    out = normalize_weights_all_types(
        {
            "statement of work": "0.5",
            "solution description document": 0.5,
            "proposal document": 0,
            "risk register": 0,
            "issue log": 0,
        }
    )
    assert out["statement of work"] == pytest.approx(0.5)
    assert out["solution description document"] == pytest.approx(0.5)


def test_mask_weights_only_types_in_project():
    base = normalize_weights_all_types({t: 1.0 for t in DOCUMENT_TYPE_VALUES})
    masked = mask_weights_to_project_types(
        base,
        ["statement of work", "solution description document"],
    )
    assert set(masked.keys()) == {"statement of work", "solution description document"}
    assert abs(sum(masked.values()) - 1.0) < 1e-9


def test_process_llm_output_partial_weights_normalized():
    svc = QueryIntentService()
    payload = {
        "intent_summary": "x",
        "document_weights": {"statement of work": 3, "solution description document": 1},
    }
    r = svc.process_llm_output(json.dumps(payload), None)
    assert not r.used_fallback
    assert abs(sum(r.document_weights.values()) - 1.0) < 1e-6
    assert r.document_weights["statement of work"] > r.document_weights["proposal document"]


def test_process_llm_output_invalid_json_fallback():
    svc = QueryIntentService()
    r = svc.process_llm_output("not json {{{", None)
    assert r.used_fallback
    assert abs(sum(r.document_weights.values()) - 1.0) < 1e-6


def test_extract_json_from_markdown_fence():
    inner = '{"intent_summary":"a","document_weights":{"statement of work":1}}'
    text = f"Here:\n```json\n{inner}\n```"
    assert extract_json_object(text) == inner


def test_strip_reasoning_then_json():
    think_open = "<" + "think" + ">"
    think_close = "<" + "/" + "think" + ">"
    inner = '{"intent_summary":"z","document_weights":{}}'
    text = think_open + "reasoning" + think_close + inner
    assert extract_json_object(text) == inner


def test_sample_query_allocation_favors_sow_and_solution_over_proposal():
    """
    Plan sample: risks in solution vs gaps in SOW — mock LLM-style weights; allocation
    should assign more slots to SOW and solution than to proposal.
    """
    svc = QueryIntentService()
    payload = {
        "intent_summary": "Compare implied solution risks with contractual SOW scope.",
        "document_weights": {
            "statement of work": 0.45,
            "solution description document": 0.45,
            "proposal document": 0.05,
            "risk register": 0.03,
            "issue log": 0.02,
        },
    }
    r = svc.process_llm_output(json.dumps(payload), None)
    assert not r.used_fallback
    slots = allocate_top_k_by_weights(r.document_weights, 10)
    assert slots["statement of work"] > slots["proposal document"]
    assert slots["solution description document"] > slots["proposal document"]
    assert sum(slots.values()) == 10


@pytest.mark.asyncio
async def test_weighted_retrieval_empty_hits_for_one_type_still_reaches_top_k(monkeypatch):
    """
    Per-type search returns no rows for proposal; other types return hits; backfill fills to top_k.
    """
    search_calls = []

    class VS:
        async def search(self, **kwargs):
            search_calls.append(kwargs)
            dt = kwargs.get("document_type")
            k = kwargs["top_k"]
            pn = kwargs.get("project_name") or "proj"
            if dt == DocumentType.PROPOSAL:
                return []
            if dt is None:
                out = []
                for i in range(max(k, 20)):
                    out.append(
                        {
                            "text": f"bf-{i}",
                            "metadata": {
                                "chunk_id": f"bf-{pn}-{i}",
                                "document_type": "statement of work",
                                "project_name": pn,
                            },
                            "distance": 0.2 + i * 0.001,
                        }
                    )
                return out
            label = dt.value
            return [
                {
                    "text": f"{label}-{i}",
                    "metadata": {
                        "chunk_id": f"{pn}-{label}-{i}",
                        "document_type": label,
                        "project_name": pn,
                    },
                    "distance": 0.05 + 0.01 * i,
                }
                for i in range(k)
            ]

    async def fake_get_vector_store():
        return VS()

    monkeypatch.setattr("app.services.query_service.get_vector_store", fake_get_vector_store)

    class _FakeEmb:
        def generate_embedding(self, text: str):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    class _NoPast:
        async def find_similar_projects(self, **kwargs):
            return []

    weights = {t: 1.0 / len(DOCUMENT_TYPE_VALUES) for t in DOCUMENT_TYPE_VALUES}
    qs = QueryService(embedding_service=_FakeEmb(), project_similarity_service=_NoPast())
    out = await qs.retrieve_with_past_projects(
        query="q",
        project_name="proj",
        top_k=8,
        current_project_weight=1.0,
        past_projects_weight=0.0,
        document_weights=weights,
    )
    assert len(out["chunks"]) == 8
    proposal_calls = [c for c in search_calls if c.get("document_type") == DocumentType.PROPOSAL]
    assert len(proposal_calls) >= 1
    assert all(c["top_k"] >= 1 for c in proposal_calls)


def test_intent_info_slots_sum_to_current_budget():
    """API helper: intent-only path reports slot counts that sum to top_k."""
    from app.api.routes.query import _intent_info_from_result
    from app.models.query import QueryRequest
    from app.services.query_intent_service import QueryIntentResult

    u = 1.0 / len(DOCUMENT_TYPE_VALUES)
    ir = QueryIntentResult(
        intent_summary="s",
        document_weights={t: u for t in DOCUMENT_TYPE_VALUES},
        priority_order=None,
        used_fallback=False,
    )
    req = QueryRequest(query="q", project_name="p", top_k=10)
    info = _intent_info_from_result(
        ir,
        req,
        [],
        n_current_override=10,
        n_past_override=0,
    )
    assert info.n_current_slots == 10
    assert info.n_past_slots == 0
    assert sum(info.slots_current_project.values()) == 10


def test_intent_will_run_requires_flag_project_and_server_toggle(monkeypatch):
    from app.api.routes.query import _intent_will_run
    from app.config import settings
    from app.models.query import QueryRequest

    monkeypatch.setattr(settings, "QUERY_INTENT_ENABLED", True)
    req = QueryRequest(query="q", project_name="p", use_query_intent=True)
    assert _intent_will_run(req) is True

    monkeypatch.setattr(settings, "QUERY_INTENT_ENABLED", False)
    assert _intent_will_run(req) is False

    monkeypatch.setattr(settings, "QUERY_INTENT_ENABLED", True)
    req2 = QueryRequest(query="q", project_name=None, use_query_intent=True)
    assert _intent_will_run(req2) is False
