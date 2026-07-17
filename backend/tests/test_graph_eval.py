"""Phase 5: graph eval recall@k unit tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.graph_eval import GraphEvalCase, GraphEvalSet
from app.services.graph_eval_service import (
    chunk_ids_from_results,
    format_report_text,
    recall_at_k,
    run_eval_set,
)


def test_recall_at_k_partial():
    recall, hit, missing = recall_at_k(
        ["a", "b", "c", "d"],
        {"a", "x", "y"},
        k=3,
    )
    assert recall == pytest.approx(1 / 3)
    assert hit is False
    assert missing == ["x", "y"]


def test_recall_at_k_full_hit():
    recall, hit, missing = recall_at_k(
        ["a", "b", "c"],
        {"a", "b"},
        k=3,
    )
    assert recall == 1.0
    assert hit is True
    assert missing == []


def test_recall_at_k_empty_must_include():
    recall, hit, missing = recall_at_k(["a"], set(), k=5)
    assert recall == 1.0
    assert hit is True
    assert missing == []


def test_chunk_ids_from_results():
    chunks = [
        {"metadata": {"chunk_id": "c1"}},
        {"metadata": {}},
        {"metadata": {"chunk_id": "c2"}},
    ]
    assert chunk_ids_from_results(chunks) == ["c1", "c2"]


@pytest.mark.asyncio
async def test_run_eval_set_skips_template_cases():
    eval_set = GraphEvalSet(
        name="test",
        cases=[
            GraphEvalCase(
                id="ok",
                query="q",
                project_name="P1",
                must_include_chunk_ids=["target"],
                top_k=5,
            ),
            GraphEvalCase(
                id="template",
                query="q2",
                project_name="P2",
                must_include_chunk_ids=[],
                top_k=5,
            ),
        ],
    )

    mock_chunks = [{"text": "t", "metadata": {"chunk_id": "target", "project_name": "P1"}}]

    with patch(
        "app.services.graph_eval_service.get_vector_store",
        AsyncMock(
            return_value=MagicMock(search=AsyncMock(return_value=mock_chunks))
        ),
    ), patch(
        "app.services.graph_eval_service.EmbeddingService",
        return_value=MagicMock(generate_embedding=MagicMock(return_value=MagicMock(tolist=lambda: [0.1]))),
    ), patch("app.services.graph_eval_service.settings") as mock_settings:
        mock_settings.GRAPH_ENABLED = False

        report = await run_eval_set(eval_set, compare_graph=False)

    assert report.summary.cases_run == 1
    assert report.summary.cases_skipped == 1
    assert report.vector_results[0].hit_at_k is True
    assert report.vector_results[1].skipped is True


@pytest.mark.asyncio
async def test_run_eval_set_graph_improves_recall():
    eval_set = GraphEvalSet(
        name="test",
        cases=[
            GraphEvalCase(
                id="graph-win",
                query="uat",
                project_name="P1",
                must_include_chunk_ids=["graph-only"],
                top_k=3,
            ),
        ],
    )
    vector_chunks = [{"text": "v", "metadata": {"chunk_id": "v1", "project_name": "P1"}}]
    graph_chunks = [
        {"text": "v", "metadata": {"chunk_id": "v1", "project_name": "P1"}},
        {"text": "g", "metadata": {"chunk_id": "graph-only", "project_name": "P1", "_graph_augmented": True}},
    ]

    graph_result = MagicMock()
    graph_result.chunks = graph_chunks
    graph_result.degraded = False
    graph_result.graph_added_count = 1

    with patch(
        "app.services.graph_eval_service.get_vector_store",
        AsyncMock(
            return_value=MagicMock(search=AsyncMock(return_value=vector_chunks))
        ),
    ), patch(
        "app.services.graph_eval_service.augment_chunks_with_graph",
        AsyncMock(return_value=graph_result),
    ), patch(
        "app.services.graph_eval_service.EmbeddingService",
        return_value=MagicMock(generate_embedding=MagicMock(return_value=MagicMock(tolist=lambda: [0.1]))),
    ), patch("app.services.graph_eval_service.settings") as mock_settings:
        mock_settings.GRAPH_ENABLED = True

        report = await run_eval_set(eval_set)

    assert report.vector_results[0].hit_at_k is False
    assert report.graph_results[0].hit_at_k is True
    assert report.summary.graph_improved_count == 1
    assert "graph-win" in format_report_text(report)
