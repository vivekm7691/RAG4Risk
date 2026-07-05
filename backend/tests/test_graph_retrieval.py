"""Phase 3–4: graph-augmented retrieval unit tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.graph_retrieval import (
    GraphAugmentationResult,
    _seed_chunk_ids,
    augment_chunks_with_graph,
    format_graph_context_lines,
    merge_vector_and_graph_chunks,
)


def _vector_chunk(chunk_id: str, project: str = "P1", *, past: bool = False):
    return {
        "text": f"text-{chunk_id}",
        "metadata": {"chunk_id": chunk_id, "project_name": project, "document_type": "statement of work"},
        "is_past_project": past,
    }


def _mock_graph_settings(mock_settings):
    mock_settings.GRAPH_ENABLED = True
    mock_settings.GRAPH_RETRIEVAL_EXTRA_BUDGET = 5
    mock_settings.GRAPH_NEIGHBORHOOD_DEFAULT_DEPTH = 2
    mock_settings.GRAPH_NEIGHBORHOOD_DEFAULT_LIMIT = 50
    mock_settings.GRAPH_QUERY_TIMEOUT_SECONDS = 10.0
    mock_settings.GRAPH_MAX_DEGREE_PER_SEED = 25
    mock_settings.GRAPH_MAX_SEEDS = 10


def test_seed_chunk_ids_current_project_only():
    chunks = [
        _vector_chunk("a"),
        _vector_chunk("b", project="Other"),
        _vector_chunk("c", past=True),
    ]
    assert _seed_chunk_ids(chunks, "P1") == ["a"]


def test_merge_vector_and_graph_chunks_dedupes():
    vector = [_vector_chunk("a"), _vector_chunk("b")]
    graph = [_vector_chunk("b"), _vector_chunk("c")]
    merged = merge_vector_and_graph_chunks(vector, graph)
    ids = [c["metadata"]["chunk_id"] for c in merged]
    assert ids == ["a", "b", "c"]
    assert merged[2]["metadata"].get("_graph_augmented") is True


def test_format_graph_context_lines_strips_chunk_prefix():
    lines = format_graph_context_lines(["…-EXTRACTED_FROM->chunk:abc-444"])
    assert lines == ["- …-EXTRACTED_FROM->abc-444"]


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_no_seeds():
    result = await augment_chunks_with_graph([], project_name="P1")
    assert result.chunks == []


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_adds_neighbors():
    vector = [_vector_chunk("seed-1")]

    neighborhood = MagicMock()
    neighborhood.chunk_ids = ["seed-1", "extra-1"]
    neighborhood.paths_summary = ["…-EXTRACTED_FROM->chunk:extra-1"]
    neighborhood.node_ids = []

    mock_store = AsyncMock()
    mock_store.neighborhood = AsyncMock(return_value=neighborhood)

    mock_vs = AsyncMock()
    mock_vs.get_chunks_by_ids = AsyncMock(
        return_value=[_vector_chunk("extra-1")]
    )

    with patch("app.services.graph_retrieval.get_graph_store", AsyncMock(return_value=mock_store)), patch(
        "app.services.graph_retrieval.get_vector_store",
        AsyncMock(return_value=mock_vs),
    ), patch("app.services.graph_retrieval.is_graph_reachable", AsyncMock(return_value=True)), patch(
        "app.services.graph_retrieval.settings"
    ) as mock_settings:
        _mock_graph_settings(mock_settings)

        result = await augment_chunks_with_graph(vector, project_name="P1")

    assert isinstance(result, GraphAugmentationResult)
    assert [c["metadata"]["chunk_id"] for c in result.chunks] == ["seed-1", "extra-1"]
    assert result.added_chunk_ids == ["extra-1"]
    assert result.graph_context_lines
    assert result.vector_chunk_count == 1
    assert result.graph_added_count == 1
    assert result.degraded is False
    assert "total_ms" in result.timing_ms
    mock_store.neighborhood.assert_called_once()


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_degrades_on_neo4j_failure():
    vector = [_vector_chunk("seed-1")]
    mock_store = AsyncMock()
    mock_store.neighborhood = AsyncMock(side_effect=RuntimeError("neo4j down"))

    with patch("app.services.graph_retrieval.get_graph_store", AsyncMock(return_value=mock_store)), patch(
        "app.services.graph_retrieval.is_graph_reachable", AsyncMock(return_value=True)
    ), patch("app.services.graph_retrieval.settings") as mock_settings:
        _mock_graph_settings(mock_settings)

        result = await augment_chunks_with_graph(vector, project_name="P1")

    assert len(result.chunks) == 1
    assert result.added_chunk_ids == []
    assert result.degraded is True
    assert result.degrade_reason == "graph_neighborhood_error"


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_degrades_when_unreachable():
    vector = [_vector_chunk("seed-1")]

    with patch(
        "app.services.graph_retrieval.is_graph_reachable", AsyncMock(return_value=False)
    ), patch("app.services.graph_retrieval.settings") as mock_settings:
        _mock_graph_settings(mock_settings)

        result = await augment_chunks_with_graph(vector, project_name="P1")

    assert len(result.chunks) == 1
    assert result.degraded is True
    assert result.degrade_reason == "neo4j_unreachable"


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_caps_seeds():
    vector = [_vector_chunk(f"s{i}") for i in range(15)]

    neighborhood = MagicMock()
    neighborhood.chunk_ids = []
    neighborhood.paths_summary = []
    neighborhood.node_ids = []

    mock_store = AsyncMock()
    mock_store.neighborhood = AsyncMock(return_value=neighborhood)

    with patch("app.services.graph_retrieval.get_graph_store", AsyncMock(return_value=mock_store)), patch(
        "app.services.graph_retrieval.is_graph_reachable", AsyncMock(return_value=True)
    ), patch("app.services.graph_retrieval.settings") as mock_settings:
        _mock_graph_settings(mock_settings)
        mock_settings.GRAPH_MAX_SEEDS = 3

        await augment_chunks_with_graph(vector, project_name="P1")

    assert mock_store.neighborhood.call_count == 3
