"""Phase 3–5.1: graph-augmented retrieval unit tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.graph_retrieval import (
    GraphAugmentationResult,
    _seed_chunk_ids,
    augment_chunks_with_graph,
    format_graph_context_lines,
    merge_vector_and_graph_chunks,
)


def _vector_chunk(
    chunk_id: str,
    project: str = "P1",
    *,
    past: bool = False,
    document_type: str = "statement of work",
):
    return {
        "text": f"text-{chunk_id}",
        "metadata": {
            "chunk_id": chunk_id,
            "project_name": project,
            "document_type": document_type,
        },
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
    mock_settings.GRAPH_MERGE_STRATEGY = "append"
    mock_settings.GRAPH_TEXT_SEARCH_SEED_ENABLED = False
    mock_settings.GRAPH_TEXT_SEARCH_SEED_LIMIT = 10
    mock_settings.GRAPH_CROSS_LINK_ENABLED = False
    mock_settings.GRAPH_CROSS_LINK_MAX_DEPTH = 3
    mock_settings.GRAPH_CROSS_LINK_EDGE_WHITELIST = (
        "IMPLEMENTS,ADDRESSES,TRACES_TO,SAME_AS,DESCRIBES_CURRENT_STATE_OF,GAPS"
    )


def test_seed_chunk_ids_current_project_only():
    chunks = [
        _vector_chunk("a"),
        _vector_chunk("b", project="Other"),
        _vector_chunk("c", past=True),
    ]
    assert _seed_chunk_ids(chunks, "P1") == ["a"]


def test_seed_chunk_ids_override():
    assert _seed_chunk_ids([], "P1", seed_chunk_ids=["x", "y"]) == ["x", "y"]


def test_merge_append_dedupes():
    vector = [_vector_chunk("a"), _vector_chunk("b")]
    graph = [_vector_chunk("b"), _vector_chunk("c")]
    merged, sources = merge_vector_and_graph_chunks(vector, graph, strategy="append")
    ids = [c["metadata"]["chunk_id"] for c in merged]
    assert ids == ["a", "b", "c"]
    assert merged[2]["metadata"].get("_graph_augmented") is True
    assert sources == {"a": "vector", "b": "both", "c": "graph"}


def test_merge_rrf_promotes_strong_graph_hit():
    # Weak vector tail + strong graph-first candidate should promote graph chunk upward
    vector = [_vector_chunk("v1"), _vector_chunk("v2"), _vector_chunk("v3")]
    graph = [_vector_chunk("g1"), _vector_chunk("v2")]
    merged, sources = merge_vector_and_graph_chunks(vector, graph, strategy="rrf")
    ids = [c["metadata"]["chunk_id"] for c in merged]
    assert "g1" in ids
    assert ids.index("g1") < ids.index("v3")
    assert sources["g1"] == "graph"
    assert sources["v2"] == "both"


def test_merge_diversity_prefers_new_document_type():
    vector = [
        _vector_chunk("v1", document_type="statement of work"),
        _vector_chunk("v2", document_type="statement of work"),
    ]
    graph = [
        _vector_chunk("g_sow", document_type="statement of work"),
        _vector_chunk("g_risk", document_type="risk register"),
    ]
    merged, _ = merge_vector_and_graph_chunks(vector, graph, strategy="diversity")
    ids = [c["metadata"]["chunk_id"] for c in merged]
    assert ids[:2] == ["v1", "v2"]
    assert ids[2] == "g_risk"
    assert ids[3] == "g_sow"


def test_format_graph_context_lines_strips_chunk_prefix():
    lines = format_graph_context_lines(["…-EXTRACTED_FROM->chunk:abc-444"])
    assert lines == ["- …-EXTRACTED_FROM->abc-444"]


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_no_project():
    result = await augment_chunks_with_graph([], project_name="")
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
    mock_store.text_search_seed = AsyncMock(return_value=MagicMock(node_ids=[]))

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

        result = await augment_chunks_with_graph(
            vector,
            project_name="P1",
            query="UAT deliverable",
        )

    assert isinstance(result, GraphAugmentationResult)
    assert [c["metadata"]["chunk_id"] for c in result.chunks] == ["seed-1", "extra-1"]
    assert result.added_chunk_ids == ["extra-1"]
    assert result.graph_context_lines
    assert result.vector_chunk_count == 1
    assert result.graph_added_count == 1
    assert result.degraded is False
    assert result.merge_strategy == "append"
    assert result.chunk_sources["extra-1"] == "graph"
    assert "total_ms" in result.timing_ms
    mock_store.neighborhood.assert_called_once()


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_text_search_seeds():
    vector = [_vector_chunk("seed-1")]

    neighborhood = MagicMock()
    neighborhood.chunk_ids = ["seed-1"]
    neighborhood.paths_summary = []
    neighborhood.node_ids = []

    text_linked = MagicMock()
    text_linked.chunk_ids = ["from-text"]
    text_linked.paths_summary = ["…-EXTRACTED_FROM->chunk:from-text"]
    text_linked.node_ids = ["del:doc:d12"]

    mock_store = AsyncMock()
    mock_store.neighborhood = AsyncMock(return_value=neighborhood)
    mock_store.text_search_seed = AsyncMock(
        return_value=MagicMock(node_ids=["del:doc:d12"], scores=[1.0])
    )
    mock_store.chunks_linked_to_nodes = AsyncMock(return_value=text_linked)

    mock_vs = AsyncMock()
    mock_vs.get_chunks_by_ids = AsyncMock(return_value=[_vector_chunk("from-text")])

    with patch("app.services.graph_retrieval.get_graph_store", AsyncMock(return_value=mock_store)), patch(
        "app.services.graph_retrieval.get_vector_store",
        AsyncMock(return_value=mock_vs),
    ), patch("app.services.graph_retrieval.is_graph_reachable", AsyncMock(return_value=True)), patch(
        "app.services.graph_retrieval.settings"
    ) as mock_settings:
        _mock_graph_settings(mock_settings)
        mock_settings.GRAPH_TEXT_SEARCH_SEED_ENABLED = True

        result = await augment_chunks_with_graph(
            vector,
            project_name="P1",
            query="What does deliverable D-12 require?",
        )

    assert "from-text" in result.added_chunk_ids
    assert result.text_seed_count == 1
    mock_store.text_search_seed.assert_called_once()
    mock_store.chunks_linked_to_nodes.assert_called_once()


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_degrades_on_neo4j_failure():
    vector = [_vector_chunk("seed-1")]
    mock_store = AsyncMock()
    mock_store.neighborhood = AsyncMock(side_effect=RuntimeError("neo4j down"))
    mock_store.text_search_seed = AsyncMock(return_value=MagicMock(node_ids=[]))

    with patch("app.services.graph_retrieval.get_graph_store", AsyncMock(return_value=mock_store)), patch(
        "app.services.graph_retrieval.is_graph_reachable", AsyncMock(return_value=True)
    ), patch("app.services.graph_retrieval.settings") as mock_settings:
        _mock_graph_settings(mock_settings)

        result = await augment_chunks_with_graph(vector, project_name="P1", query="q")

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
    mock_store.text_search_seed = AsyncMock(return_value=MagicMock(node_ids=[]))

    with patch("app.services.graph_retrieval.get_graph_store", AsyncMock(return_value=mock_store)), patch(
        "app.services.graph_retrieval.is_graph_reachable", AsyncMock(return_value=True)
    ), patch("app.services.graph_retrieval.settings") as mock_settings:
        _mock_graph_settings(mock_settings)
        mock_settings.GRAPH_MAX_SEEDS = 3

        await augment_chunks_with_graph(vector, project_name="P1", query="q")

    assert mock_store.neighborhood.call_count == 3


@pytest.mark.asyncio
async def test_augment_chunks_with_graph_cross_link_whitelist():
    vector = [_vector_chunk("seed-1")]
    neighborhood = MagicMock()
    neighborhood.chunk_ids = []
    neighborhood.paths_summary = []
    neighborhood.node_ids = []

    mock_store = AsyncMock()
    mock_store.neighborhood = AsyncMock(return_value=neighborhood)
    mock_store.text_search_seed = AsyncMock(return_value=MagicMock(node_ids=[]))

    with patch("app.services.graph_retrieval.get_graph_store", AsyncMock(return_value=mock_store)), patch(
        "app.services.graph_retrieval.is_graph_reachable", AsyncMock(return_value=True)
    ), patch("app.services.graph_retrieval.settings") as mock_settings:
        _mock_graph_settings(mock_settings)
        mock_settings.GRAPH_CROSS_LINK_ENABLED = True
        mock_settings.GRAPH_CROSS_LINK_MAX_DEPTH = 3

        await augment_chunks_with_graph(vector, project_name="P1", query="D-12")

    kwargs = mock_store.neighborhood.call_args.kwargs
    assert kwargs["depth"] == 3
    rels = kwargs["relationship_types"]
    assert "IMPLEMENTS" in rels
    assert "EXTRACTED_FROM" in rels
    assert "CONTAINS_CHUNK" in rels
