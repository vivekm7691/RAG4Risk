"""Phase 1: GraphStore unit tests (Neo4j integration optional via GRAPH_INTEGRATION_TESTS=1)."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.graph import EdgeType, GraphEdge, GraphNode, NodeType
from app.services.graph_store import (
    DisabledGraphStore,
    Neo4jGraphStore,
    _chunk_graph_ids,
    _node_props,
    _strip_chunk_prefix,
    document_id_from_node_id,
    get_graph_store,
    reset_graph_store,
)


def test_document_id_from_node_id():
    assert document_id_from_node_id("document:abc") == "abc"
    assert document_id_from_node_id("sow:abc") == "abc"
    assert document_id_from_node_id("risk:doc1:row:3") == "doc1"
    assert document_id_from_node_id("issue:doc1:row:1") == "doc1"
    assert document_id_from_node_id("req:doc1:ch1:auth_sla") == "doc1"
    assert document_id_from_node_id("chunk:uuid") is None
    assert document_id_from_node_id("project:Acme") is None


def test_chunk_graph_id_helpers():
    assert _chunk_graph_ids(["abc", "chunk:xyz"]) == ["chunk:abc", "chunk:xyz"]
    assert _strip_chunk_prefix("chunk:abc") == "abc"
    assert _strip_chunk_prefix("abc") == "abc"


def test_node_props_sets_document_id():
    node = GraphNode(
        node_id="req:doc1:ch1:auth",
        node_type=NodeType.REQUIREMENT,
        label="Auth SLA",
        project_name="P1",
    )
    props = _node_props(node)
    assert props["document_id"] == "doc1"
    assert props["node_type"] == "Requirement"
    assert props["project_name"] == "P1"


@pytest.mark.asyncio
async def test_disabled_graph_store_noops():
    store = DisabledGraphStore()
    await store.ensure_schema()
    await store.upsert_nodes([])
    await store.upsert_edges([])
    assert await store.delete_project_subgraph("P") == 0
    assert await store.delete_document_subgraph("d1") == 0
    nb = await store.neighborhood(["c1"], depth=2, limit=10, project_name="P")
    assert nb.chunk_ids == []
    ts = await store.text_search_seed("risk", limit=5, project_name="P")
    assert ts.node_ids == []


@pytest.mark.asyncio
async def test_get_graph_store_disabled(monkeypatch):
    monkeypatch.setattr("app.services.graph_store.settings.GRAPH_ENABLED", False)
    await reset_graph_store()
    store = await get_graph_store()
    assert isinstance(store, DisabledGraphStore)
    await reset_graph_store()


@pytest.mark.asyncio
async def test_neo4j_upsert_nodes_batches_by_label():
    store = Neo4jGraphStore("bolt://localhost:7687", "neo4j", "pass")
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    store._driver = mock_driver
    store._schema_ready = True

    nodes = [
        GraphNode(
            node_id="chunk:c1",
            node_type=NodeType.CHUNK,
            label="chunk",
            project_name="P",
            properties={"document_id": "d1"},
        ),
        GraphNode(
            node_id="document:d1",
            node_type=NodeType.DOCUMENT,
            label="doc",
            project_name="P",
        ),
    ]
    await store.upsert_nodes(nodes)

    assert mock_session.run.await_count == 2
    calls = [c.args[0] for c in mock_session.run.await_args_list]
    assert any("GraphNode:Chunk" in q for q in calls)
    assert any("GraphNode:Document" in q for q in calls)


@pytest.mark.asyncio
async def test_neo4j_neighborhood_empty_seeds():
    store = Neo4jGraphStore("bolt://localhost:7687", "neo4j", "pass")
    result = await store.neighborhood([], depth=2, limit=10, project_name="P")
    assert result.chunk_ids == []


@pytest.mark.asyncio
async def test_neo4j_delete_project_subgraph():
    store = Neo4jGraphStore("bolt://localhost:7687", "neo4j", "pass")
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value={"deleted": 3})

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    store._driver = mock_driver

    deleted = await store.delete_project_subgraph("Acme")
    assert deleted == 3


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("GRAPH_INTEGRATION_TESTS", "").lower() not in ("1", "true", "yes"),
    reason="Set GRAPH_INTEGRATION_TESTS=1 and run Neo4j to execute",
)
async def test_neo4j_integration_roundtrip():
    """Live Neo4j: upsert, neighborhood, delete document subgraph."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "rag4risk-dev")

    store = Neo4jGraphStore(uri, user, password)
    try:
        await store.ensure_schema()
        project = "test-graph-phase1"
        doc_id = "integration-doc-1"
        chunk_id = "integration-chunk-1"

        await store.delete_project_subgraph(project)

        await store.upsert_nodes(
            [
                GraphNode(
                    node_id=f"project:{project}",
                    node_type=NodeType.PROJECT,
                    label=project,
                    project_name=project,
                ),
                GraphNode(
                    node_id=f"document:{doc_id}",
                    node_type=NodeType.DOCUMENT,
                    label="Test doc",
                    project_name=project,
                    properties={"document_id": doc_id},
                ),
                GraphNode(
                    node_id=f"chunk:{chunk_id}",
                    node_type=NodeType.CHUNK,
                    label="chunk text",
                    project_name=project,
                    properties={"document_id": doc_id},
                ),
            ]
        )
        await store.upsert_edges(
            [
                GraphEdge(
                    edge_type=EdgeType.HAS_DOCUMENT,
                    source_id=f"project:{project}",
                    target_id=f"document:{doc_id}",
                    project_name=project,
                ),
                GraphEdge(
                    edge_type=EdgeType.CONTAINS_CHUNK,
                    source_id=f"document:{doc_id}",
                    target_id=f"chunk:{chunk_id}",
                    project_name=project,
                ),
            ]
        )

        nb = await store.neighborhood(
            [chunk_id],
            depth=1,
            limit=10,
            project_name=project,
        )
        assert chunk_id in nb.chunk_ids or f"document:{doc_id}" in nb.node_ids

        deleted = await store.delete_document_subgraph(doc_id)
        assert deleted >= 2

        await store.delete_project_subgraph(project)
    finally:
        await store.close()
