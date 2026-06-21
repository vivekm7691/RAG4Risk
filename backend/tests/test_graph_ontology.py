"""Phase 0: graph ontology IDs and edge validation."""

from app.models.graph import (
    EdgeType,
    NodeType,
    RelationExtractionResult,
    ExtractedNode,
    ExtractedEdge,
    SOW_DOCUMENT_TYPE,
    filter_extraction_by_confidence,
    graph_id_chunk,
    graph_id_risk_row,
    graph_id_semantic,
    graph_id_statement_of_work,
    normalize_slug,
    validate_edge_endpoints,
)


def test_graph_id_helpers():
    assert graph_id_chunk("abc-123") == "chunk:abc-123"
    assert graph_id_risk_row("doc1", 5) == "risk:doc1:row:5"
    assert graph_id_semantic("req", "doc1", "ch1", "Auth latency SLA") == "req:doc1:ch1:auth_latency_sla"
    assert graph_id_statement_of_work("doc1") == "sow:doc1"


def test_normalize_slug():
    assert normalize_slug("  Hello World! ") == "hello_world"
    assert normalize_slug("") == "unknown"


def test_validate_mitigates_edge():
    assert validate_edge_endpoints(EdgeType.MITIGATES, NodeType.CONTROL, NodeType.RISK)
    assert not validate_edge_endpoints(EdgeType.MITIGATES, NodeType.RISK, NodeType.CONTROL)


def test_validate_enterprise_defines_sow_to_deliverable():
    assert validate_edge_endpoints(
        EdgeType.DEFINES, NodeType.STATEMENT_OF_WORK, NodeType.DELIVERABLE
    )
    assert validate_edge_endpoints(
        EdgeType.DEFINES,
        NodeType.DOCUMENT,
        NodeType.DELIVERABLE,
        source_document_type=SOW_DOCUMENT_TYPE,
    )
    assert not validate_edge_endpoints(
        EdgeType.DEFINES,
        NodeType.DOCUMENT,
        NodeType.DELIVERABLE,
        source_document_type="proposal document",
    )


def test_validate_is_part_of_risk_to_project():
    assert validate_edge_endpoints(EdgeType.IS_PART_OF, NodeType.RISK, NodeType.PROJECT)


def test_filter_extraction_by_confidence():
    r = RelationExtractionResult(
        chunk_id="c1",
        document_id="d1",
        project_name="P",
        nodes=[
            ExtractedNode(node_id="n1", node_type=NodeType.RISK, label="R1", confidence=0.9),
            ExtractedNode(node_id="n2", node_type=NodeType.CONTROL, label="C1", confidence=0.5),
        ],
        edges=[
            ExtractedEdge(
                edge_type=EdgeType.MITIGATES,
                source_id="n2",
                target_id="n1",
                confidence=0.8,
            ),
        ],
    )
    out = filter_extraction_by_confidence(r, min_confidence=0.75)
    assert len(out.nodes) == 1
    assert out.nodes[0].node_id == "n1"
    assert len(out.edges) == 0  # n2 dropped, edge orphaned
