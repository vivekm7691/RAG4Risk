"""Phase 2: graph sync and relation extraction unit tests."""

import json

import pytest

from app.models.document import DocumentType, ProjectMetadata
from app.models.graph import EdgeType, NodeType
from app.services.graph_sync_service import (
    build_project_metadata_graph,
    build_structural_graph,
    extraction_to_graph,
)
from app.services.relation_extractor_service import (
    CHUNK_SOURCE_ID,
    SOW_SOURCE_ID,
    parse_extraction_json,
    validate_extraction_result,
)
from app.models.graph import (
    ExtractedEdge,
    ExtractedNode,
    RelationExtractionResult,
)


def _excel_chunk(row_number: int, chunk_id: str, description: str = "Late delivery"):
    return {
        "text": f"Risk Description: {description}, Severity: High",
        "metadata": {
            "chunk_id": chunk_id,
            "chunk_index": row_number - 12,
            "row_number": row_number,
            "sheet_name": "Risk Log (Detailed Info)",
            "severity": "High",
            "status": "Open",
            "description": description,
        },
    }


def test_build_structural_graph_excel_risk():
    chunks = [_excel_chunk(12, "c1"), _excel_chunk(13, "c2", "Budget overrun")]
    nodes, edges = build_structural_graph(
        chunks,
        document_id="doc-1",
        project_name="Acme",
        document_type=DocumentType.RISK_REGISTER.value,
        file_name="risks.xlsx",
    )
    node_ids = {n.node_id for n in nodes}
    assert "project:Acme" in node_ids
    assert "document:doc-1" in node_ids
    assert "chunk:c1" in node_ids
    assert "risk:doc-1:row:12" in node_ids
    assert "risk:doc-1:row:13" in node_ids

    edge_types = {(e.edge_type, e.source_id, e.target_id) for e in edges}
    assert (EdgeType.HAS_DOCUMENT, "project:Acme", "document:doc-1") in edge_types
    assert (EdgeType.CONTAINS_CHUNK, "document:doc-1", "chunk:c1") in edge_types
    assert (EdgeType.FROM_ROW, "risk:doc-1:row:12", "document:doc-1") in edge_types
    assert (EdgeType.HAS_RISK, "project:Acme", "risk:doc-1:row:12") in edge_types


def test_build_structural_graph_sow():
    chunks = [
        {
            "text": "The vendor shall deliver authentication module.",
            "metadata": {"chunk_id": "ch1", "chunk_index": 0},
        }
    ]
    nodes, edges = build_structural_graph(
        chunks,
        document_id="doc-sow",
        project_name="P1",
        document_type=DocumentType.STATEMENT_OF_WORK.value,
        file_name="sow.docx",
        title="SOW 2024",
    )
    node_ids = {n.node_id for n in nodes}
    assert "sow:doc-sow" in node_ids
    assert any(
        e.edge_type == EdgeType.REPRESENTS_SOW
        and e.source_id == "document:doc-sow"
        and e.target_id == "sow:doc-sow"
        for e in edges
    )


def test_build_project_metadata_graph():
    meta = ProjectMetadata(
        project_name="Acme",
        customer="BigCo",
        csg_products=["Billing", "Payments"],
    )
    nodes, edges = build_project_metadata_graph(meta)
    assert any(n.node_type == NodeType.CUSTOMER for n in nodes)
    assert any(n.node_type == NodeType.PRODUCT and "billing" in n.node_id for n in nodes)
    assert any(e.edge_type == EdgeType.USES_PRODUCT for e in edges)


def test_parse_extraction_json():
    raw = json.dumps(
        {
            "nodes": [
                {
                    "node_id": "n1",
                    "node_type": "Requirement",
                    "label": "Auth SLA",
                    "confidence": 0.9,
                }
            ],
            "edges": [
                {
                    "edge_type": "EXTRACTED_FROM",
                    "source_id": "n1",
                    "target_id": CHUNK_SOURCE_ID,
                    "confidence": 0.85,
                }
            ],
        }
    )
    result = parse_extraction_json(
        raw,
        chunk_id="ch1",
        document_id="doc1",
        project_name="P",
    )
    assert len(result.nodes) == 1
    assert len(result.edges) == 1


def test_extraction_to_graph_stable_ids():
    result = RelationExtractionResult(
        chunk_id="ch1",
        document_id="doc1",
        project_name="P",
        nodes=[
            ExtractedNode(
                node_id="n1",
                node_type=NodeType.REQUIREMENT,
                label="Auth SLA",
                confidence=0.9,
            )
        ],
        edges=[
            ExtractedEdge(
                edge_type=EdgeType.EXTRACTED_FROM,
                source_id="n1",
                target_id=CHUNK_SOURCE_ID,
                confidence=0.85,
            )
        ],
    )
    nodes, edges = extraction_to_graph(result, document_type="solution description document")
    assert nodes[0].node_id == "req:doc1:ch1:auth_sla"
    assert edges[0].target_id == "chunk:ch1"


def test_validate_extraction_sow_defines():
    result = RelationExtractionResult(
        chunk_id="ch1",
        document_id="doc1",
        project_name="P",
        nodes=[
            ExtractedNode(
                node_id="n1",
                node_type=NodeType.DELIVERABLE,
                label="Auth Module",
                confidence=0.9,
            )
        ],
        edges=[
            ExtractedEdge(
                edge_type=EdgeType.DEFINES,
                source_id=SOW_SOURCE_ID,
                target_id="n1",
                confidence=0.88,
            )
        ],
    )
    validated = validate_extraction_result(result, DocumentType.STATEMENT_OF_WORK.value)
    assert len(validated.edges) == 1
