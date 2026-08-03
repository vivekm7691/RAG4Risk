"""Phase 2: sync documents and project metadata into the knowledge graph."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.models.document import DocumentType, ProjectMetadata
from app.models.graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    RelationExtractionResult,
    SEMANTIC_ID_PREFIX,
    SOW_DOCUMENT_TYPE,
    graph_id_chunk,
    graph_id_customer,
    graph_id_document,
    graph_id_issue_row,
    graph_id_product,
    graph_id_project,
    graph_id_risk_row,
    graph_id_semantic,
    graph_id_statement_of_work,
    graph_id_system_component,
)
from app.services.graph_store import get_graph_store
from app.services.relation_extractor_service import (
    CHUNK_SOURCE_ID,
    PROJECT_SOURCE_ID,
    SOW_SOURCE_ID,
    extract_relations_from_chunk,
)
from app.services.project_metadata import ProjectMetadataService

logger = logging.getLogger(__name__)


def build_structural_graph(
    chunks: List[Dict[str, Any]],
    *,
    document_id: str,
    project_name: str,
    document_type: str,
    file_name: str,
    title: Optional[str] = None,
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """Deterministic Project / Document / Chunk (+ Excel Risk/Issue) nodes and edges."""
    project_name = project_name.strip()
    doc_label = (title or file_name or document_id).strip()
    proj_id = graph_id_project(project_name)
    doc_id = graph_id_document(document_id)

    nodes: List[GraphNode] = [
        GraphNode(
            node_id=proj_id,
            node_type=NodeType.PROJECT,
            label=project_name,
            project_name=project_name,
        ),
        GraphNode(
            node_id=doc_id,
            node_type=NodeType.DOCUMENT,
            label=doc_label,
            project_name=project_name,
            properties={
                "document_id": document_id,
                "document_type": document_type,
                "file_name": file_name,
            },
        ),
    ]
    edges: List[GraphEdge] = [
        GraphEdge(
            edge_type=EdgeType.HAS_DOCUMENT,
            source_id=proj_id,
            target_id=doc_id,
            project_name=project_name,
        ),
    ]

    if document_type == SOW_DOCUMENT_TYPE:
        sow_id = graph_id_statement_of_work(document_id)
        nodes.append(
            GraphNode(
                node_id=sow_id,
                node_type=NodeType.STATEMENT_OF_WORK,
                label=doc_label,
                project_name=project_name,
                properties={"document_id": document_id},
            )
        )
        edges.append(
            GraphEdge(
                edge_type=EdgeType.REPRESENTS_SOW,
                source_id=doc_id,
                target_id=sow_id,
                project_name=project_name,
            )
        )

    is_excel = document_type in (
        DocumentType.RISK_REGISTER.value,
        DocumentType.ISSUE_LOG.value,
    )

    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        chunk_uuid = meta.get("chunk_id")
        if not chunk_uuid:
            continue
        chunk_gid = graph_id_chunk(chunk_uuid)
        chunk_label = (chunk.get("text") or "")[:120].strip() or f"chunk {meta.get('chunk_index', 0)}"

        nodes.append(
            GraphNode(
                node_id=chunk_gid,
                node_type=NodeType.CHUNK,
                label=chunk_label,
                project_name=project_name,
                properties={
                    "document_id": document_id,
                    "chunk_id": chunk_uuid,
                    "chunk_index": meta.get("chunk_index"),
                },
            )
        )
        edges.append(
            GraphEdge(
                edge_type=EdgeType.CONTAINS_CHUNK,
                source_id=doc_id,
                target_id=chunk_gid,
                project_name=project_name,
            )
        )

        if is_excel:
            row_number = meta.get("row_number")
            if row_number is None:
                continue
            try:
                row_num = int(row_number)
            except (TypeError, ValueError):
                continue

            if document_type == DocumentType.RISK_REGISTER.value:
                entity_gid = graph_id_risk_row(document_id, row_num)
                entity_type = NodeType.RISK
                row_label = _excel_entity_label(meta, "risk")
            else:
                entity_gid = graph_id_issue_row(document_id, row_num)
                entity_type = NodeType.ISSUE
                row_label = _excel_entity_label(meta, "issue")

            row_props = {
                k: meta[k]
                for k in (
                    "severity",
                    "status",
                    "category",
                    "owner",
                    "description",
                    "mitigation",
                    "date",
                    "row_number",
                    "sheet_name",
                    "chunk_id",
                )
                if meta.get(k) is not None
            }
            row_props["document_id"] = document_id
            row_props["chunk_id"] = chunk_uuid

            nodes.append(
                GraphNode(
                    node_id=entity_gid,
                    node_type=entity_type,
                    label=row_label,
                    project_name=project_name,
                    properties=row_props,
                )
            )
            edges.append(
                GraphEdge(
                    edge_type=EdgeType.FROM_ROW,
                    source_id=entity_gid,
                    target_id=doc_id,
                    project_name=project_name,
                    evidence_text=chunk.get("text"),
                )
            )
            if entity_type == NodeType.RISK:
                edges.append(
                    GraphEdge(
                        edge_type=EdgeType.RECORDED_IN,
                        source_id=entity_gid,
                        target_id=doc_id,
                        project_name=project_name,
                    )
                )
                edges.append(
                    GraphEdge(
                        edge_type=EdgeType.IS_PART_OF,
                        source_id=entity_gid,
                        target_id=proj_id,
                        project_name=project_name,
                    )
                )
                edges.append(
                    GraphEdge(
                        edge_type=EdgeType.HAS_RISK,
                        source_id=proj_id,
                        target_id=entity_gid,
                        project_name=project_name,
                    )
                )

    return nodes, edges


def _excel_entity_label(meta: Dict[str, Any], kind: str) -> str:
    desc = (meta.get("description") or "").strip()
    if desc:
        return desc[:200]
    row_id = meta.get(f"{kind}_id") or meta.get("risk_id") or meta.get("issue_id")
    if row_id:
        return f"{kind.title()} {row_id}"
    return f"{kind.title()} row {meta.get('row_number', '?')}"


def extraction_to_graph(
    result: RelationExtractionResult,
    *,
    document_type: str,
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """Map validated RelationExtractionResult to stable graph nodes/edges."""
    project_name = result.project_name.strip()
    doc_id = graph_id_document(result.document_id)
    chunk_gid = graph_id_chunk(result.chunk_id)
    proj_gid = graph_id_project(project_name)
    sow_gid = graph_id_statement_of_work(result.document_id)

    local_to_stable: Dict[str, str] = {
        CHUNK_SOURCE_ID: chunk_gid,
        PROJECT_SOURCE_ID: proj_gid,
        SOW_SOURCE_ID: sow_gid if document_type == SOW_DOCUMENT_TYPE else doc_id,
    }

    nodes: List[GraphNode] = []
    for en in result.nodes:
        if en.node_type == NodeType.SYSTEM_COMPONENT:
            # Project-scoped hub — do not stamp document_id (survives doc re-ingest)
            stable_id = graph_id_system_component(project_name, en.label)
            local_to_stable[en.node_id] = stable_id
            props = dict(en.properties)
            props["source"] = props.get("source") or "extraction"
            nodes.append(
                GraphNode(
                    node_id=stable_id,
                    node_type=NodeType.SYSTEM_COMPONENT,
                    label=en.label,
                    project_name=project_name,
                    properties=props,
                    confidence=en.confidence,
                )
            )
            continue
        prefix = SEMANTIC_ID_PREFIX.get(en.node_type)
        if not prefix:
            continue
        stable_id = graph_id_semantic(
            prefix,
            result.document_id,
            result.chunk_id,
            en.label,
        )
        local_to_stable[en.node_id] = stable_id
        props = dict(en.properties)
        props["document_id"] = result.document_id
        props["chunk_id"] = result.chunk_id
        nodes.append(
            GraphNode(
                node_id=stable_id,
                node_type=en.node_type,
                label=en.label,
                project_name=project_name,
                properties=props,
                confidence=en.confidence,
            )
        )

    edges: List[GraphEdge] = []
    for ee in result.edges:
        src = local_to_stable.get(ee.source_id)
        tgt = local_to_stable.get(ee.target_id)
        if not src or not tgt:
            continue
        edges.append(
            GraphEdge(
                edge_type=ee.edge_type,
                source_id=src,
                target_id=tgt,
                project_name=project_name,
                evidence_text=ee.evidence_text,
                confidence=ee.confidence,
            )
        )

    return nodes, edges


def build_project_metadata_graph(
    meta: ProjectMetadata,
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """Project + Customer + Product nodes from SQLite project metadata."""
    project_name = meta.project_name.strip()
    proj_id = graph_id_project(project_name)
    nodes: List[GraphNode] = [
        GraphNode(
            node_id=proj_id,
            node_type=NodeType.PROJECT,
            label=project_name,
            project_name=project_name,
            properties={
                "customer": meta.customer,
                "csg_role": meta.csg_role,
                "integration_complexity": meta.integration_complexity,
                "client_type": meta.client_type,
                "project_size": meta.project_size,
                "project_complexity": meta.project_complexity,
            },
        ),
        GraphNode(
            node_id=graph_id_customer(meta.customer),
            node_type=NodeType.CUSTOMER,
            label=meta.customer,
            project_name=project_name,
        ),
    ]
    edges: List[GraphEdge] = []
    for product in meta.csg_products or []:
        if not (product or "").strip():
            continue
        prod_id = graph_id_product(product)
        nodes.append(
            GraphNode(
                node_id=prod_id,
                node_type=NodeType.PRODUCT,
                label=product.strip(),
                project_name=project_name,
            )
        )
        edges.append(
            GraphEdge(
                edge_type=EdgeType.USES_PRODUCT,
                source_id=proj_id,
                target_id=prod_id,
                project_name=project_name,
            )
        )
        # Phase 5.2: mirror products as SystemComponent hubs for SOW↔solution bridging
        comp_id = graph_id_system_component(project_name, product)
        nodes.append(
            GraphNode(
                node_id=comp_id,
                node_type=NodeType.SYSTEM_COMPONENT,
                label=product.strip(),
                project_name=project_name,
                properties={"source": "project_metadata"},
                confidence=1.0,
            )
        )
    return nodes, edges


async def sync_document_to_graph(
    chunks: List[Dict[str, Any]],
    *,
    document_id: str,
    project_name: str,
    document_type: str,
    file_name: str,
    title: Optional[str] = None,
) -> Dict[str, int]:
    """
    Replace graph subgraph for document_id and upsert structural (+ optional LLM) nodes.

    Returns counts: nodes, edges, chunks_extracted.
    """
    if not settings.GRAPH_ENABLED:
        return {"nodes": 0, "edges": 0, "chunks_extracted": 0}

    store = await get_graph_store()
    await store.delete_document_subgraph(document_id)

    nodes, edges = build_structural_graph(
        chunks,
        document_id=document_id,
        project_name=project_name,
        document_type=document_type,
        file_name=file_name,
        title=title,
    )

    chunks_extracted = 0
    is_word = document_type not in (
        DocumentType.RISK_REGISTER.value,
        DocumentType.ISSUE_LOG.value,
    )

    if is_word and settings.GRAPH_EXTRACT_ON_INGEST:
        max_chunks = max(0, settings.GRAPH_EXTRACT_MAX_CHUNKS)
        llm_chunks = chunks[:max_chunks] if max_chunks else chunks
        if len(chunks) > len(llm_chunks):
            logger.info(
                "Graph LLM extraction capped at %d/%d chunks for document_id=%s",
                len(llm_chunks),
                len(chunks),
                document_id,
            )
        for chunk in llm_chunks:
            meta = chunk.get("metadata") or {}
            chunk_id = meta.get("chunk_id")
            if not chunk_id:
                continue
            extraction = await extract_relations_from_chunk(
                chunk.get("text") or "",
                chunk_id=chunk_id,
                document_id=document_id,
                project_name=project_name,
                document_type=document_type,
            )
            if extraction.nodes or extraction.edges:
                en, ee = extraction_to_graph(extraction, document_type=document_type)
                nodes.extend(en)
                edges.extend(ee)
                chunks_extracted += 1

    # Ensure project metadata nodes exist when available
    meta_service = ProjectMetadataService()
    pm = meta_service.get_project_metadata(project_name)
    if pm:
        pn, pe = build_project_metadata_graph(pm)
        nodes.extend(pn)
        edges.extend(pe)

    await store.upsert_nodes(nodes)
    await store.upsert_edges(edges)

    link_stats: Dict[str, int] = {}
    if settings.GRAPH_CROSS_LINK_ON_INGEST is True:
        from app.services.entity_linking_service import resolve_project_cross_links

        link_stats = await resolve_project_cross_links(project_name)

    logger.info(
        "Graph sync document_id=%s nodes=%d edges=%d chunks_extracted=%d cross_link=%s",
        document_id,
        len(nodes),
        len(edges),
        chunks_extracted,
        link_stats or "skipped",
    )
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "chunks_extracted": chunks_extracted,
        **{f"cross_link_{k}": v for k, v in link_stats.items()},
    }


async def sync_project_metadata_to_graph(meta: ProjectMetadata) -> Dict[str, int]:
    """Upsert Project / Customer / Product nodes from project metadata API."""
    if not settings.GRAPH_ENABLED:
        return {"nodes": 0, "edges": 0}

    nodes, edges = build_project_metadata_graph(meta)
    store = await get_graph_store()
    await store.upsert_nodes(nodes)
    await store.upsert_edges(edges)
    logger.info(
        "Graph sync project=%s nodes=%d edges=%d",
        meta.project_name,
        len(nodes),
        len(edges),
    )
    return {"nodes": len(nodes), "edges": len(edges)}


async def delete_document_from_graph(document_id: str) -> int:
    """Cascade delete graph nodes for a document."""
    if not settings.GRAPH_ENABLED:
        return 0
    store = await get_graph_store()
    deleted = await store.delete_document_subgraph(document_id)
    logger.info("Graph deleted document_id=%s nodes=%d", document_id, deleted)
    return deleted
