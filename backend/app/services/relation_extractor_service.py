"""Phase 2: LLM relation extraction from Word document chunks (Ollama /api/chat JSON)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.models.graph import (
    DEFAULT_MIN_CONFIDENCE,
    SEMANTIC_NODES_BY_DOCUMENT_TYPE,
    SOW_DOCUMENT_TYPE,
    SOW_EXTRACTABLE_EDGES,
    EdgeType,
    ExtractedEdge,
    ExtractedNode,
    NodeType,
    RelationExtractionResult,
    validate_edge_endpoints,
)
from app.services.query_intent_service import extract_json_object, resolve_intent_ollama_base_url

logger = logging.getLogger(__name__)


def _format_extraction_error(
    exc: BaseException,
    *,
    timeout_seconds: float,
    model: str,
) -> str:
    """Human-readable failure text; httpx timeouts often have an empty str()."""
    name = type(exc).__name__
    if isinstance(exc, httpx.TimeoutException):
        return f"{name} after {timeout_seconds:g}s (model={model})"
    msg = str(exc).strip()
    if msg:
        return f"{name}: {msg}"
    return f"{name} ({exc!r})"


# Sentinel IDs in extractor JSON (mapped to stable graph IDs on ingest)
SOW_SOURCE_ID = "sow"
CHUNK_SOURCE_ID = "chunk"
PROJECT_SOURCE_ID = "project"


class _RawExtractionJson(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


def _allowed_node_types(document_type: str) -> List[str]:
    types = SEMANTIC_NODES_BY_DOCUMENT_TYPE.get(document_type, frozenset())
    return sorted(t.value for t in types)


def _allowed_edge_types(document_type: str) -> List[str]:
    if document_type == SOW_DOCUMENT_TYPE:
        edges = SOW_EXTRACTABLE_EDGES
    else:
        edges = frozenset(
            {
                EdgeType.EXTRACTED_FROM,
                EdgeType.DESCRIBES,
                EdgeType.MITIGATES,
                EdgeType.REFERENCES,
                EdgeType.RELATES_TO,
                EdgeType.IMPLIES_ASSUMPTION,
            }
        )
    return sorted(e.value for e in edges)


def build_extraction_system_prompt(document_type: str) -> str:
    node_types = _allowed_node_types(document_type)
    edge_types = _allowed_edge_types(document_type)
    return f"""You extract knowledge-graph entities and relationships from a single document chunk.

Document type: "{document_type}"

Allowed node types (use exact strings): {json.dumps(node_types)}
Allowed edge types (use exact strings): {json.dumps(edge_types)}

Rules:
- Output ONLY one JSON object (no markdown fences).
- Use local node_id strings (e.g. "n1", "n2") unique within this chunk.
- For edges from the SOW hub use source_id "{SOW_SOURCE_ID}" (only when document_type is statement of work).
- For edges from the chunk use source_id "{CHUNK_SOURCE_ID}" where appropriate (e.g. IMPLIES_ASSUMPTION, DESCRIBES).
- For edges from the project use source_id "{PROJECT_SOURCE_ID}".
- Include confidence 0.0-1.0 per node and edge; omit uncertain items (confidence < 0.75).
- Assumption nodes must set properties.is_explicit true/false.

JSON shape:
{{"nodes": [{{"node_id": "n1", "node_type": "Requirement", "label": "short label", "properties": {{}}, "confidence": 0.9}}],
  "edges": [{{"edge_type": "EXTRACTED_FROM", "source_id": "n1", "target_id": "{CHUNK_SOURCE_ID}", "evidence_text": "quote", "confidence": 0.85}}]}}
"""


def build_extraction_user_message(chunk_text: str) -> str:
    text = (chunk_text or "").strip()
    if len(text) > 6000:
        text = text[:6000] + "\n...[truncated]"
    return f'Extract entities and relationships from this chunk:\n"""\n{text}\n"""'


async def ollama_extract_relations(
    chunk_text: str,
    *,
    document_type: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """Call Ollama /api/chat and return raw assistant text for JSON parsing."""
    bu = resolve_intent_ollama_base_url(base_url)
    m = model or settings.INTENT_LLM_MODEL
    to = settings.GRAPH_EXTRACT_TIMEOUT_SECONDS if timeout is None else timeout

    options: Dict[str, Any] = {
        "num_predict": settings.INTENT_LLM_MAX_TOKENS,
        "temperature": settings.INTENT_LLM_TEMPERATURE,
    }
    nctx = int(getattr(settings, "OLLAMA_NUM_CTX", 0) or 0)
    if nctx > 0:
        options["num_ctx"] = nctx

    payload: Dict[str, Any] = {
        "model": m,
        "messages": [
            {"role": "system", "content": build_extraction_system_prompt(document_type)},
            {"role": "user", "content": build_extraction_user_message(chunk_text)},
        ],
        "stream": False,
        "options": options,
    }

    url = f"{bu}/api/chat"
    async with httpx.AsyncClient(timeout=to) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    msg = data.get("message") or {}
    content = (msg.get("content") or "").strip()
    thinking = (msg.get("thinking") or "").strip()
    if content:
        return content
    if thinking:
        return thinking
    raise RuntimeError("Relation extractor: empty Ollama response")


def _filter_extraction_result(
    result: RelationExtractionResult,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> RelationExtractionResult:
    """Drop low-confidence items; keep edges to/from sentinel hub ids."""
    nodes = [n for n in result.nodes if n.confidence >= min_confidence]
    kept_ids = {n.node_id for n in nodes}
    sentinels = {CHUNK_SOURCE_ID, SOW_SOURCE_ID, PROJECT_SOURCE_ID}
    edges = [
        e
        for e in result.edges
        if e.confidence >= min_confidence
        and (e.source_id in kept_ids or e.source_id in sentinels)
        and (e.target_id in kept_ids or e.target_id in sentinels)
    ]
    return result.model_copy(update={"nodes": nodes, "edges": edges})


def parse_extraction_json(
    raw_text: str,
    *,
    chunk_id: str,
    document_id: str,
    project_name: str,
) -> RelationExtractionResult:
    """Parse and validate LLM JSON into RelationExtractionResult."""
    blob = extract_json_object(raw_text)
    if not blob:
        return RelationExtractionResult(
            chunk_id=chunk_id,
            document_id=document_id,
            project_name=project_name,
        )
    try:
        parsed = _RawExtractionJson.model_validate(json.loads(blob))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Relation extractor JSON invalid: %s", exc)
        return RelationExtractionResult(
            chunk_id=chunk_id,
            document_id=document_id,
            project_name=project_name,
        )

    nodes: List[ExtractedNode] = []
    for item in parsed.nodes:
        try:
            nodes.append(ExtractedNode.model_validate(item))
        except ValidationError:
            continue

    edges: List[ExtractedEdge] = []
    for item in parsed.edges:
        try:
            edges.append(ExtractedEdge.model_validate(item))
        except ValidationError:
            continue

    result = RelationExtractionResult(
        chunk_id=chunk_id,
        document_id=document_id,
        project_name=project_name,
        nodes=nodes,
        edges=edges,
    )
    return _filter_extraction_result(result, min_confidence=DEFAULT_MIN_CONFIDENCE)


def validate_extraction_result(
    result: RelationExtractionResult,
    document_type: str,
) -> RelationExtractionResult:
    """Drop nodes/edges that violate ontology endpoint rules."""
    node_type_by_id = {n.node_id: n.node_type for n in result.nodes}
    valid_nodes = list(result.nodes)
    kept_ids = {n.node_id for n in valid_nodes}

    valid_edges: List[ExtractedEdge] = []
    for edge in result.edges:
        src_type = _resolve_endpoint_type(edge.source_id, node_type_by_id, document_type, is_source=True)
        tgt_type = _resolve_endpoint_type(edge.target_id, node_type_by_id, document_type, is_source=False)
        if src_type is None or tgt_type is None:
            continue
        if not validate_edge_endpoints(
            edge.edge_type,
            src_type,
            tgt_type,
            source_document_type=document_type if src_type == NodeType.DOCUMENT else None,
        ):
            continue
        valid_edges.append(edge)

    # Re-filter edges whose endpoints were sentinel-only (chunk/sow/project) — always kept
    final_edges = [
        e
        for e in valid_edges
        if e.source_id in kept_ids
        or e.source_id in (SOW_SOURCE_ID, CHUNK_SOURCE_ID, PROJECT_SOURCE_ID)
    ]
    final_edges = [
        e
        for e in final_edges
        if e.target_id in kept_ids
        or e.target_id in (SOW_SOURCE_ID, CHUNK_SOURCE_ID, PROJECT_SOURCE_ID)
    ]

    return result.model_copy(update={"nodes": valid_nodes, "edges": final_edges})


def _resolve_endpoint_type(
    endpoint_id: str,
    node_type_by_id: Dict[str, NodeType],
    document_type: str,
    *,
    is_source: bool,
) -> Optional[NodeType]:
    if endpoint_id in node_type_by_id:
        return node_type_by_id[endpoint_id]
    if endpoint_id == CHUNK_SOURCE_ID:
        return NodeType.CHUNK
    if endpoint_id == SOW_SOURCE_ID:
        return NodeType.STATEMENT_OF_WORK if document_type == SOW_DOCUMENT_TYPE else NodeType.DOCUMENT
    if endpoint_id == PROJECT_SOURCE_ID:
        return NodeType.PROJECT
    return None


async def extract_relations_from_chunk(
    chunk_text: str,
    *,
    chunk_id: str,
    document_id: str,
    project_name: str,
    document_type: str,
) -> RelationExtractionResult:
    """Full pipeline: Ollama call → parse → validate."""
    allowed = SEMANTIC_NODES_BY_DOCUMENT_TYPE.get(document_type)
    if not allowed:
        return RelationExtractionResult(
            chunk_id=chunk_id,
            document_id=document_id,
            project_name=project_name,
        )
    try:
        raw = await ollama_extract_relations(chunk_text, document_type=document_type)
        result = parse_extraction_json(
            raw,
            chunk_id=chunk_id,
            document_id=document_id,
            project_name=project_name,
        )
        return validate_extraction_result(result, document_type)
    except Exception as exc:
        logger.warning(
            "Relation extraction failed for chunk %s: %s",
            chunk_id,
            _format_extraction_error(
                exc,
                timeout_seconds=settings.GRAPH_EXTRACT_TIMEOUT_SECONDS,
                model=settings.INTENT_LLM_MODEL,
            ),
        )
        return RelationExtractionResult(
            chunk_id=chunk_id,
            document_id=document_id,
            project_name=project_name,
        )
